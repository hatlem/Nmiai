"""Fine-tune EfficientNet-B3 for product classification.

Two data sources:
  1. Crops from training annotations (22,700 bounding boxes -> cropped images)
  2. Product reference images (327 products x 7 angles = ~2,289 images)

Trains with ArcFace metric learning loss for better embedding quality.
Freezes early layers, trains last 2 blocks + new classification head.
After training, re-builds product_embeddings.npy with the fine-tuned model.

Requirements: torch, torchvision, timm==0.9.12, PIL

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 30 --batch-size 64
    python train_classifier.py --epochs 20 --batch-size 32 --device cpu
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_CATEGORIES = 356
UNKNOWN_CATEGORY_ID = 355
MODEL_NAME = "efficientnet_b3"
IMAGE_SIZE = 300
ANGLES = ["main", "front", "back", "left", "right", "top", "bottom"]

DATA_DIR = Path("data")
TRAIN_IMAGES_DIR = DATA_DIR / "train" / "images"
ANNOTATIONS_FILE = DATA_DIR / "train" / "annotations.json"
PRODUCT_IMAGES_DIR = DATA_DIR / "NM_NGD_product_images"
METADATA_FILE = PRODUCT_IMAGES_DIR / "metadata.json"
MODELS_DIR = Path("models")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ProductCropDataset(Dataset):
    """Dataset combining annotation crops and reference product images."""

    def __init__(self, samples: list[tuple[str, int]], transform):
        """
        Args:
            samples: list of (image_path, category_id) tuples.
                     For crops: path is "img_path|x1,y1,x2,y2"
                     For product images: path is direct file path
            transform: torchvision transform
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_str, cat_id = self.samples[idx]

        try:
            if "|" in path_str:
                # Crop from training image
                img_path, bbox_str = path_str.split("|")
                x1, y1, x2, y2 = map(float, bbox_str.split(","))
                img = Image.open(img_path).convert("RGB")
                # Add small padding around crop (5%) for context
                w, h = img.size
                pad_x = (x2 - x1) * 0.05
                pad_y = (y2 - y1) * 0.05
                crop_box = (
                    max(0, int(x1 - pad_x)),
                    max(0, int(y1 - pad_y)),
                    min(w, int(x2 + pad_x)),
                    min(h, int(y2 + pad_y)),
                )
                img = img.crop(crop_box)
            else:
                img = Image.open(path_str).convert("RGB")

            # Ensure minimum size to avoid transform errors
            if img.size[0] < 10 or img.size[1] < 10:
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

            img = self.transform(img)
            return img, cat_id
        except Exception:
            # Return a random valid sample on error
            fallback_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(fallback_idx)


# ---------------------------------------------------------------------------
# ArcFace head for metric learning
# ---------------------------------------------------------------------------
class ArcFaceHead(nn.Module):
    """Additive Angular Margin Loss (ArcFace) for discriminative embeddings.

    Pushes intra-class embeddings closer and inter-class further apart
    in angular space, producing better features for nearest-neighbor matching.
    """

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        s: float = 30.0,
        m: float = 0.3,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute for numerical stability
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        cosine = embeddings @ weights.T  # (B, num_classes)

        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).float()
        logits = cosine * (1 - one_hot) + phi * one_hot

        return logits * self.s


# ---------------------------------------------------------------------------
# Build samples
# ---------------------------------------------------------------------------
def build_samples() -> tuple[list, list]:
    """Build training samples from annotations + product images.

    Returns (train_samples, val_samples) as lists of (path_str, category_id).
    Stratified split: ensures each category appears in both train and val.
    """
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    # Build image info lookup
    img_info = {img["id"]: img for img in coco["images"]}

    # Crop samples from annotations, grouped by category
    category_samples: dict[int, list[tuple[str, int]]] = {}

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cat_id = ann["category_id"]
        if cat_id == UNKNOWN_CATEGORY_ID:
            continue  # Skip unknown_product

        img = img_info[ann["image_id"]]
        img_path = str(TRAIN_IMAGES_DIR / img["file_name"])
        bx, by, bw, bh = ann["bbox"]
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh

        # Skip tiny boxes
        if bw < 10 or bh < 10:
            continue

        path_str = f"{img_path}|{x1},{y1},{x2},{y2}"
        category_samples.setdefault(cat_id, []).append((path_str, cat_id))

    crop_count = sum(len(v) for v in category_samples.values())
    print(f"Annotation crops: {crop_count} across {len(category_samples)} categories")

    # Product reference image samples
    ref_count = 0
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            metadata = json.load(f)

        name_to_cat_id = {cat["name"]: cat["id"] for cat in coco["categories"]}

        for product in metadata["products"]:
            product_code = product["product_code"]
            product_name = product["product_name"]
            cat_id = name_to_cat_id.get(product_name)
            if cat_id is None:
                for name, cid in name_to_cat_id.items():
                    if name.lower() == product_name.lower():
                        cat_id = cid
                        break
            if cat_id is None or cat_id == UNKNOWN_CATEGORY_ID:
                continue

            product_dir = PRODUCT_IMAGES_DIR / product_code
            for angle in ANGLES:
                img_path = product_dir / f"{angle}.jpg"
                if img_path.exists():
                    category_samples.setdefault(cat_id, []).append((str(img_path), cat_id))
                    ref_count += 1

    print(f"Reference images: {ref_count}")

    # Stratified split: 90% train, 10% val per category
    random.seed(42)
    train_samples = []
    val_samples = []

    for cat_id, samples in category_samples.items():
        random.shuffle(samples)
        val_count = max(1, int(len(samples) * 0.1))
        val_samples.extend(samples[:val_count])
        train_samples.extend(samples[val_count:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"Categories with samples: {len(category_samples)}")
    return train_samples, val_samples


def build_class_weights(samples: list[tuple[str, int]]) -> torch.Tensor:
    """Build per-sample weights for balanced sampling (inverse class frequency)."""
    from collections import Counter

    labels = [s[1] for s in samples]
    counts = Counter(labels)
    total = len(labels)
    # Inverse frequency, capped to avoid extreme weights
    max_weight = 10.0
    weights = []
    for _, cat_id in samples:
        w = min(max_weight, total / (len(counts) * counts[cat_id]))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Layer freezing
# ---------------------------------------------------------------------------
def freeze_backbone_early_layers(backbone: nn.Module):
    """Freeze all layers except last 2 blocks of EfficientNet-B3.

    EfficientNet-B3 has blocks 0-6 (7 blocks total).
    We freeze blocks 0-4 and train blocks 5-6 + classifier/head layers.
    """
    # Freeze everything first
    for param in backbone.parameters():
        param.requires_grad = False

    # Unfreeze last 2 blocks (blocks[5] and blocks[6])
    for name, param in backbone.named_parameters():
        if "blocks.5." in name or "blocks.6." in name:
            param.requires_grad = True
        # Also unfreeze batch norm in final conv and head
        if "conv_head" in name or "bn2" in name or "classifier" in name:
            param.requires_grad = True
        # Global pool / head layers
        if "global_pool" in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone: {trainable:,} / {total:,} parameters trainable ({100*trainable/total:.1f}%)")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def build_train_transform() -> transforms.Compose:
    """Strong augmentation pipeline for shelf crop images."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15)),
    ])


def build_val_transform() -> transforms.Compose:
    """Deterministic transform for validation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(epochs: int, batch_size: int, lr: float, device: str, unfreeze_epoch: int = 5):
    import timm

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Build data
    train_samples, val_samples = build_samples()

    train_ds = ProductCropDataset(train_samples, build_train_transform())
    val_ds = ProductCropDataset(val_samples, build_val_transform())

    # Balanced sampling to handle class imbalance
    sample_weights = build_class_weights(train_samples)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_samples), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True,
    )

    # Model: EfficientNet-B3 backbone (feature extractor)
    print(f"Loading pretrained {MODEL_NAME}...")
    backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    embed_dim = backbone.num_features
    backbone = backbone.to(device)

    # Freeze early layers — train only last 2 blocks + head
    freeze_backbone_early_layers(backbone)

    # ArcFace classification head
    arcface = ArcFaceHead(embed_dim, NUM_CATEGORIES, s=30.0, m=0.3).to(device)

    # Separate parameter groups: backbone (lower lr) vs head (full lr)
    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(arcface.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},  # Lower lr for pretrained layers
        {"params": head_params, "lr": lr},
    ], weight_decay=1e-4)

    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 2), T_mult=2, eta_min=1e-6,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Mixed precision training
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 7  # Early stopping patience

    for epoch in range(epochs):
        # Unfreeze all backbone layers after warmup
        if epoch == unfreeze_epoch:
            print(f"\n--- Unfreezing all backbone layers at epoch {epoch+1} ---")
            for param in backbone.parameters():
                param.requires_grad = True
            # Reset optimizer with all params
            optimizer = torch.optim.AdamW([
                {"params": list(backbone.parameters()), "lr": lr * 0.01},
                {"params": head_params, "lr": lr * 0.5},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch, eta_min=1e-6,
            )

        # Train
        backbone.train()
        arcface.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                embeddings = backbone(imgs)
                logits = arcface(embeddings, labels)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(arcface.parameters()), max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += imgs.size(0)

            if (batch_idx + 1) % 50 == 0:
                batch_acc = train_correct / train_total
                print(f"  batch {batch_idx+1}: loss={loss.item():.4f}, acc={batch_acc:.3f}")

        scheduler.step()

        train_acc = train_correct / max(1, train_total)
        avg_loss = train_loss / max(1, train_total)

        # Validate
        backbone.eval()
        val_correct = 0
        val_total = 0
        val_top5_correct = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    embeddings = backbone(imgs)
                    embeddings = F.normalize(embeddings, dim=1)

                # For validation, use cosine similarity (no arcface margin)
                weights = F.normalize(arcface.weight, dim=1)
                cosine = embeddings @ weights.T
                preds = cosine.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

                # Top-5 accuracy
                _, top5 = cosine.topk(5, dim=1)
                val_top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        val_acc = val_correct / max(1, val_total)
        val_top5 = val_top5_correct / max(1, val_total)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{epochs} — "
            f"loss: {avg_loss:.4f}, train_acc: {train_acc:.3f}, "
            f"val_acc: {val_acc:.3f}, val_top5: {val_top5:.3f}, lr: {current_lr:.2e}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights_path = MODELS_DIR / "efficientnet_b3_weights.pt"
            torch.save(backbone.state_dict(), str(weights_path))
            print(f"  -> Saved best model (val_acc={val_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {patience} epochs without improvement")
                break

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {MODELS_DIR / 'efficientnet_b3_weights.pt'}")

    # Load best weights for embedding rebuild
    best_state = torch.load(str(MODELS_DIR / "efficientnet_b3_weights.pt"), map_location=device)
    backbone.load_state_dict(best_state)

    # Re-build embeddings with fine-tuned model
    print("\nRebuilding product embeddings with fine-tuned model...")
    rebuild_embeddings(backbone, device)


def rebuild_embeddings(backbone: nn.Module, device: str):
    """Rebuild product_embeddings.npy using the fine-tuned backbone."""
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    if not METADATA_FILE.exists():
        print("No metadata.json found, skipping embedding rebuild")
        return

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    name_to_cat_id = {cat["name"]: cat["id"] for cat in coco["categories"]}

    transform = build_val_transform()

    # Get embedding dim
    backbone.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
        embed_dim = backbone(dummy).shape[1]

    embeddings = np.zeros((NUM_CATEGORIES, embed_dim), dtype=np.float32)
    processed = 0

    for product in metadata["products"]:
        product_code = product["product_code"]
        product_name = product["product_name"]
        cat_id = name_to_cat_id.get(product_name)
        if cat_id is None:
            for name, cid in name_to_cat_id.items():
                if name.lower() == product_name.lower():
                    cat_id = cid
                    break
        if cat_id is None or cat_id == UNKNOWN_CATEGORY_ID:
            continue

        product_dir = PRODUCT_IMAGES_DIR / product_code
        angle_embeddings = []

        for angle in ANGLES:
            img_path = product_dir / f"{angle}.jpg"
            if not img_path.exists():
                continue
            try:
                img = Image.open(str(img_path)).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = backbone(tensor).squeeze(0).cpu().numpy()
                angle_embeddings.append(emb)
            except Exception:
                continue

        if angle_embeddings:
            avg_emb = np.mean(angle_embeddings, axis=0)
            norm = np.linalg.norm(avg_emb)
            if norm > 0:
                avg_emb = avg_emb / norm
            embeddings[cat_id] = avg_emb
            processed += 1

    np.save(str(MODELS_DIR / "product_embeddings.npy"), embeddings)
    print(f"Rebuilt embeddings for {processed} products, shape={embeddings.shape}")

    # Save config
    config = {
        "model_name": MODEL_NAME,
        "embedding_dim": embed_dim,
        "num_categories": NUM_CATEGORIES,
        "unknown_category_id": UNKNOWN_CATEGORY_ID,
        "l2_normalized": True,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "angles_used": ANGLES,
        "fine_tuned": True,
    }
    with open(str(MODELS_DIR / "embedding_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {MODELS_DIR / 'embedding_config.json'}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B3 classifier")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--unfreeze-epoch", type=int, default=5,
                        help="Epoch at which to unfreeze all backbone layers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, unfreeze_epoch={args.unfreeze_epoch}, device={args.device}")

    train(args.epochs, args.batch_size, args.lr, args.device, args.unfreeze_epoch)


if __name__ == "__main__":
    main()
