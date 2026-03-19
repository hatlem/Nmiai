"""Fine-tune DINOv2 ViT for product classification.

Two data sources:
  1. Crops from training annotations (22,700 bounding boxes -> cropped images)
  2. Product reference images (327 products x 7 angles = ~2,289 images)

Two training modes:
  a. ArcFace metric learning — for embedding matching (nearest-neighbor classification)
  b. Supervised CrossEntropy — direct 356-class classification with linear head

DINOv2 produces superior visual features compared to EfficientNet thanks to
self-supervised pretraining on 142M images (LVD-142M dataset).

Requirements: torch, torchvision, timm==0.9.12, PIL

Usage:
    python train_dinov2.py
    python train_dinov2.py --epochs 30 --batch-size 32
    python train_dinov2.py --mode arcface --epochs 20
    python train_dinov2.py --mode crossentropy --epochs 15
    python train_dinov2.py --mode both --epochs 20
"""

import argparse
import json
import math
import random
from collections import Counter
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
IMAGE_SIZE = 224  # Use 224 to match inference & save GPU memory (DINOv2 handles any resolution)
ANGLES = ["main", "front", "back", "left", "right", "top", "bottom"]

DATA_DIR = Path("data")
TRAIN_IMAGES_DIR = DATA_DIR / "train" / "images"
ANNOTATIONS_FILE = DATA_DIR / "train" / "annotations.json"
PRODUCT_IMAGES_DIR = DATA_DIR / "NM_NGD_product_images"
METADATA_FILE = PRODUCT_IMAGES_DIR / "metadata.json"
MODELS_DIR = Path("models")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# DINOv2 model candidates (in order of preference)
DINOV2_MODEL_CANDIDATES = [
    "vit_base_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_small_patch14_dinov2",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ProductCropDataset(Dataset):
    """Dataset combining annotation crops and reference product images."""

    def __init__(self, samples: list[tuple[str, int]], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_str, cat_id = self.samples[idx]

        try:
            if "|" in path_str:
                img_path, bbox_str = path_str.split("|")
                x1, y1, x2, y2 = map(float, bbox_str.split(","))
                img = Image.open(img_path).convert("RGB")
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

            if img.size[0] < 10 or img.size[1] < 10:
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

            img = self.transform(img)
            return img, cat_id
        except Exception:
            fallback_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(fallback_idx)


# ---------------------------------------------------------------------------
# ArcFace head for metric learning
# ---------------------------------------------------------------------------
class ArcFaceHead(nn.Module):
    """Additive Angular Margin Loss (ArcFace) for discriminative embeddings."""

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

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)
        cosine = embeddings @ weights.T

        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
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
    """
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}
    category_samples: dict[int, list[tuple[str, int]]] = {}

    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        cat_id = ann["category_id"]
        if cat_id == UNKNOWN_CATEGORY_ID:
            continue

        img = img_info[ann["image_id"]]
        img_path = str(TRAIN_IMAGES_DIR / img["file_name"])
        bx, by, bw, bh = ann["bbox"]
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh

        if bw < 10 or bh < 10:
            continue

        path_str = f"{img_path}|{x1},{y1},{x2},{y2}"
        category_samples.setdefault(cat_id, []).append((path_str, cat_id))

    crop_count = sum(len(v) for v in category_samples.values())
    print(f"Annotation crops: {crop_count} across {len(category_samples)} categories")

    # Product reference images
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

    # Stratified split: 90% train, 10% val
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
    labels = [s[1] for s in samples]
    counts = Counter(labels)
    total = len(labels)
    max_weight = 10.0
    weights = []
    for _, cat_id in samples:
        w = min(max_weight, total / (len(counts) * counts[cat_id]))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float64)


# ---------------------------------------------------------------------------
# DINOv2 model loading
# ---------------------------------------------------------------------------
def resolve_dinov2_model_name() -> str:
    """Find the best available DINOv2 model in timm."""
    import timm

    available = timm.list_models("*dinov2*")
    print(f"Available DINOv2 models in timm: {available}")

    for candidate in DINOV2_MODEL_CANDIDATES:
        if candidate in available:
            print(f"Selected model: {candidate}")
            return candidate

    # If no exact match, try partial matching
    for candidate in DINOV2_MODEL_CANDIDATES:
        base_name = candidate.split(".")[0]
        matches = [m for m in available if base_name in m]
        if matches:
            selected = matches[0]
            print(f"Selected model (partial match): {selected}")
            return selected

    raise RuntimeError(
        f"No DINOv2 model found in timm. Available models: {available}. "
        f"Tried candidates: {DINOV2_MODEL_CANDIDATES}"
    )


def load_dinov2_backbone(device: str) -> tuple[nn.Module, int, str]:
    """Load DINOv2 backbone in feature extraction mode.

    Returns (backbone, embed_dim, model_name).
    """
    import timm

    model_name = resolve_dinov2_model_name()
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=IMAGE_SIZE)
    embed_dim = backbone.num_features
    backbone = backbone.to(device)

    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"DINOv2 backbone: {model_name}, embed_dim={embed_dim}, params={total_params:,}")

    return backbone, embed_dim, model_name


def freeze_dinov2_early_layers(backbone: nn.Module, freeze_ratio: float = 0.75):
    """Freeze first 75% of transformer blocks, train last 25% + head.

    DINOv2 ViT-B has 12 transformer blocks (backbone.blocks[0..11]).
    With freeze_ratio=0.75, we freeze blocks 0-8 and train blocks 9-11.
    """
    # Freeze everything first
    for param in backbone.parameters():
        param.requires_grad = False

    # Find transformer blocks
    if hasattr(backbone, "blocks"):
        num_blocks = len(backbone.blocks)
        freeze_until = int(num_blocks * freeze_ratio)
        print(f"ViT blocks: {num_blocks} total, freezing first {freeze_until}, training last {num_blocks - freeze_until}")

        for i in range(freeze_until, num_blocks):
            for param in backbone.blocks[i].parameters():
                param.requires_grad = True

    # Always unfreeze norm layer, head, fc_norm
    for name, param in backbone.named_parameters():
        if any(k in name for k in ["norm.", "head.", "fc_norm.", "cls_token"]):
            param.requires_grad = True

    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone: {trainable:,} / {total:,} parameters trainable ({100 * trainable / total:.1f}%)")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
def build_train_transform() -> transforms.Compose:
    """Strong augmentation pipeline for DINOv2."""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
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
# Training loops
# ---------------------------------------------------------------------------
def train_arcface(
    epochs: int, batch_size: int, lr: float, device: str, unfreeze_epoch: int = 5,
):
    """Train DINOv2 with ArcFace metric learning loss."""
    print("\n" + "=" * 70)
    print("TRAINING MODE: ArcFace (metric learning for embedding matching)")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples = build_samples()

    train_ds = ProductCropDataset(train_samples, build_train_transform())
    val_ds = ProductCropDataset(val_samples, build_val_transform())

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

    backbone, embed_dim, model_name = load_dinov2_backbone(device)
    freeze_dinov2_early_layers(backbone)

    arcface = ArcFaceHead(embed_dim, NUM_CATEGORIES, s=30.0, m=0.3).to(device)

    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(arcface.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 2), T_mult=2, eta_min=1e-6,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 7

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            print(f"\n--- Unfreezing all backbone layers at epoch {epoch + 1} ---")
            for param in backbone.parameters():
                param.requires_grad = True
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
                print(f"  batch {batch_idx + 1}: loss={loss.item():.4f}, acc={batch_acc:.3f}")

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

                weights = F.normalize(arcface.weight, dim=1)
                cosine = embeddings @ weights.T
                preds = cosine.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

                _, top5 = cosine.topk(5, dim=1)
                val_top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        val_acc = val_correct / max(1, val_total)
        val_top5 = val_top5_correct / max(1, val_total)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} — "
            f"loss: {avg_loss:.4f}, train_acc: {train_acc:.3f}, "
            f"val_acc: {val_acc:.3f}, val_top5: {val_top5:.3f}, lr: {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            weights_path = MODELS_DIR / "dinov2_embeddings_weights.pt"
            torch.save(backbone.state_dict(), str(weights_path))
            print(f"  -> Saved best ArcFace model (val_acc={val_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {patience} epochs without improvement")
                break

    print(f"\nBest ArcFace val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {MODELS_DIR / 'dinov2_embeddings_weights.pt'}")

    # Rebuild embeddings with fine-tuned model
    best_state = torch.load(str(MODELS_DIR / "dinov2_embeddings_weights.pt"), map_location=device)
    backbone.load_state_dict(best_state)

    print("\nRebuilding DINOv2 product embeddings with fine-tuned model...")
    rebuild_embeddings(backbone, embed_dim, model_name, device)

    return backbone, embed_dim, model_name


def train_crossentropy(
    epochs: int, batch_size: int, lr: float, device: str, unfreeze_epoch: int = 5,
):
    """Train DINOv2 with CrossEntropy loss for direct classification."""
    print("\n" + "=" * 70)
    print("TRAINING MODE: CrossEntropy (direct 356-class classification)")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_samples, val_samples = build_samples()

    train_ds = ProductCropDataset(train_samples, build_train_transform())
    val_ds = ProductCropDataset(val_samples, build_val_transform())

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

    backbone, embed_dim, model_name = load_dinov2_backbone(device)
    freeze_dinov2_early_layers(backbone)

    # Linear classification head
    classifier_head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(0.3),
        nn.Linear(embed_dim, NUM_CATEGORIES),
    ).to(device)

    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(classifier_head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs // 2), T_mult=2, eta_min=1e-6,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0.0
    patience_counter = 0
    patience = 7

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            print(f"\n--- Unfreezing all backbone layers at epoch {epoch + 1} ---")
            for param in backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": list(backbone.parameters()), "lr": lr * 0.01},
                {"params": head_params, "lr": lr * 0.5},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch, eta_min=1e-6,
            )

        # Train
        backbone.train()
        classifier_head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                features = backbone(imgs)
                logits = classifier_head(features)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + head_params, max_norm=1.0,
            )
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += imgs.size(0)

            if (batch_idx + 1) % 50 == 0:
                batch_acc = train_correct / train_total
                print(f"  batch {batch_idx + 1}: loss={loss.item():.4f}, acc={batch_acc:.3f}")

        scheduler.step()

        train_acc = train_correct / max(1, train_total)
        avg_loss = train_loss / max(1, train_total)

        # Validate
        backbone.eval()
        classifier_head.eval()
        val_correct = 0
        val_total = 0
        val_top5_correct = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    features = backbone(imgs)
                    logits = classifier_head(features)

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

                _, top5 = logits.topk(5, dim=1)
                val_top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        val_acc = val_correct / max(1, val_total)
        val_top5 = val_top5_correct / max(1, val_total)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} — "
            f"loss: {avg_loss:.4f}, train_acc: {train_acc:.3f}, "
            f"val_acc: {val_acc:.3f}, val_top5: {val_top5:.3f}, lr: {current_lr:.2e}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save backbone + head together
            state = {
                "backbone": backbone.state_dict(),
                "classifier_head": classifier_head.state_dict(),
                "embed_dim": embed_dim,
                "num_categories": NUM_CATEGORIES,
                "model_name": model_name,
            }
            weights_path = MODELS_DIR / "dinov2_classifier_weights.pt"
            torch.save(state, str(weights_path))
            print(f"  -> Saved best CrossEntropy model (val_acc={val_acc:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {patience} epochs without improvement")
                break

    print(f"\nBest CrossEntropy val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {MODELS_DIR / 'dinov2_classifier_weights.pt'}")

    return backbone, embed_dim, model_name


# ---------------------------------------------------------------------------
# Embedding rebuild (after ArcFace training)
# ---------------------------------------------------------------------------
def rebuild_embeddings(backbone: nn.Module, embed_dim: int, model_name: str, device: str):
    """Rebuild product_embeddings.npy using the fine-tuned DINOv2 backbone."""
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    if not METADATA_FILE.exists():
        print("No metadata.json found, skipping embedding rebuild")
        return

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    name_to_cat_id = {cat["name"]: cat["id"] for cat in coco["categories"]}
    transform = build_val_transform()

    backbone.eval()
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

    emb_path = MODELS_DIR / "dinov2_product_embeddings.npy"
    np.save(str(emb_path), embeddings)
    print(f"Rebuilt DINOv2 embeddings for {processed} products, shape={embeddings.shape}")

    config = {
        "model_name": model_name,
        "embedding_dim": embed_dim,
        "num_categories": NUM_CATEGORIES,
        "unknown_category_id": UNKNOWN_CATEGORY_ID,
        "image_size": IMAGE_SIZE,
        "l2_normalized": True,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "angles_used": ANGLES,
        "fine_tuned": True,
        "backbone": "dinov2",
    }
    config_path = MODELS_DIR / "dinov2_embedding_config.json"
    with open(str(config_path), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")


# ---------------------------------------------------------------------------
# Consolidate into single file for submission (max 3 weight files rule)
# ---------------------------------------------------------------------------
def consolidate_weights():
    """Pack classifier weights + embeddings into a single .pt file.

    Submission gets: best.pt (detector) + dinov2_all.pt (classifier+embeddings) = 2 weight files.
    This leaves 1 slot free for a secondary detector if needed.
    """
    combined = {}

    # Classifier weights (supervised head)
    cls_path = MODELS_DIR / "dinov2_classifier_weights.pt"
    if cls_path.exists():
        combined["classifier"] = torch.load(str(cls_path), map_location="cpu")
        print(f"[CONSOLIDATE] Added classifier weights from {cls_path.name}")

    # Embedding weights (arcface-trained backbone)
    emb_weights_path = MODELS_DIR / "dinov2_embeddings_weights.pt"
    if emb_weights_path.exists():
        combined["embedding_backbone"] = torch.load(str(emb_weights_path), map_location="cpu")
        print(f"[CONSOLIDATE] Added embedding backbone from {emb_weights_path.name}")

    # Product embeddings (.npy -> tensor inside .pt)
    emb_path = MODELS_DIR / "dinov2_product_embeddings.npy"
    if emb_path.exists():
        embeddings = np.load(str(emb_path))
        combined["product_embeddings"] = torch.from_numpy(embeddings)
        print(f"[CONSOLIDATE] Added product embeddings shape={embeddings.shape}")

    # Config
    for cfg_name in ["dinov2_embedding_config.json", "embedding_config.json"]:
        cfg_path = MODELS_DIR / cfg_name
        if cfg_path.exists():
            with open(str(cfg_path)) as f:
                combined["config"] = json.load(f)
            print(f"[CONSOLIDATE] Added config from {cfg_name}")
            break

    if combined:
        out_path = MODELS_DIR / "dinov2_all.pt"
        torch.save(combined, str(out_path))
        size_mb = out_path.stat().st_size / 1024 / 1024
        print(f"[CONSOLIDATE] Saved consolidated file: {out_path} ({size_mb:.1f} MB)")
        print(f"[CONSOLIDATE] Submission: best.pt + dinov2_all.pt = 2 weight files (1 slot free)")
    else:
        print("[CONSOLIDATE] No weights found to consolidate")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 for product classification")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (DINOv2 is larger, use smaller batch)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--unfreeze-epoch", type=int, default=5,
                        help="Epoch at which to unfreeze all backbone layers")
    parser.add_argument("--mode", choices=["arcface", "crossentropy", "both"], default="both",
                        help="Training mode: arcface, crossentropy, or both")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Config: epochs={args.epochs}, batch_size={args.batch_size}, "
          f"lr={args.lr}, unfreeze_epoch={args.unfreeze_epoch}, "
          f"mode={args.mode}, device={args.device}")

    if args.mode in ("arcface", "both"):
        train_arcface(args.epochs, args.batch_size, args.lr, args.device, args.unfreeze_epoch)

    if args.mode in ("crossentropy", "both"):
        train_crossentropy(args.epochs, args.batch_size, args.lr, args.device, args.unfreeze_epoch)

    # Consolidate into a single .pt file for submission (max 3 weight files rule)
    consolidate_weights()

    print("\nDone! Weights saved to models/ directory.")


if __name__ == "__main__":
    main()
