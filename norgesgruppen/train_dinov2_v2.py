"""Fine-tune DINOv2 ViT for product classification — V2 (competition winner edition).

Improvements over v1:
  1. Focal Loss — handles class imbalance far better than CrossEntropy
  2. Mixup augmentation — proven +2-3% accuracy on fine-grained tasks
  3. Stronger retail-specific augmentation (shadows, occlusion simulation)
  4. Per-category accuracy tracking — identifies weak classes
  5. Cosine annealing with linear warmup
  6. Full backbone unfreezing from epoch 3 (not 5)
  7. 50 epochs default (not 20)
  8. EMA (Exponential Moving Average) model weights

Target: 98%+ val accuracy for competition-winning classification mAP.

Usage:
    python train_dinov2_v2.py
    python train_dinov2_v2.py --epochs 50 --batch-size 48 --lr 2e-4
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict
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
IMAGE_SIZE = 224
ANGLES = ["main", "front", "back", "left", "right", "top", "bottom"]

DATA_DIR = Path("data")
TRAIN_IMAGES_DIR = DATA_DIR / "train" / "images"
ANNOTATIONS_FILE = DATA_DIR / "train" / "annotations.json"
PRODUCT_IMAGES_DIR = DATA_DIR / "NM_NGD_product_images"
METADATA_FILE = PRODUCT_IMAGES_DIR / "metadata.json"
MODELS_DIR = Path("models")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DINOV2_MODEL_CANDIDATES = [
    "vit_base_patch14_dinov2.lvd142m",
    "vit_base_patch14_dinov2",
    "vit_small_patch14_dinov2.lvd142m",
    "vit_small_patch14_dinov2",
]


# ---------------------------------------------------------------------------
# Focal Loss — critical for class imbalance (356 classes, long tail)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Focal Loss: down-weights easy examples, focuses on hard ones.

    For class imbalance: easy-to-classify majority classes get reduced loss,
    forcing the model to learn rare classes better.

    gamma=2.0 is standard. Higher gamma = more focus on hard examples.
    """

    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none",
                                  label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# ---------------------------------------------------------------------------
# Mixup — augments by interpolating between pairs of images+labels
# ---------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.2):
    """Mixup: interpolate between random pairs. Returns mixed inputs + targets."""
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # Ensure lam >= 0.5

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixed targets."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ---------------------------------------------------------------------------
# EMA — Exponential Moving Average for smoother, better-generalizing weights
# ---------------------------------------------------------------------------
class EMAModel:
    """Maintains exponential moving average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + (1 - self.decay) * param.data
                )

    def apply(self, model):
        """Replace model params with EMA params. Returns backup for restore."""
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model, backup):
        """Restore original params from backup."""
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


# ---------------------------------------------------------------------------
# Dataset (same as v1)
# ---------------------------------------------------------------------------
class ProductCropDataset(Dataset):
    def __init__(self, samples, transform):
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
                    max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y)),
                    min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y)),
                )
                img = img.crop(crop_box)
            else:
                img = Image.open(path_str).convert("RGB")

            if img.size[0] < 10 or img.size[1] < 10:
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

            return self.transform(img), cat_id
        except Exception:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


# ---------------------------------------------------------------------------
# Build samples (same as v1)
# ---------------------------------------------------------------------------
def build_samples():
    with open(ANNOTATIONS_FILE) as f:
        coco = json.load(f)

    img_info = {img["id"]: img for img in coco["images"]}
    category_samples = {}

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
    train_samples, val_samples = [], []

    for cat_id, samples in category_samples.items():
        random.shuffle(samples)
        val_count = max(1, int(len(samples) * 0.1))
        val_samples.extend(samples[:val_count])
        train_samples.extend(samples[val_count:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"Categories with samples: {len(category_samples)}")

    # Report class distribution
    train_cats = Counter(s[1] for s in train_samples)
    min_cat = min(train_cats.values())
    max_cat = max(train_cats.values())
    median_cat = sorted(train_cats.values())[len(train_cats) // 2]
    print(f"Class distribution: min={min_cat}, median={median_cat}, max={max_cat}")

    return train_samples, val_samples


def build_class_weights(samples):
    labels = [s[1] for s in samples]
    counts = Counter(labels)
    total = len(labels)
    max_weight = 20.0  # Higher than v1 (was 10) to focus on rare classes
    weights = []
    for _, cat_id in samples:
        w = min(max_weight, total / (len(counts) * counts[cat_id]))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float64)


# ---------------------------------------------------------------------------
# DINOv2 model loading
# ---------------------------------------------------------------------------
def resolve_dinov2_model_name():
    import timm
    available = timm.list_models("*dinov2*")
    print(f"Available DINOv2 models: {available}")
    for candidate in DINOV2_MODEL_CANDIDATES:
        if candidate in available:
            print(f"Selected: {candidate}")
            return candidate
    for candidate in DINOV2_MODEL_CANDIDATES:
        base = candidate.split(".")[0]
        matches = [m for m in available if base in m]
        if matches:
            print(f"Selected (partial): {matches[0]}")
            return matches[0]
    raise RuntimeError(f"No DINOv2 model found. Available: {available}")


def load_dinov2_backbone(device):
    import timm
    model_name = resolve_dinov2_model_name()
    backbone = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=IMAGE_SIZE)
    embed_dim = backbone.num_features
    backbone = backbone.to(device)
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"DINOv2: {model_name}, embed_dim={embed_dim}, params={total_params:,}")
    return backbone, embed_dim, model_name


def freeze_backbone(backbone, freeze_ratio=0.75):
    for param in backbone.parameters():
        param.requires_grad = False
    if hasattr(backbone, "blocks"):
        n = len(backbone.blocks)
        freeze_until = int(n * freeze_ratio)
        for i in range(freeze_until, n):
            for param in backbone.blocks[i].parameters():
                param.requires_grad = True
    for name, param in backbone.named_parameters():
        if any(k in name for k in ["norm.", "head.", "fc_norm.", "cls_token"]):
            param.requires_grad = True
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone: {trainable:,}/{total:,} trainable ({100*trainable/total:.1f}%)")


# ---------------------------------------------------------------------------
# Augmentation — stronger for retail
# ---------------------------------------------------------------------------
def build_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0), ratio=(0.6, 1.4)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.25, p=0.4),
        transforms.RandomAffine(degrees=20, shear=15, scale=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.25)),
    ])


def build_val_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(epochs, batch_size, lr, device, unfreeze_epoch=3, mixup_alpha=0.2):
    print("\n" + "=" * 70)
    print("DINOv2 V2 — Focal Loss + Mixup + EMA + Per-Category Tracking")
    print("=" * 70)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples = build_samples()

    train_ds = ProductCropDataset(train_samples, build_train_transform())
    val_ds = ProductCropDataset(val_samples, build_val_transform())

    sample_weights = build_class_weights(train_samples)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_samples), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    backbone, embed_dim, model_name = load_dinov2_backbone(device)
    freeze_backbone(backbone)

    # Classification head with LayerNorm + Dropout
    head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(0.2),
        nn.Linear(embed_dim, NUM_CATEGORIES),
    ).to(device)

    backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    head_params = list(head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ], weight_decay=1e-4)

    # Cosine annealing with linear warmup
    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ema_backbone = None
    ema_head = None

    best_val_acc = 0.0
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # Unfreeze all at epoch 3
        if epoch == unfreeze_epoch:
            print(f"\n--- Unfreezing ALL backbone layers at epoch {epoch+1} ---")
            for param in backbone.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": list(backbone.parameters()), "lr": lr * 0.01},
                {"params": head_params, "lr": lr * 0.3},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch, eta_min=1e-7,
            )
            # Init EMA after unfreezing
            ema_backbone = EMAModel(backbone, decay=0.999)
            ema_head = EMAModel(head, decay=0.999)

        # Train
        backbone.train()
        head.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixup
            if mixup_alpha > 0 and epoch >= warmup_epochs:
                imgs, targets_a, targets_b, lam = mixup_data(imgs, labels, mixup_alpha)
            else:
                targets_a = labels
                targets_b = labels
                lam = 1.0

            with torch.amp.autocast("cuda", enabled=use_amp):
                features = backbone(imgs)
                logits = head(features)
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + head_params, max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            # Update EMA
            if ema_backbone is not None:
                ema_backbone.update(backbone)
                ema_head.update(head)

            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)

            if (batch_idx + 1) % 100 == 0:
                print(f"  batch {batch_idx+1}: loss={loss.item():.4f}, "
                      f"acc={train_correct/train_total:.3f}")

        scheduler.step()
        train_acc = train_correct / max(1, train_total)
        avg_loss = train_loss / max(1, train_total)

        # Validate (use EMA weights if available)
        backbone.eval()
        head.eval()

        if ema_backbone is not None:
            backup_b = ema_backbone.apply(backbone)
            backup_h = ema_head.apply(head)

        val_correct = 0
        val_total = 0
        val_top5_correct = 0
        per_cat_correct = defaultdict(int)
        per_cat_total = defaultdict(int)

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    features = backbone(imgs)
                    logits = head(features)

                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)

                _, top5 = logits.topk(5, dim=1)
                val_top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

                for i in range(len(labels)):
                    cat = labels[i].item()
                    per_cat_total[cat] += 1
                    if preds[i].item() == cat:
                        per_cat_correct[cat] += 1

        val_acc = val_correct / max(1, val_total)
        val_top5 = val_top5_correct / max(1, val_total)
        current_lr = optimizer.param_groups[0]["lr"]

        # Per-category analysis
        cat_accs = []
        worst_cats = []
        for cat in sorted(per_cat_total.keys()):
            acc = per_cat_correct[cat] / per_cat_total[cat] if per_cat_total[cat] > 0 else 0
            cat_accs.append(acc)
            if acc < 0.5 and per_cat_total[cat] >= 2:
                worst_cats.append((cat, acc, per_cat_total[cat]))

        mean_per_cat = np.mean(cat_accs) if cat_accs else 0
        cats_above_90 = sum(1 for a in cat_accs if a >= 0.9)
        cats_below_50 = sum(1 for a in cat_accs if a < 0.5)

        print(
            f"Epoch {epoch+1}/{epochs} — "
            f"loss: {avg_loss:.4f}, train: {train_acc:.3f}, "
            f"val: {val_acc:.3f}, top5: {val_top5:.3f}, "
            f"mean_per_cat: {mean_per_cat:.3f}, lr: {current_lr:.2e}"
        )
        print(
            f"  Categories: {cats_above_90}/{len(cat_accs)} >90%, "
            f"{cats_below_50}/{len(cat_accs)} <50%"
        )
        if worst_cats and epoch % 5 == 0:
            print(f"  Worst categories: {worst_cats[:10]}")

        if ema_backbone is not None:
            ema_backbone.restore(backbone, backup_b)
            ema_head.restore(head, backup_h)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            # Save with EMA weights
            if ema_backbone is not None:
                backup_b = ema_backbone.apply(backbone)
                backup_h = ema_head.apply(head)

            state = {
                "backbone": backbone.state_dict(),
                "classifier_head": head.state_dict(),
                "embed_dim": embed_dim,
                "num_categories": NUM_CATEGORIES,
                "model_name": model_name,
            }
            torch.save(state, str(MODELS_DIR / "dinov2_classifier_weights.pt"))
            print(f"  -> Saved best model (val_acc={val_acc:.3f}, mean_per_cat={mean_per_cat:.3f})")

            if ema_backbone is not None:
                ema_backbone.restore(backbone, backup_b)
                ema_head.restore(head, backup_h)

            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after {patience} epochs without improvement")
                break

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {MODELS_DIR / 'dinov2_classifier_weights.pt'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--unfreeze-epoch", type=int, default=3)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"V2 Config: epochs={args.epochs}, bs={args.batch_size}, "
          f"lr={args.lr}, unfreeze={args.unfreeze_epoch}, "
          f"mixup={args.mixup_alpha}, device={args.device}")

    train(args.epochs, args.batch_size, args.lr, args.device,
          args.unfreeze_epoch, args.mixup_alpha)


if __name__ == "__main__":
    main()
