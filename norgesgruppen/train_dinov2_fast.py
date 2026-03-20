"""DINOv2 classifier training — FAST version using precropped images.

Run precrop.py first, then this script.

Key improvements over v1:
  - Reads small pre-cropped JPGs instead of cropping from full images on-the-fly
  - Focal Loss for class imbalance
  - Mixup augmentation
  - EMA model weights
  - Per-category accuracy tracking
  - 50 epochs with patience=10

Usage:
    python precrop.py                    # First: extract crops (~30s)
    python train_dinov2_fast.py          # Then: train (~20-30 min on T4)
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

NUM_CATEGORIES = 356
UNKNOWN_CATEGORY_ID = 355
IMAGE_SIZE = 224
ANGLES = ["main", "front", "back", "left", "right", "top", "bottom"]

DATA_DIR = Path("data")
PRODUCT_IMAGES_DIR = DATA_DIR / "NM_NGD_product_images"
METADATA_FILE = PRODUCT_IMAGES_DIR / "metadata.json"
ANNOTATIONS_FILE = DATA_DIR / "train" / "annotations.json"
CROP_MANIFEST = DATA_DIR / "crop_manifest.json"
MODELS_DIR = Path("models")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none",
                             label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class FastCropDataset(Dataset):
    """Reads pre-cropped images — much faster than cropping on-the-fly."""

    def __init__(self, samples, transform):
        self.samples = samples  # list of (path_str, category_id)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path_str, cat_id = self.samples[idx]
        try:
            img = Image.open(path_str).convert("RGB")
            return self.transform(img), cat_id
        except Exception:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))


def build_samples():
    """Build train/val samples from precropped manifest + product reference images."""
    with open(CROP_MANIFEST) as f:
        manifest = json.load(f)

    category_samples = {}
    for item in manifest:
        cat_id = item["category_id"]
        category_samples.setdefault(cat_id, []).append((item["path"], cat_id))

    crop_count = sum(len(v) for v in category_samples.values())
    print(f"Precropped samples: {crop_count} across {len(category_samples)} categories")

    # Add product reference images
    ref_count = 0
    if METADATA_FILE.exists() and ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            coco = json.load(f)
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

    print(f"Reference images added: {ref_count}")

    # Stratified split
    random.seed(42)
    train_samples, val_samples = [], []

    for cat_id, samples in category_samples.items():
        random.shuffle(samples)
        val_count = max(1, int(len(samples) * 0.1))
        val_samples.extend(samples[:val_count])
        train_samples.extend(samples[val_count:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    train_cats = Counter(s[1] for s in train_samples)
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, "
          f"Categories: {len(train_cats)}, "
          f"min/med/max samples: {min(train_cats.values())}/{sorted(train_cats.values())[len(train_cats)//2]}/{max(train_cats.values())}")

    return train_samples, val_samples


def build_class_weights(samples):
    labels = [s[1] for s in samples]
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for _, cat_id in samples:
        w = min(20.0, total / (len(counts) * counts[cat_id]))
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float64)


def build_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0), ratio=(0.6, 1.4)),
        transforms.RandomHorizontalFlip(p=0.5),
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


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n] = self.decay * self.shadow[n] + (1 - self.decay) * p.data

    def apply(self, model):
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


def train(epochs, batch_size, lr, device, unfreeze_epoch=3, mixup_alpha=0.2):
    import timm

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_samples, val_samples = build_samples()

    train_ds = FastCropDataset(train_samples, build_train_transform())
    val_ds = FastCropDataset(val_samples, build_val_transform())

    sample_weights = build_class_weights(train_samples)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_samples), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # Model
    available = timm.list_models("*dinov2*")
    model_name = "vit_base_patch14_dinov2"
    for c in ["vit_base_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2"]:
        if c in available:
            model_name = c
            break
    print(f"Model: {model_name}")

    backbone = timm.create_model(model_name, pretrained=True, num_classes=0, img_size=IMAGE_SIZE).to(device)
    embed_dim = backbone.num_features

    # Freeze first 75% of blocks
    for p in backbone.parameters():
        p.requires_grad = False
    if hasattr(backbone, "blocks"):
        n = len(backbone.blocks)
        for i in range(int(n * 0.75), n):
            for p in backbone.blocks[i].parameters():
                p.requires_grad = True
    for name, p in backbone.named_parameters():
        if any(k in name for k in ["norm.", "head.", "fc_norm.", "cls_token"]):
            p.requires_grad = True

    head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Dropout(0.2),
        nn.Linear(embed_dim, NUM_CATEGORIES),
    ).to(device)

    bb_params = [p for p in backbone.parameters() if p.requires_grad]
    hd_params = list(head.parameters())
    optimizer = torch.optim.AdamW([
        {"params": bb_params, "lr": lr * 0.1},
        {"params": hd_params, "lr": lr},
    ], weight_decay=1e-4)

    warmup = 2
    def lr_fn(ep):
        if ep < warmup:
            return (ep + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (ep - warmup) / max(1, epochs - warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    ema_bb = None
    ema_hd = None
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        if epoch == unfreeze_epoch:
            print(f"\n--- Unfreezing ALL backbone at epoch {epoch+1} ---")
            for p in backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": list(backbone.parameters()), "lr": lr * 0.01},
                {"params": hd_params, "lr": lr * 0.3},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch, eta_min=1e-7)
            ema_bb = EMA(backbone, 0.999)
            ema_hd = EMA(head, 0.999)

        backbone.train()
        head.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for bi, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if mixup_alpha > 0 and epoch >= warmup:
                imgs, ya, yb, lam = mixup_data(imgs, labels, mixup_alpha)
            else:
                ya, yb, lam = labels, labels, 1.0

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = head(backbone(imgs))
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + hd_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            if ema_bb:
                ema_bb.update(backbone)
                ema_hd.update(head)

            t_loss += loss.item() * imgs.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            t_total += imgs.size(0)

            if (bi + 1) % 100 == 0:
                print(f"  [{bi+1}] loss={loss.item():.4f} acc={t_correct/t_total:.3f}")

        scheduler.step()

        # Validate with EMA
        backbone.eval()
        head.eval()
        if ema_bb:
            bk_b = ema_bb.apply(backbone)
            bk_h = ema_hd.apply(head)

        v_correct, v_total, v_top5 = 0, 0, 0
        per_cat_c = defaultdict(int)
        per_cat_t = defaultdict(int)

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    logits = head(backbone(imgs))
                preds = logits.argmax(1)
                v_correct += (preds == labels).sum().item()
                v_total += imgs.size(0)
                _, t5 = logits.topk(5, dim=1)
                v_top5 += (t5 == labels.unsqueeze(1)).any(1).sum().item()
                for i in range(len(labels)):
                    c = labels[i].item()
                    per_cat_t[c] += 1
                    if preds[i].item() == c:
                        per_cat_c[c] += 1

        val_acc = v_correct / max(1, v_total)
        top5_acc = v_top5 / max(1, v_total)
        cat_accs = [per_cat_c[c] / per_cat_t[c] for c in per_cat_t if per_cat_t[c] > 0]
        mean_cat = np.mean(cat_accs) if cat_accs else 0
        above90 = sum(1 for a in cat_accs if a >= 0.9)
        below50 = sum(1 for a in cat_accs if a < 0.5)

        if ema_bb:
            ema_bb.restore(backbone, bk_b)
            ema_hd.restore(head, bk_h)

        print(f"E{epoch+1}/{epochs} loss={t_loss/max(1,t_total):.4f} "
              f"train={t_correct/max(1,t_total):.3f} val={val_acc:.3f} "
              f"top5={top5_acc:.3f} mean_cat={mean_cat:.3f} "
              f"[{above90}>90% {below50}<50%] lr={optimizer.param_groups[0]['lr']:.1e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if ema_bb:
                bk_b = ema_bb.apply(backbone)
                bk_h = ema_hd.apply(head)
            state = {
                "backbone": backbone.state_dict(),
                "classifier_head": head.state_dict(),
                "embed_dim": embed_dim,
                "num_categories": NUM_CATEGORIES,
                "model_name": model_name,
            }
            torch.save(state, str(MODELS_DIR / "dinov2_classifier_weights.pt"))
            if ema_bb:
                ema_bb.restore(backbone, bk_b)
                ema_hd.restore(head, bk_h)
            print(f"  -> SAVED best (val={val_acc:.3f}, mean_cat={mean_cat:.3f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("  Early stopping")
                break

    print(f"\nBest val accuracy: {best_val_acc:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--unfreeze-epoch", type=int, default=3)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    print(f"Config: {vars(args)}")
    train(args.epochs, args.batch_size, args.lr, args.device, args.unfreeze_epoch, args.mixup_alpha)


if __name__ == "__main__":
    main()
