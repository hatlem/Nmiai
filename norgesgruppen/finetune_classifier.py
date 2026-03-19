#!/usr/bin/env python3.11
"""
Fine-tune EfficientNet-B3 on cropped product images from training annotations.

Sources:
  1. Crops from data/train/images/ using bboxes in annotations.json (~22K crops)
  2. Reference product images from data/NM_NGD_product_images/ (~2400 images)

After training:
  - Overwrites models/efficientnet_b3_weights.pt (state_dict only, backbone without classifier)
  - Recomputes models/product_embeddings.npy using fine-tuned model
"""

import json
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Patch torch.load to use weights_only=False for compatibility
_original_torch_load = torch.load


def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)


torch.load = _patched_torch_load

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
PRODUCT_IMAGES_DIR = DATA_DIR / "NM_NGD_product_images"
ANNOTATIONS_FILE = DATA_DIR / "train" / "annotations.json"
METADATA_FILE = PRODUCT_IMAGES_DIR / "metadata.json"
TRAIN_IMAGES_DIR = DATA_DIR / "train" / "images"
MODELS_DIR = Path("models")

WEIGHTS_IN = MODELS_DIR / "efficientnet_b3_weights.pt"
WEIGHTS_OUT = MODELS_DIR / "efficientnet_b3_weights.pt"
EMBEDDINGS_OUT = MODELS_DIR / "product_embeddings.npy"

NUM_CLASSES = 356
UNKNOWN_CATEGORY_ID = 355
IMAGE_SIZE = 300

# Training hyperparameters
MAX_CROPS_PER_CATEGORY = 0    # 0 = no limit, use ALL crops
EPOCHS = 50
BATCH_SIZE = 128              # GPU-optimized
LR = 1e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
USE_ARCFACE = True            # ArcFace loss for better embeddings
ARCFACE_S = 30.0              # ArcFace scale
ARCFACE_M = 0.5               # ArcFace margin

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ANGLES = ["main", "front", "back", "left", "right", "top", "bottom"]

# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------


def build_product_code_to_category_id() -> dict:
    with open(ANNOTATIONS_FILE) as f:
        annotations = json.load(f)

    if not METADATA_FILE.exists():
        print("WARNING: metadata.json not found — reference images will not be used")
        return {}

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    name_to_cat_id = {cat["name"]: cat["id"] for cat in annotations["categories"]}

    product_code_to_cat_id = {}
    unmatched = []

    for product in metadata["products"]:
        product_code = product["product_code"]
        product_name = product["product_name"]

        if product_name in name_to_cat_id:
            product_code_to_cat_id[product_code] = name_to_cat_id[product_name]
        else:
            name_lower = product_name.lower()
            matched = False
            for cat_name, cat_id in name_to_cat_id.items():
                if cat_name.lower() == name_lower:
                    product_code_to_cat_id[product_code] = cat_id
                    matched = True
                    break
            if not matched:
                unmatched.append(f"{product_code}: {product_name}")

    if unmatched:
        print(f"WARNING: {len(unmatched)} products unmatched to category")

    print(f"Mapped {len(product_code_to_cat_id)} product codes to category IDs")
    return product_code_to_cat_id


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ProductDataset(Dataset):
    """Items are (image_path, bbox_or_None, category_id)."""

    def __init__(self, items: list, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, bbox, category_id = self.items[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            if bbox is not None:
                x, y, w, h = bbox
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = x1 + max(1, int(w)), y1 + max(1, int(h))
                img = img.crop((x1, y1, x2, y2))
        except Exception as e:
            # Return a black image on failure
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        if self.transform:
            img = self.transform(img)

        return img, category_id


def build_items(annotations_data: dict) -> list:
    """Build list of (image_path, bbox, category_id) from COCO annotations."""
    image_id_to_file = {
        img["id"]: img["file_name"] for img in annotations_data["images"]
    }

    # Collect all items per category first
    items_by_cat: dict[int, list] = defaultdict(list)

    for ann in annotations_data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id == UNKNOWN_CATEGORY_ID:
            continue
        image_file = image_id_to_file.get(ann["image_id"])
        if image_file is None:
            continue
        img_path = TRAIN_IMAGES_DIR / image_file
        if not img_path.exists():
            continue
        items_by_cat[cat_id].append((img_path, ann["bbox"], cat_id))

    # Apply per-category cap (0 = no limit)
    items = []
    for cat_id, cat_items in items_by_cat.items():
        if MAX_CROPS_PER_CATEGORY > 0 and len(cat_items) > MAX_CROPS_PER_CATEGORY:
            # Sort by bbox area descending (larger crops = better quality)
            cat_items.sort(key=lambda x: x[1][2] * x[1][3], reverse=True)
            cat_items = cat_items[:MAX_CROPS_PER_CATEGORY]
        items.extend(cat_items)

    print(f"Crop dataset: {len(items)} items from {len(items_by_cat)} categories")
    return items


def build_reference_items(product_code_to_cat_id: dict) -> list:
    """Build list of (image_path, None, category_id) from reference images."""
    items = []
    if not PRODUCT_IMAGES_DIR.exists():
        return items

    for product_dir in PRODUCT_IMAGES_DIR.iterdir():
        if not product_dir.is_dir():
            continue
        product_code = product_dir.name
        cat_id = product_code_to_cat_id.get(product_code)
        if cat_id is None or cat_id == UNKNOWN_CATEGORY_ID:
            continue
        for angle in ANGLES:
            img_path = product_dir / f"{angle}.jpg"
            if img_path.exists():
                items.append((img_path, None, cat_id))

    print(f"Reference dataset: {len(items)} items")
    return items


def train_val_split(items: list, val_fraction: float = 0.2):
    """Stratified split by category_id."""
    by_cat: dict[int, list] = defaultdict(list)
    for item in items:
        by_cat[item[2]].append(item)

    train_items, val_items = [], []
    for cat_id, cat_items in by_cat.items():
        np.random.shuffle(cat_items)
        n_val = max(1, int(len(cat_items) * val_fraction))
        val_items.extend(cat_items[:n_val])
        train_items.extend(cat_items[n_val:])

    return train_items, val_items


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(weights_path: Path | None = None, num_classes: int = NUM_CLASSES):
    """Load EfficientNet-B3 as a classifier."""
    import timm

    print(f"Loading timm EfficientNet-B3 (num_classes={num_classes})...")
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=num_classes)

    if weights_path is not None and weights_path.exists():
        print(f"Loading weights from {weights_path} ...")
        state_dict = torch.load(str(weights_path), map_location="cpu")
        # The saved weights may be a feature-extractor (num_classes=0), so load with strict=False
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys: {len(missing)} (classifier head will be random — expected)")
        if unexpected:
            print(f"  Unexpected keys: {len(unexpected)}")
    else:
        print("No pretrained weights found — starting from random init")

    return model


def get_transforms():
    """Return (train_transform, val_transform)."""
    import torchvision.transforms as T

    train_transform = T.Compose([
        T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        T.RandomCrop(IMAGE_SIZE),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.RandomErasing(p=0.2),
    ])

    val_transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


class ArcFaceLoss(nn.Module):
    """ArcFace angular margin loss for better embedding discrimination.

    Creates more discriminative embeddings by adding an angular margin
    to the target logit, forcing the model to learn tighter clusters
    per category. Much better than CrossEntropy for many-class, few-shot.
    """

    def __init__(self, embed_dim: int, num_classes: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        W = F.normalize(self.W, dim=1)
        cosine = embeddings @ W.T
        theta = torch.acos(torch.clamp(cosine, -1 + 1e-7, 1 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, cosine.size(1)).float()
        logits = self.s * (one_hot * target_logits + (1 - one_hot) * cosine)
        return F.cross_entropy(logits, labels)


def load_backbone(weights_path: Path | None = None):
    """Load EfficientNet-B3 as backbone (num_classes=0 for embeddings)."""
    import timm

    print("Loading timm EfficientNet-B3 backbone (num_classes=0)...")
    model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)

    if weights_path is not None and weights_path.exists():
        print(f"Loading backbone weights from {weights_path} ...")
        state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict, strict=False)

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)

        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}  loss={total_loss/total:.4f}  acc={correct/total:.3f}")

    return total_loss / total, correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Embedding computation (reuses build_embeddings.py logic)
# ---------------------------------------------------------------------------


def compute_product_embeddings(backbone, transform, product_code_to_cat_id: dict) -> np.ndarray:
    """Extract reference embeddings using the fine-tuned backbone (num_classes=0)."""
    backbone.eval()
    device = next(backbone.parameters()).device

    # Determine embedding dim
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
        emb_dim = backbone(dummy).shape[1]
    print(f"Embedding dimension: {emb_dim}")

    embeddings = np.zeros((NUM_CLASSES, emb_dim), dtype=np.float32)
    counts = np.zeros(NUM_CLASSES, dtype=np.int32)

    product_dirs = [d for d in PRODUCT_IMAGES_DIR.iterdir() if d.is_dir()]
    print(f"Processing {len(product_dirs)} product directories...")

    for product_dir in product_dirs:
        product_code = product_dir.name
        cat_id = product_code_to_cat_id.get(product_code)
        if cat_id is None or cat_id == UNKNOWN_CATEGORY_ID:
            continue

        for angle in ANGLES:
            img_path = product_dir / f"{angle}.jpg"
            if not img_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = backbone(tensor).squeeze(0).cpu().numpy()
                embeddings[cat_id] += emb
                counts[cat_id] += 1
            except Exception as e:
                print(f"  WARNING: could not process {img_path}: {e}")

    # Average and L2-normalize
    for i in range(NUM_CLASSES):
        if counts[i] > 0:
            embeddings[i] /= counts[i]
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                embeddings[i] /= norm

    covered = int((counts > 0).sum())
    print(f"Embeddings computed for {covered}/{NUM_CLASSES} categories")
    return embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    t_start = time.time()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    print("\n--- Building dataset ---")
    with open(ANNOTATIONS_FILE) as f:
        annotations_data = json.load(f)

    product_code_to_cat_id = build_product_code_to_category_id()

    crop_items = build_items(annotations_data)
    ref_items = build_reference_items(product_code_to_cat_id)

    all_items = crop_items + ref_items
    print(f"Total items: {len(all_items)}")

    # Stratified split (crops only — refs go to train only to avoid leakage)
    train_crops, val_crops = train_val_split(crop_items, VAL_SPLIT)
    train_items = train_crops + ref_items
    val_items = val_crops

    # Shuffle
    random.shuffle(train_items)
    random.shuffle(val_items)

    print(f"Train: {len(train_items)}  Val: {len(val_items)}")

    # ------------------------------------------------------------------
    # Transforms & DataLoaders
    # ------------------------------------------------------------------
    train_transform, val_transform = get_transforms()

    train_ds = ProductDataset(train_items, transform=train_transform)
    val_ds = ProductDataset(val_items, transform=val_transform)

    use_gpu = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4 if use_gpu else 0, pin_memory=use_gpu,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4 if use_gpu else 0, pin_memory=use_gpu,
    )

    print(f"Train batches: {len(train_loader)}  Val batches: {len(val_loader)}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n--- Loading model ---")

    if USE_ARCFACE:
        # ArcFace mode: backbone (num_classes=0) + ArcFace head
        print("Using ArcFace loss for better embedding discrimination")
        model = load_backbone(WEIGHTS_IN)
        model = model.to(device)

        # Get embedding dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            emb_dim = model(dummy).shape[1]
        print(f"Embedding dim: {emb_dim}")

        arcface_head = ArcFaceLoss(emb_dim, NUM_CLASSES, s=ARCFACE_S, m=ARCFACE_M).to(device)

        # Count params
        total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in arcface_head.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in arcface_head.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

        all_params = list(model.parameters()) + list(arcface_head.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_acc = 0.0
        best_state = None
        patience_counter = 0
        patience_limit = 15

        print(f"\n--- Training for {EPOCHS} epochs (ArcFace s={ARCFACE_S}, m={ARCFACE_M}) ---")
        print(f"Batch size: {BATCH_SIZE}, LR: {LR}")

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            # Train epoch (ArcFace)
            model.train()
            arcface_head.train()
            total_loss, correct, total = 0.0, 0, 0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                embeddings = model(images)
                loss = arcface_head(embeddings, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(labels)
                # Compute accuracy using cosine similarity (no margin)
                with torch.no_grad():
                    emb_norm = F.normalize(embeddings, dim=1)
                    W_norm = F.normalize(arcface_head.W, dim=1)
                    preds = (emb_norm @ W_norm.T).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                total += len(labels)

                if (batch_idx + 1) % 50 == 0:
                    print(f"    batch {batch_idx+1}/{len(train_loader)}  loss={total_loss/total:.4f}  acc={correct/total:.3f}")

            train_loss = total_loss / total
            train_acc = correct / total

            # Val epoch (ArcFace — evaluate with cosine similarity, no margin)
            model.eval()
            val_loss_total, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    embeddings = model(images)
                    loss = arcface_head(embeddings, labels)
                    val_loss_total += loss.item() * len(labels)
                    emb_norm = F.normalize(embeddings, dim=1)
                    W_norm = F.normalize(arcface_head.W, dim=1)
                    preds = (emb_norm @ W_norm.T).argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += len(labels)

            val_loss = val_loss_total / val_total
            val_acc = val_correct / val_total

            scheduler.step()
            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch:3d}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                f"lr={lr_now:.2e}  time={elapsed:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                patience_counter = 0
                print(f"  *** New best val_acc={best_val_acc:.3f} — saving backbone weights ***")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"  Early stopping after {patience_limit} epochs without improvement")
                    break
    else:
        # Standard CrossEntropy mode
        model = load_model(WEIGHTS_IN, num_classes=NUM_CLASSES)
        model = model.to(device)

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,} total, {trainable:,} trainable")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_acc = 0.0
        best_state = None

        print(f"\n--- Training for {EPOCHS} epochs ---")
        print(f"Batch size: {BATCH_SIZE}, LR: {LR}")

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
            scheduler.step()

            elapsed = time.time() - t0
            lr_now = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch:3d}/{EPOCHS}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
                f"lr={lr_now:.2e}  time={elapsed:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                import timm
                backbone = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
                classifier_sd = model.state_dict()
                backbone_sd = {k: v for k, v in classifier_sd.items() if k in backbone.state_dict()}
                backbone.load_state_dict(backbone_sd, strict=False)
                best_state = {k: v.clone() for k, v in backbone.state_dict().items()}
                print(f"  *** New best val_acc={best_val_acc:.3f} — saving backbone weights ***")

    total_time = time.time() - t_start
    print(f"\nTraining complete in {total_time/60:.1f} min")
    print(f"Best val accuracy: {best_val_acc:.3f}")

    # ------------------------------------------------------------------
    # Save backbone weights
    # ------------------------------------------------------------------
    if best_state is not None:
        print(f"\n--- Saving backbone weights to {WEIGHTS_OUT} ---")
        torch.save(best_state, str(WEIGHTS_OUT))
        print("Saved.")
    else:
        print("WARNING: no best_state captured — not overwriting weights")

    # ------------------------------------------------------------------
    # Recompute product embeddings with fine-tuned backbone
    # ------------------------------------------------------------------
    print("\n--- Recomputing product embeddings ---")
    import timm
    from timm.data import resolve_data_config, create_transform

    backbone = timm.create_model("efficientnet_b3", pretrained=False, num_classes=0)
    if best_state is not None:
        backbone.load_state_dict(best_state, strict=True)
    backbone = backbone.to(device)
    backbone.eval()

    # Use timm's canonical val transform
    data_config = resolve_data_config({}, model=backbone)
    data_config["mean"] = IMAGENET_MEAN
    data_config["std"] = IMAGENET_STD
    emb_transform = create_transform(**data_config)

    embeddings = compute_product_embeddings(backbone, emb_transform, product_code_to_cat_id)

    np.save(str(EMBEDDINGS_OUT), embeddings)
    print(f"Saved embeddings: {EMBEDDINGS_OUT}  shape={embeddings.shape}")

    # ------------------------------------------------------------------
    # Update embedding_config.json
    # ------------------------------------------------------------------
    config = {
        "model_name": "efficientnet_b3",
        "embedding_dim": int(embeddings.shape[1]),
        "num_categories": NUM_CLASSES,
    }
    with open(MODELS_DIR / "embedding_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Updated embedding_config.json: {config}")

    print(f"\nDone! Total wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
