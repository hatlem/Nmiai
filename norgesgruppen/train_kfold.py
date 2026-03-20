#!/usr/bin/env python3
"""5-fold cross-validation training with YOLO26-x for grocery product detection."""

import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
torch.load = _patched

import os
import shutil
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

# Configuration
BASE_DIR = Path("/tmp/train")
DATASET_DIR = BASE_DIR / "data" / "yolo_dataset"
FOLDS_DIR = BASE_DIR / "data" / "folds"
WEIGHTS_PATH = str(BASE_DIR / "yolo26x.pt")
NUM_FOLDS = 5
EPOCHS = 150
BATCH = 6
IMGSZ = 1280
PATIENCE = 40
LR0 = 0.001
SEED = 42

def collect_all_images():
    """Collect all image paths from both train and val directories."""
    images = []
    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        if img_dir.exists():
            for f in sorted(img_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                    images.append(f)
    print(f"Total images collected: {len(images)}")
    return images

def get_label_path(img_path):
    """Get corresponding label path for an image."""
    # Try same split first
    parts = img_path.parts
    # .../images/train/img.jpg -> .../labels/train/img.txt
    idx = parts.index("images")
    label_parts = list(parts)
    label_parts[idx] = "labels"
    label_path = Path(*label_parts).with_suffix(".txt")
    if label_path.exists():
        return label_path
    # Try other split
    split = parts[idx + 1]
    other_split = "val" if split == "train" else "train"
    label_parts[idx + 1] = other_split
    label_path2 = Path(*label_parts).with_suffix(".txt")
    if label_path2.exists():
        return label_path2
    return None

def read_data_yaml():
    """Read the original data.yaml to get class names."""
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path) as f:
        return yaml.safe_load(f)

def create_fold_directories(images, fold_indices, fold_num, data_config):
    """Create directory structure for a single fold."""
    fold_dir = FOLDS_DIR / f"fold_{fold_num}"

    train_img_dir = fold_dir / "images" / "train"
    val_img_dir = fold_dir / "images" / "val"
    train_lbl_dir = fold_dir / "labels" / "train"
    val_lbl_dir = fold_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_idx, val_idx = fold_indices

    # Symlink train images and labels
    for i in train_idx:
        img = images[i]
        dst = train_img_dir / img.name
        if not dst.exists():
            os.symlink(img, dst)
        lbl = get_label_path(img)
        if lbl:
            lbl_dst = train_lbl_dir / lbl.name
            if not lbl_dst.exists():
                os.symlink(lbl, lbl_dst)

    # Symlink val images and labels
    for i in val_idx:
        img = images[i]
        dst = val_img_dir / img.name
        if not dst.exists():
            os.symlink(img, dst)
        lbl = get_label_path(img)
        if lbl:
            lbl_dst = val_lbl_dir / lbl.name
            if not lbl_dst.exists():
                os.symlink(lbl, lbl_dst)

    # Write data.yaml for this fold
    fold_yaml = {
        "path": str(fold_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": data_config["nc"],
        "names": data_config["names"],
    }
    yaml_path = fold_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(fold_yaml, f, default_flow_style=False)

    print(f"  Fold {fold_num}: {len(train_idx)} train, {len(val_idx)} val images")
    return yaml_path

def main():
    print("=" * 60)
    print("YOLO26-x 5-Fold Cross-Validation Training")
    print("=" * 60)

    # Collect all images
    images = collect_all_images()
    if not images:
        raise RuntimeError("No images found!")

    # Read original config
    data_config = read_data_yaml()
    print(f"Classes: {data_config['nc']}")

    # Clean old folds
    if FOLDS_DIR.exists():
        shutil.rmtree(FOLDS_DIR)
    FOLDS_DIR.mkdir(parents=True)

    # Create K-Fold splits
    random.seed(SEED)
    indices = list(range(len(images)))
    random.shuffle(indices)

    # Manual KFold split (no sklearn dependency)
    fold_size = len(indices) // NUM_FOLDS
    folds = []
    for i in range(NUM_FOLDS):
        val_start = i * fold_size
        val_end = val_start + fold_size if i < NUM_FOLDS - 1 else len(indices)
        val_idx = list(range(val_start, val_end))
        train_idx = list(range(0, val_start)) + list(range(val_end, len(indices)))
        folds.append((train_idx, val_idx))

    print(f"\nCreating {NUM_FOLDS} fold directories...")
    fold_yamls = []
    for fold_num, (train_idx_pos, val_idx_pos) in enumerate(folds):
        # Map back to actual indices
        train_idx = [indices[i] for i in train_idx_pos]
        val_idx = [indices[i] for i in val_idx_pos]
        yaml_path = create_fold_directories(images, (train_idx, val_idx), fold_num, data_config)
        fold_yamls.append(yaml_path)

    # Train each fold
    for fold_num, yaml_path in enumerate(fold_yamls):
        print(f"\n{'=' * 60}")
        print(f"TRAINING FOLD {fold_num + 1}/{NUM_FOLDS}")
        print(f"{'=' * 60}")

        model = YOLO(WEIGHTS_PATH)

        results = model.train(
            data=str(yaml_path),
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            patience=PATIENCE,
            optimizer="AdamW",
            lr0=LR0,
            mosaic=1.0,
            mixup=0.15,
            copy_paste=0.1,
            device=0,
            project=str(BASE_DIR / "runs" / "kfold"),
            name=f"fold_{fold_num}",
            exist_ok=True,
            verbose=True,
            save=True,
            save_period=25,
            workers=8,
            seed=SEED + fold_num,
            close_mosaic=15,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.0,
        )

        # Copy best weights
        best_src = BASE_DIR / "runs" / "kfold" / f"fold_{fold_num}" / "weights" / "best.pt"
        best_dst = BASE_DIR / f"best_fold_{fold_num}.pt"
        if best_src.exists():
            shutil.copy2(best_src, best_dst)
            print(f"Saved best weights: {best_dst}")
        else:
            print(f"WARNING: No best.pt found for fold {fold_num}")

    print(f"\n{'=' * 60}")
    print("ALL FOLDS COMPLETE!")
    print(f"Best weights saved to: {BASE_DIR}/best_fold_*.pt")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
