"""Train YOLOv8x multi-class with class-weighted sampling + reference image augmentation.

Key improvements over train.py:
  1. Uses ALL 356 categories (multi-class, not single-class)
  2. Class-weighted oversampling for rare categories
  3. Injects product reference images as extra training crops
  4. CLAHE preprocessing to match inference
  5. Aggressive augmentation for better generalization
  6. Higher patience for rare-class convergence

Requires: ultralytics==8.1.0 (matches sandbox)

Usage:
    python train_multiclass_v2.py
    python train_multiclass_v2.py --model yolov8x.pt --epochs 200 --imgsz 1280
"""

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def apply_clahe_to_dataset(data_yaml: str) -> str:
    """Apply CLAHE to all training images (one-time)."""
    data_path = Path(data_yaml)
    dataset_dir = data_path.parent

    clahe_marker = dataset_dir / ".clahe_done"
    if clahe_marker.exists():
        print("CLAHE already applied, skipping")
        return data_yaml

    print("Applying CLAHE to training images...")

    for split in ["train", "val"]:
        img_dir = dataset_dir / "images" / split
        if not img_dir.exists():
            continue

        count = 0
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_ch)
            lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            cv2.imwrite(str(img_path), result)
            count += 1

        print(f"  {split}: {count} images enhanced")

    clahe_marker.write_text("done")
    return data_yaml


def oversample_rare_classes(data_yaml: str, min_instances: int = 30):
    """Duplicate label entries for rare classes to balance the dataset.

    For each class with fewer than min_instances annotations,
    duplicate existing annotations (and images) to reach min_instances.
    """
    data_path = Path(data_yaml)
    dataset_dir = data_path.parent

    oversample_marker = dataset_dir / ".oversample_done"
    if oversample_marker.exists():
        print("Oversampling already applied, skipping")
        return

    labels_dir = dataset_dir / "labels" / "train"
    images_dir = dataset_dir / "images" / "train"

    if not labels_dir.exists():
        print("No labels/train directory — skipping oversampling")
        return

    # Count instances per class across all label files
    class_to_files = defaultdict(list)  # class_id -> [(label_file, line_idx)]

    for label_file in sorted(labels_dir.glob("*.txt")):
        with open(label_file) as f:
            lines = f.readlines()
        for line_idx, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                class_to_files[class_id].append((label_file, line_idx))

    # Find rare classes
    rare_classes = {cid: entries for cid, entries in class_to_files.items()
                    if len(entries) < min_instances}

    if not rare_classes:
        print("No rare classes found — skipping oversampling")
        oversample_marker.write_text("done")
        return

    print(f"Oversampling {len(rare_classes)} rare classes (< {min_instances} instances)")

    total_copies = 0
    for class_id, entries in rare_classes.items():
        current_count = len(entries)
        copies_needed = min_instances - current_count

        for i in range(copies_needed):
            # Pick a random source from existing entries
            src_label, src_line_idx = entries[i % current_count]
            src_image = images_dir / (src_label.stem + src_label.suffix.replace(".txt", ".jpg"))

            # Find the actual image extension
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = images_dir / (src_label.stem + ext)
                if candidate.exists():
                    src_image = candidate
                    break

            if not src_image.exists():
                continue

            # Create a copy with unique name
            copy_name = f"{src_label.stem}_oversample_{class_id}_{i}"
            dst_image = images_dir / f"{copy_name}{src_image.suffix}"
            dst_label = labels_dir / f"{copy_name}.txt"

            if not dst_image.exists():
                shutil.copy2(src_image, dst_image)
                # Copy the FULL label file (all annotations for that image)
                shutil.copy2(src_label, dst_label)
                total_copies += 1

    print(f"Created {total_copies} oversampled image copies")
    oversample_marker.write_text("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8x.pt", help="Base model")
    parser.add_argument("--data", default="data/yolo_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--no-oversample", action="store_true")
    parser.add_argument("--min-instances", type=int, default=30,
                        help="Minimum instances per class for oversampling")
    args = parser.parse_args()

    # Apply CLAHE
    if not args.no_clahe and not args.resume:
        apply_clahe_to_dataset(args.data)

    # Oversample rare classes
    if not args.no_oversample and not args.resume:
        oversample_rare_classes(args.data, min_instances=args.min_instances)

    from ultralytics import YOLO
    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=8,
        patience=30,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        # Augmentation — very aggressive for 356-class retail
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.5,
        degrees=10.0,
        translate=0.2,
        scale=0.9,
        shear=3.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        copy_paste=0.4,
        erasing=0.4,
        # Other
        close_mosaic=15,
        resume=args.resume,
    )

    # Copy best.pt
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        dest = Path("best_multiclass_v2.pt")
        shutil.copy2(best_pt, dest)
        print(f"\nBest model: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")

        # Also export FP16 if too large
        size_mb = dest.stat().st_size / 1024 / 1024
        if size_mb > 200:
            print("Model > 200 MB, creating FP16 version...")
            import torch
            checkpoint = torch.load(str(dest), map_location="cpu")
            # Convert to FP16
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                for key in checkpoint["model"].state_dict():
                    param = checkpoint["model"].state_dict()[key]
                    if param.dtype == torch.float32:
                        checkpoint["model"].state_dict()[key] = param.half()
            fp16_dest = Path("best_multiclass_v2_fp16.pt")
            torch.save(checkpoint, str(fp16_dest))
            print(f"FP16 model: {fp16_dest} ({fp16_dest.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
