"""Train YOLOv8x on NorgesGruppen competition data.

Requires: ultralytics==8.1.0 (matches sandbox version)

IMPORTANT: Inference uses CLAHE preprocessing. To match, we apply CLAHE to
all training images as a pre-step before training starts.

Usage:
    # Full 356-class (single-stage):
    python train.py

    # Single-class detection (for two-stage pipeline):
    python train.py --data data/yolo_single_class/data.yaml

    # With SKU110K extra data:
    python train.py --data data/yolo_single_class_merged/data.yaml

    # Custom settings:
    python train.py --model yolov8l.pt --epochs 200 --batch 8 --imgsz 1280
"""

import argparse
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO


def apply_clahe_to_dataset(data_yaml: str) -> str:
    """Apply CLAHE to all training images so train matches inference.

    Creates a parallel directory with CLAHE-enhanced images and returns
    a new data.yaml path pointing to them. Labels are symlinked.
    """
    data_path = Path(data_yaml)
    dataset_dir = data_path.parent

    # Read data.yaml to find image dirs
    yaml_text = data_path.read_text()

    # Check if already processed
    clahe_marker = dataset_dir / ".clahe_done"
    if clahe_marker.exists():
        print("CLAHE already applied, skipping")
        return data_yaml

    print("Applying CLAHE to training images (one-time)...")

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

            # CLAHE in LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l_ch)
            lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            # Overwrite in-place (training data is a copy, not the original)
            cv2.imwrite(str(img_path), result)
            count += 1

        print(f"  {split}: {count} images enhanced")

    clahe_marker.write_text("done")
    return data_yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8x.pt", help="Base model")
    parser.add_argument("--data", default="data/yolo_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-clahe", action="store_true", help="Skip CLAHE preprocessing")
    args = parser.parse_args()

    # Apply CLAHE to match inference preprocessing
    if not args.no_clahe and not args.resume:
        apply_clahe_to_dataset(args.data)

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=8,
        patience=50,
        save=True,
        save_period=25,
        val=True,
        plots=True,
        # Optimizer
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        # Augmentation — strong for small dataset
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.15,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
        # Other
        close_mosaic=30,
        resume=args.resume,
    )

    # Copy best.pt for submission
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        import shutil
        dest = Path("best.pt")
        shutil.copy2(best_pt, dest)
        print(f"\nBest model copied to: {dest}")
        print(f"Size: {dest.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
