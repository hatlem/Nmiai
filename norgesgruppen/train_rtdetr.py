"""Train RT-DETR on NorgesGruppen competition data for ensemble with YOLOv8x.

Requires: ultralytics==8.1.0 (matches sandbox version)

RT-DETR (Real-Time Detection Transformer) is a transformer-based detector
that complements YOLO's CNN-based architecture. Ensembling both via WBF
improves detection by combining diverse feature representations.

IMPORTANT: Inference uses CLAHE preprocessing. To match, we apply CLAHE to
all training images as a pre-step before training starts (same as train.py).

Usage:
    # Single-class detection (for two-stage pipeline):
    python train_rtdetr.py --data data/yolo_single_class/data.yaml

    # Full 356-class (single-stage):
    python train_rtdetr.py --data data/yolo_dataset/data.yaml --model rtdetr-l.pt

    # RT-DETR-x (larger, better accuracy):
    python train_rtdetr.py --model rtdetr-x.pt --epochs 200

    # Resume training:
    python train_rtdetr.py --resume
"""

import argparse
import shutil
from pathlib import Path

import cv2
from ultralytics import RTDETR


def apply_clahe_to_dataset(data_yaml: str) -> str:
    """Apply CLAHE to all training images so train matches inference.

    Creates CLAHE-enhanced images in-place and marks with a sentinel file.
    Identical to train.py — ensures both models see the same preprocessing.
    """
    data_path = Path(data_yaml)
    dataset_dir = data_path.parent

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

            # CLAHE in LAB color space — matches inference preprocessing
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


def estimate_combined_size(rtdetr_model: str) -> None:
    """Print estimated combined weight size for YOLOv8x + RT-DETR."""
    # Approximate .pt sizes (after training)
    sizes = {
        "rtdetr-l.pt": 65,   # ~65 MB
        "rtdetr-x.pt": 135,  # ~135 MB
    }
    yolov8x_size = 131  # ~131 MB

    rtdetr_size = sizes.get(rtdetr_model, 65)
    total = yolov8x_size + rtdetr_size
    limit = 420

    print(f"\nEstimated combined weight sizes:")
    print(f"  YOLOv8x best.pt:    ~{yolov8x_size} MB")
    print(f"  {rtdetr_model}:  ~{rtdetr_size} MB")
    print(f"  Total:               ~{total} MB / {limit} MB limit")
    if total > limit:
        print(f"  WARNING: Exceeds {limit} MB limit! Consider rtdetr-l.pt or ONNX export.")
    else:
        print(f"  OK: {limit - total} MB headroom remaining.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Train RT-DETR for ensemble with YOLOv8x")
    parser.add_argument(
        "--model", default="rtdetr-l.pt",
        help="Base model: rtdetr-l.pt (~65MB) or rtdetr-x.pt (~135MB)",
    )
    parser.add_argument("--data", default="data/yolo_single_class/data.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-clahe", action="store_true", help="Skip CLAHE preprocessing")
    parser.add_argument(
        "--name", default="rtdetr",
        help="Run name for saving results (default: rtdetr)",
    )
    args = parser.parse_args()

    estimate_combined_size(args.model)

    # Apply CLAHE to match inference preprocessing
    if not args.no_clahe and not args.resume:
        apply_clahe_to_dataset(args.data)

    model = RTDETR(args.model)

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
        name=args.name,
        # Optimizer — AdamW with warmup (RT-DETR benefits from lower lr)
        optimizer="AdamW",
        lr0=0.0001,       # Lower than YOLO — transformers need smaller lr
        lrf=0.01,
        weight_decay=0.0001,
        warmup_epochs=5,
        # Augmentation — strong for small dataset
        # RT-DETR uses same ultralytics augmentation pipeline as YOLO
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
        dest = Path("rtdetr_best.pt")
        shutil.copy2(best_pt, dest)
        print(f"\nBest RT-DETR model copied to: {dest}")
        print(f"Size: {dest.stat().st_size / 1024 / 1024:.1f} MB")

        # Also check combined size with YOLOv8x
        yolo_best = Path("best.pt")
        if yolo_best.exists():
            combined = dest.stat().st_size + yolo_best.stat().st_size
            combined_mb = combined / 1024 / 1024
            print(f"Combined with best.pt: {combined_mb:.1f} MB / 420 MB limit")
            if combined_mb > 420:
                print("WARNING: Combined weights exceed 420 MB! Export to ONNX FP16.")


if __name__ == "__main__":
    main()
