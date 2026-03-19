"""Train YOLO11x on NorgesGruppen data and export to ONNX for sandbox.

YOLO11 has C2PSA spatial attention — better for dense shelves and small objects.
22% fewer params than YOLOv8, better mAP. Must export to ONNX since
ultralytics==8.1.0 in sandbox doesn't support YOLO11.

Requires: pip install ultralytics  (latest, NOT 8.1.0)

Usage:
    # Train on GCP GPU:
    python train_yolo11.py --device 0 --batch 8

    # Export only (after training):
    python train_yolo11.py --export-only --weights runs/detect/train/weights/best.pt

    # Train + auto-export:
    python train_yolo11.py --device 0 --batch 8 --export
"""

import argparse
import shutil
from pathlib import Path

import cv2


def apply_clahe_to_dataset(data_yaml: str) -> str:
    """Apply CLAHE to all training images so train matches inference."""
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


def export_onnx(weights_path: str, imgsz: int = 1280, half: bool = True):
    """Export trained YOLO11 model to ONNX for sandbox inference."""
    from ultralytics import YOLO

    model = YOLO(weights_path)
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=17,
        half=half,
    )
    print(f"ONNX exported: {onnx_path}")

    # Copy to project root
    dest = Path("best.onnx")
    shutil.copy2(onnx_path, dest)
    size_mb = dest.stat().st_size / 1024 / 1024
    print(f"Copied to {dest} ({size_mb:.1f} MB)")
    return str(dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11x.pt", help="Base YOLO11 model")
    parser.add_argument("--data", default="data/yolo_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--export", action="store_true", help="Export to ONNX after training")
    parser.add_argument("--export-only", action="store_true", help="Only export, no training")
    parser.add_argument("--weights", default=None, help="Weights path for --export-only")
    parser.add_argument("--no-half", action="store_true", help="Export FP32 instead of FP16")
    args = parser.parse_args()

    if args.export_only:
        weights = args.weights or "runs/detect/train/weights/best.pt"
        export_onnx(weights, imgsz=args.imgsz, half=not args.no_half)
        return

    from ultralytics import YOLO

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
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        # Augmentation — aggressive for dense retail shelves
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.15,
        scale=0.9,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        erasing=0.3,
        # Other
        close_mosaic=10,
        resume=args.resume,
    )

    # Copy best.pt
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        dest = Path("yolo11_best.pt")
        shutil.copy2(best_pt, dest)
        print(f"\nBest model copied to: {dest}")
        print(f"Size: {dest.stat().st_size / 1024 / 1024:.1f} MB")

        if args.export:
            export_onnx(str(dest), imgsz=args.imgsz, half=not args.no_half)


if __name__ == "__main__":
    main()
