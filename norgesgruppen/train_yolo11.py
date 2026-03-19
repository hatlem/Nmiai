"""Train YOLO11x on NorgesGruppen competition data + export to ONNX.

Requires: pip install ultralytics (latest, NOT 8.1.0 — we export to ONNX for sandbox)

Usage:
    python train_yolo11.py
    python train_yolo11.py --model yolo11x.pt --epochs 300 --batch 4 --imgsz 1280
    python train_yolo11.py --resume
    python train_yolo11.py --export-only --weights runs/detect/train/weights/best.pt
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


def export_to_onnx(weights_path: str, imgsz: int = 1280):
    """Export trained model to ONNX FP16 for sandbox deployment."""
    from ultralytics import YOLO

    print(f"\n--- Exporting {weights_path} to ONNX ---")
    model = YOLO(weights_path)
    onnx_path = model.export(
        format="onnx",
        opset=17,
        imgsz=imgsz,
        half=True,
        simplify=True,
    )
    print(f"Exported: {onnx_path}")

    onnx_file = Path(onnx_path)
    # Name matches run_best.py secondary_names_onnx convention
    dest = Path("yolo11_best.onnx")
    shutil.copy2(onnx_file, dest)
    print(f"Copied to: {dest}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return str(dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11x.pt", help="Base model")
    parser.add_argument("--data", default="data/yolo_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-clahe", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--weights", default=None, help="Path to weights for export-only")
    args = parser.parse_args()

    if args.export_only:
        weights = args.weights or "runs/detect/train/weights/best.pt"
        export_to_onnx(weights, args.imgsz)
        return

    if not args.no_clahe and not args.resume:
        apply_clahe_to_dataset(args.data)

    from ultralytics import YOLO

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

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        dest_pt = Path("yolo11_best.pt")
        shutil.copy2(best_pt, dest_pt)
        print(f"\nBest .pt model: {dest_pt} ({dest_pt.stat().st_size / 1024 / 1024:.1f} MB)")
        export_to_onnx(str(best_pt), args.imgsz)


if __name__ == "__main__":
    main()
