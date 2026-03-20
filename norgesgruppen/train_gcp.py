"""Train improved single-class YOLO detector on GCP with GPU.

Single-class detection maximizes detection mAP (70% of score).
Classification handled separately by DINOv2 (30% of score).

Key improvements over previous training:
  1. YOLO11x (22% fewer params, better mAP than YOLOv8x)
  2. 300 epochs with patience=50
  3. Image size 1280 (matches inference)
  4. Aggressive augmentation for dense retail shelves
  5. Copy-paste augmentation (shelf products)
  6. Auto-exports to ONNX for NMS-free inference

Usage:
    # On GCP Compute Engine with GPU:
    pip install ultralytics==8.1.0 onnx onnxruntime-gpu
    python train_gcp.py

    # Or with custom args:
    python train_gcp.py --model yolo11x.pt --epochs 300 --batch 4 --imgsz 1280
"""

import torch
_orig_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(f, *a, **kw)
torch.load = _patched_load

import argparse
import shutil
from pathlib import Path

import cv2


def convert_to_single_class(data_yaml: str):
    """Convert multi-class labels to single-class (class 0 = product)."""
    dataset_dir = Path(data_yaml).parent
    marker = dataset_dir / ".single_class_done"
    if marker.exists():
        print("Already converted to single-class")
        return

    for split in ["train", "val"]:
        labels_dir = dataset_dir / "labels" / split
        if not labels_dir.exists():
            continue
        count = 0
        for lf in sorted(labels_dir.glob("*.txt")):
            lines = lf.read_text().strip().split("\n")
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    # Replace class ID with 0
                    parts[0] = "0"
                    new_lines.append(" ".join(parts))
            lf.write_text("\n".join(new_lines) + "\n")
            count += 1
        print(f"  {split}: converted {count} label files to single-class")

    # Update data.yaml
    yaml_path = Path(data_yaml)
    content = yaml_path.read_text()
    # Replace names section with single class
    import re
    content = re.sub(r'names:.*?(?=\nnc:|\Z)', 'names:\n  0: product\n', content, flags=re.DOTALL)
    content = re.sub(r'nc: \d+', 'nc: 1', content)
    yaml_path.write_text(content)
    print("Updated data.yaml to single-class")

    marker.write_text("done")


def apply_clahe(data_yaml: str):
    """Apply CLAHE to match inference preprocessing."""
    dataset_dir = Path(data_yaml).parent
    marker = dataset_dir / ".clahe_done"
    if marker.exists():
        print("CLAHE already applied")
        return

    print("Applying CLAHE...")
    for split in ["train", "val"]:
        img_dir = dataset_dir / "images" / split
        if not img_dir.exists():
            continue
        count = 0
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.merge([clahe.apply(l), a, b])
            cv2.imwrite(str(p), cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
            count += 1
        print(f"  {split}: {count} images")
    marker.write_text("done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11x.pt",
                       help="Base model (yolo11x.pt, yolov8x.pt)")
    parser.add_argument("--data", default="data/yolo_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--single-class", action="store_true", default=True,
                       help="Convert to single-class detection (default: True)")
    parser.add_argument("--export-onnx", action="store_true", default=True)
    args = parser.parse_args()

    # Preprocessing
    apply_clahe(args.data)
    if args.single_class:
        convert_to_single_class(args.data)

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
        save_period=10,
        val=True,
        plots=True,
        # Optimizer — AdamW with cosine schedule
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,
        # Augmentation — tuned for dense grocery shelves
        hsv_h=0.02,       # Slight hue shift (lighting variations)
        hsv_s=0.7,        # Saturation (packaging colors)
        hsv_v=0.5,        # Value/brightness (shelf lighting)
        degrees=5.0,      # Small rotation (products rarely tilted)
        translate=0.2,    # Translation (products anywhere on shelf)
        scale=0.8,        # Scale variation (different shelf depths)
        shear=2.0,        # Minor shear
        flipud=0.0,       # No vertical flip (products don't flip)
        fliplr=0.5,       # Horizontal flip (shelf symmetry)
        mosaic=1.0,       # Full mosaic (more products per training image)
        mixup=0.15,       # Light mixup (don't confuse detections)
        copy_paste=0.5,   # Copy-paste products (boost small object detection)
        erasing=0.3,      # Random erasing (occlusion robustness)
        close_mosaic=20,  # Disable mosaic for last 20 epochs (fine-tune)
    )

    # Save best weights
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        # Copy locally
        shutil.copy2(best_pt, "best_single_new.pt")
        size = best_pt.stat().st_size / 1024 / 1024
        print(f"\nbest_single_new.pt: {size:.1f} MB")

        # Export to ONNX (opset 17, dynamic batch)
        if args.export_onnx:
            print("\nExporting to ONNX...")
            best_model = YOLO(str(best_pt))
            best_model.export(
                format="onnx",
                imgsz=args.imgsz,
                half=True,
                simplify=True,
                opset=17,
                dynamic=True,
            )
            onnx_path = best_pt.with_suffix(".onnx")
            if onnx_path.exists():
                shutil.copy2(onnx_path, "best_new.onnx")
                onnx_size = onnx_path.stat().st_size / 1024 / 1024
                print(f"best_new.onnx: {onnx_size:.1f} MB")

        # Upload to GCS
        try:
            import subprocess
            bucket = "gs://ainm26osl-710-norgesgruppen/models"
            subprocess.run(["gsutil", "cp", "best_single_new.pt", f"{bucket}/best_single_new.pt"])
            if Path("best_new.onnx").exists():
                subprocess.run(["gsutil", "cp", "best_new.onnx", f"{bucket}/best_new.onnx"])
            print("Uploaded to GCS")
        except Exception as e:
            print(f"GCS upload failed: {e}")


if __name__ == "__main__":
    main()
