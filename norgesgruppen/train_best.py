"""Best-in-class multi-class YOLOv8x training for NorgesGruppen.

Single model that handles BOTH detection AND classification end-to-end.
No two-stage pipeline, no hacks — just a well-trained multi-class detector.

Key techniques:
  1. Class-weighted oversampling (rare categories get more training signal)
  2. CLAHE preprocessing (matches inference)
  3. Aggressive augmentation tuned for grocery shelves
  4. FP16 export for submission (fits under 420 MB)

Requires: ultralytics==8.1.0 (matches sandbox exactly)

Usage:
    python train_best.py                    # Full training
    python train_best.py --epochs 100       # Faster
    python train_best.py --export-only      # Just export existing best.pt to FP16
"""

import torch
_orig_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(f, *a, **kw)
torch.load = _patched_load

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


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


def oversample_rare(data_yaml: str, min_count: int = 30):
    """Duplicate images containing rare categories to balance training."""
    dataset_dir = Path(data_yaml).parent
    marker = dataset_dir / ".oversample_done"
    if marker.exists():
        print("Oversampling already done")
        return

    labels_dir = dataset_dir / "labels" / "train"
    images_dir = dataset_dir / "images" / "train"
    if not labels_dir.exists():
        return

    # Count per-class instances
    class_counts = Counter()
    class_files = defaultdict(list)
    for lf in sorted(labels_dir.glob("*.txt")):
        classes_in_file = set()
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    class_counts[cid] += 1
                    classes_in_file.add(cid)
        for cid in classes_in_file:
            class_files[cid].append(lf)

    rare = {cid: files for cid, files in class_files.items()
            if class_counts[cid] < min_count}
    if not rare:
        print("No rare classes")
        marker.write_text("done")
        return

    print(f"Oversampling {len(rare)} rare classes (< {min_count} instances)")
    copies = 0
    for cid, files in rare.items():
        needed = min_count - class_counts[cid]
        for i in range(needed):
            src_lbl = files[i % len(files)]
            # Find matching image
            src_img = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = images_dir / (src_lbl.stem + ext)
                if candidate.exists():
                    src_img = candidate
                    break
            if src_img is None:
                continue

            name = f"{src_lbl.stem}_ov{cid}_{i}"
            dst_img = images_dir / f"{name}{src_img.suffix}"
            dst_lbl = labels_dir / f"{name}.txt"
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_lbl, dst_lbl)
                copies += 1

    print(f"Created {copies} oversampled copies")
    marker.write_text("done")


def export_fp16(weights_path: str, dest: str = "best_final.pt"):
    """Export model weights as FP16 for smaller submission size."""
    print(f"Exporting {weights_path} as FP16...")
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            model = ckpt["model"]
            if hasattr(model, "half"):
                ckpt["model"] = model.half()
            elif hasattr(model, "state_dict"):
                for k, v in model.state_dict().items():
                    if v.dtype == torch.float32:
                        model.state_dict()[k] = v.half()
    torch.save(ckpt, dest)
    size = Path(dest).stat().st_size / 1024 / 1024
    print(f"Saved: {dest} ({size:.1f} MB)")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8x.pt")
    parser.add_argument("--data", default="data/yolo_dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--min-instances", type=int, default=30)
    args = parser.parse_args()

    if args.export_only:
        w = args.weights or "runs/detect/train/weights/best.pt"
        export_fp16(w)
        return

    if not args.resume:
        apply_clahe(args.data)
        oversample_rare(args.data, min_count=args.min_instances)

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
        # Augmentation — aggressive for dense retail, 356 classes
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
        close_mosaic=15,
        resume=args.resume,
    )

    # Save and export
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, "best_multiclass.pt")
        size = best_pt.stat().st_size / 1024 / 1024
        print(f"\nbest_multiclass.pt: {size:.1f} MB")

        export_fp16(str(best_pt), "best_final.pt")

        # Upload to GCS
        import subprocess
        subprocess.run(["gsutil", "cp", "best_multiclass.pt",
                       "gs://ainm26osl-710-norgesgruppen/models/best_multiclass.pt"])
        subprocess.run(["gsutil", "cp", "best_final.pt",
                       "gs://ainm26osl-710-norgesgruppen/models/best_final.pt"])
        # Also upload checkpoint every 10 epochs
        for ckpt in Path(results.save_dir).glob("weights/epoch*.pt"):
            subprocess.run(["gsutil", "cp", str(ckpt),
                           f"gs://ainm26osl-710-norgesgruppen/checkpoints/{ckpt.name}"])
        print("Uploaded to GCS")


if __name__ == "__main__":
    main()
