"""Train YOLO26-x on pseudo-labeled dataset.

Uses the enriched dataset from pseudo_label.py which contains original GT
labels plus high-confidence model predictions for missed annotations.

Usage:
    python train_pseudo.py
"""

import torch
_orig_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(f, *a, **kw)
torch.load = _patched_load

import shutil
import subprocess
from pathlib import Path

from ultralytics import YOLO


# --- Config ---
MODEL_PATH = "/tmp/train/yolo26x.pt"
DATA_YAML = "/tmp/train/data/yolo_pseudo/data.yaml"
EPOCHS = 200
BATCH = 4
IMGSZ = 1280


def main():
    print("=" * 60)
    print("TRAINING ON PSEUDO-LABELED DATASET")
    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA_YAML}")
    print(f"Epochs: {EPOCHS}, Batch: {BATCH}, ImgSz: {IMGSZ}")
    print("=" * 60)

    # Verify pseudo dataset exists
    data_path = Path(DATA_YAML)
    if not data_path.exists():
        print(f"ERROR: {DATA_YAML} not found. Run pseudo_label.py first.")
        return

    # Count labels
    pseudo_dir = data_path.parent
    for split in ["train", "val"]:
        label_dir = pseudo_dir / "labels" / split
        if label_dir.exists():
            n_files = len(list(label_dir.glob("*.txt")))
            n_labels = 0
            for lf in label_dir.glob("*.txt"):
                n_labels += sum(1 for line in open(lf) if line.strip())
            print(f"  {split}: {n_files} images, {n_labels} labels")

    model = YOLO(MODEL_PATH)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        device=0,
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
        # Augmentation — same as train_best.py
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
    )

    # Save best weights
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        shutil.copy2(best_pt, "/tmp/train/best_pseudo.pt")
        size = best_pt.stat().st_size / 1024 / 1024
        print(f"\nbest_pseudo.pt: {size:.1f} MB")

        # Upload to GCS
        try:
            bucket = "gs://ainm26osl-710-norgesgruppen/models"
            subprocess.run(["gsutil", "cp", "/tmp/train/best_pseudo.pt",
                           f"{bucket}/best_pseudo.pt"])
            subprocess.run(["gsutil", "cp", str(best_pt),
                           f"{bucket}/best_pseudo_orig.pt"])
            print("Uploaded to GCS")
        except Exception as e:
            print(f"GCS upload failed: {e}")

    print("\nDONE. Best weights at /tmp/train/best_pseudo.pt")


if __name__ == "__main__":
    main()
