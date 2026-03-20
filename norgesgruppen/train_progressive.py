#!/usr/bin/env python3
"""
Progressive resize pretraining + fine-tuning for grocery product detection.
Fallback approach since SKU-110K download failed.

Strategy: Progressive resolution training with heavy augmentation
  Phase 1: 640px, heavy augmentation, 50 epochs (warm up on domain)
  Phase 2: 960px, moderate augmentation, 30 epochs (intermediate)
  Phase 3: 1280px, full fine-tuning, 200 epochs (final high-res)

Run on GCP VM nmiai-train-fast (L4 GPU, 23GB VRAM).
"""

import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(*a, **kw)
torch.load = _patched

import os
import sys
import shutil
from pathlib import Path
from ultralytics import YOLO

WEIGHTS = "/home/andreashatlem/train/yolo26x.pt"
DATA_YAML = "/home/andreashatlem/train/data/yolo_dataset/data.yaml"
PROJECT_DIR = "/home/andreashatlem/train/runs"
SEP = "=" * 60


def run_phase(phase_name, model_path, imgsz, epochs, batch, lr0, lrf,
              mosaic, mixup, copy_paste, degrees, translate, scale,
              flipud, erasing, close_mosaic, patience, cos_lr=True,
              freeze=None, warmup_epochs=3):
    print()
    print(SEP)
    print("PHASE: " + phase_name)
    print("  Model: " + str(model_path))
    print("  imgsz=%d, epochs=%d, batch=%d" % (imgsz, epochs, batch))
    print("  lr0=%s, lrf=%s" % (lr0, lrf))
    print(SEP)
    print()

    model = YOLO(model_path)
    train_args = dict(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        lrf=lrf,
        cos_lr=cos_lr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        degrees=degrees,
        translate=translate,
        scale=scale,
        flipud=flipud,
        erasing=erasing,
        close_mosaic=close_mosaic,
        patience=patience,
        warmup_epochs=warmup_epochs,
        project=PROJECT_DIR,
        name=phase_name,
        exist_ok=True,
        save=True,
        save_period=10,
        device=0,
        workers=8,
        amp=True,
        verbose=True,
        plots=True,
        val=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
    )
    if freeze is not None:
        train_args["freeze"] = freeze

    results = model.train(**train_args)

    best_path = Path(PROJECT_DIR) / phase_name / "weights" / "best.pt"
    if best_path.exists():
        print("Best weights saved to: " + str(best_path))
        return str(best_path)
    last_path = Path(PROJECT_DIR) / phase_name / "weights" / "last.pt"
    print("Using last weights: " + str(last_path))
    return str(last_path)


def main():
    print(SEP)
    print("PROGRESSIVE RESIZE PRETRAINING + FINE-TUNING")
    print("Base weights: " + WEIGHTS)
    print("Dataset: " + DATA_YAML)
    print(SEP)

    # Phase 1: 640px - Heavy augmentation, freeze first 10 backbone layers
    # batch=4 is safe for YOLO26-x on L4 (23GB VRAM)
    p1 = run_phase(
        "phase1_640", WEIGHTS,
        imgsz=640, epochs=50, batch=4,
        lr0=0.01, lrf=0.1,
        mosaic=1.0, mixup=0.3, copy_paste=0.3,
        degrees=10.0, translate=0.2, scale=0.9,
        flipud=0.3, erasing=0.4,
        close_mosaic=10, patience=20,
        warmup_epochs=5, freeze=10,
    )

    # Phase 2: 960px - Moderate augmentation, unfreeze all layers
    p2 = run_phase(
        "phase2_960", p1,
        imgsz=960, epochs=30, batch=2,
        lr0=0.005, lrf=0.1,
        mosaic=1.0, mixup=0.2, copy_paste=0.2,
        degrees=5.0, translate=0.15, scale=0.5,
        flipud=0.2, erasing=0.3,
        close_mosaic=5, patience=15,
        warmup_epochs=3,
    )

    # Phase 3: 1280px - Final fine-tuning, lighter augmentation
    p3 = run_phase(
        "phase3_1280", p2,
        imgsz=1280, epochs=200, batch=2,
        lr0=0.002, lrf=0.01,
        mosaic=1.0, mixup=0.15, copy_paste=0.15,
        degrees=3.0, translate=0.1, scale=0.3,
        flipud=0.1, erasing=0.2,
        close_mosaic=20, patience=40,
        warmup_epochs=3,
    )

    # Copy final best weights
    final_dest = "/home/andreashatlem/train/best_progressive.pt"
    shutil.copy2(p3, final_dest)

    # Upload to GCS
    print()
    print("Uploading to GCS...")
    os.system("gsutil cp " + final_dest + " gs://ainm26osl-710-norgesgruppen/models/best_progressive.pt")
    os.system("gsutil -m cp -r " + PROJECT_DIR + "/ gs://ainm26osl-710-norgesgruppen/runs_progressive/")

    print()
    print(SEP)
    print("TRAINING COMPLETE!")
    print("Final weights: " + final_dest)
    print("GCS: gs://ainm26osl-710-norgesgruppen/models/best_progressive.pt")
    print(SEP)


if __name__ == "__main__":
    main()
