"""Research-backed multi-class YOLOv8x — ALL findings applied.

Consensus from 5 research agents (EN, DE+FR, ZH+JA+KO, RU+ES+HI, Kaggle):
- Reduced augmentation (was over-augmented for 248 images)
- cos_lr=True for +1-2% mAP
- label_smoothing=0.05
- sqrt-weighted oversampling (not flat)
- close_mosaic=30 (more clean fine-tuning)
- Fine-tune from best multi-class model (0.799 mAP50)
"""
import torch
_orig = torch.load
def _p(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig(f, *a, **kw)
torch.load = _p

import shutil, subprocess
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np


def oversample_sqrt(data_yaml, target=50):
    d = Path(data_yaml).parent
    if (d / ".ov_research").exists():
        print("Oversample done")
        return
    labels_dir = d / "labels" / "train"
    images_dir = d / "images" / "train"
    if not labels_dir.exists():
        return

    class_counts = Counter()
    class_files = defaultdict(list)
    for lf in sorted(labels_dir.glob("*.txt")):
        classes_in = set()
        with open(lf) as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    cid = int(p[0])
                    class_counts[cid] += 1
                    classes_in.add(cid)
        for cid in classes_in:
            if lf not in class_files[cid]:
                class_files[cid].append(lf)

    copies = 0
    for cid, files in class_files.items():
        count = class_counts[cid]
        if count >= target:
            continue
        n = min(int(np.sqrt(target / max(count, 1))), 5)
        seen = set()
        for i in range(n):
            src = files[i % len(files)]
            if src.stem in seen:
                continue
            seen.add(src.stem)
            src_img = None
            for ext in [".jpg", ".jpeg", ".png"]:
                c = images_dir / (src.stem + ext)
                if c.exists():
                    src_img = c
                    break
            if not src_img:
                continue
            name = f"{src.stem}_ovr_{cid}_{i}"
            di = images_dir / f"{name}{src_img.suffix}"
            dl = labels_dir / f"{name}.txt"
            if not di.exists():
                shutil.copy2(src_img, di)
                shutil.copy2(src, dl)
                copies += 1
    print(f"Sqrt-oversampled: {copies} copies")
    (d / ".ov_research").write_text("done")


def apply_clahe(data_yaml):
    import cv2
    d = Path(data_yaml).parent
    if (d / ".clahe_r").exists():
        print("CLAHE done")
        return
    for split in ["train", "val"]:
        img_dir = d / "images" / split
        if not img_dir.exists():
            continue
        n = 0
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cv2.imwrite(str(p), cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR))
            n += 1
        print(f"CLAHE {split}: {n}")
    (d / ".clahe_r").write_text("done")


if __name__ == "__main__":
    data = "yolo_dataset/data.yaml"
    apply_clahe(data)
    oversample_sqrt(data, target=50)

    from ultralytics import YOLO
    model = YOLO("best_mc.pt")

    results = model.train(
        data=data,
        epochs=300,
        batch=16,
        imgsz=1280,
        device=0,
        workers=8,
        patience=60,
        save=True,
        save_period=5,
        val=True,
        plots=True,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.005,
        weight_decay=0.0005,
        warmup_epochs=10,
        cos_lr=True,
        label_smoothing=0.05,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.1,
        scale=0.5,
        shear=1.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.15,
        erasing=0.15,
        close_mosaic=30,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, "best_research.pt")
        subprocess.run(["gsutil", "cp", "best_research.pt",
                       "gs://ainm26osl-710-norgesgruppen/models/best_research.pt"])
        print("Uploaded best_research.pt")
