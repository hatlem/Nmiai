#!/bin/bash
set -e

echo "=== Installing deps ==="
pip install ultralytics==8.1.0 opencv-python-headless 2>&1 | tail -3

# Patch torch.load for compatibility
cat > /tmp/torch_patch.py << 'EOF'
import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
torch.load = _patched
EOF

echo "=== Downloading data ==="
mkdir -p /tmp/train/data
gsutil -m cp -r gs://ainm26osl-710-norgesgruppen/yolo_dataset/yolo_dataset /tmp/train/data/ 2>&1 | tail -3
gsutil -m cp -r gs://ainm26osl-710-norgesgruppen/yolo_single_class/yolo_single_class /tmp/train/data/ 2>&1 | tail -3
gsutil cp gs://ainm26osl-710-norgesgruppen/scripts/train.py /tmp/train/
gsutil cp gs://ainm26osl-710-norgesgruppen/scripts/src/utils.py /tmp/train/ 2>/dev/null || true

# Fix paths in data.yaml
sed -i 's|path:.*|path: /tmp/train/data/yolo_dataset|' /tmp/train/data/yolo_dataset/data.yaml
sed -i 's|path:.*|path: /tmp/train/data/yolo_single_class|' /tmp/train/data/yolo_single_class/data.yaml

echo "=== Starting training ==="
cd /tmp/train

# ---- TRAIN 1: Single-class detector (for two-stage pipeline) ----
echo ">>> Training single-class detector..."
python3 -c "
import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
torch.load = _patched

import cv2
import numpy as np
from pathlib import Path

# Apply CLAHE to training images (match inference preprocessing)
for split in ['train', 'val']:
    img_dir = Path(f'data/yolo_single_class/images/{split}')
    if not img_dir.exists():
        continue
    marker = img_dir.parent.parent / f'.clahe_{split}'
    if marker.exists():
        continue
    count = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(img_path), img)
        count += 1
    marker.write_text('done')
    print(f'CLAHE {split}: {count} images')

from ultralytics import YOLO
model = YOLO('yolov8x.pt')
results = model.train(
    data='data/yolo_single_class/data.yaml',
    epochs=100,
    batch=16,
    imgsz=1280,
    device=0,
    workers=16,
    patience=30,
    save=True,
    save_period=10,
    val=True,
    plots=True,
    optimizer='AdamW',
    lr0=0.002,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
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
    close_mosaic=15,
)

import shutil
best = Path(results.save_dir) / 'weights' / 'best.pt'
if best.exists():
    shutil.copy2(best, '/tmp/train/best_single_class.pt')
    print(f'Single-class model: {best.stat().st_size / 1024 / 1024:.1f} MB')
" 2>&1 | tee /tmp/train/training_single.log

# ---- TRAIN 2: 356-class detector (for single-stage pipeline) ----
echo ">>> Training 356-class detector..."
python3 -c "
import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
torch.load = _patched

import cv2
from pathlib import Path

# Apply CLAHE
for split in ['train', 'val']:
    img_dir = Path(f'data/yolo_dataset/images/{split}')
    if not img_dir.exists():
        continue
    marker = img_dir.parent.parent / f'.clahe_{split}'
    if marker.exists():
        continue
    count = 0
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        cv2.imwrite(str(img_path), img)
        count += 1
    marker.write_text('done')
    print(f'CLAHE {split}: {count} images')

from ultralytics import YOLO
model = YOLO('yolov8x.pt')
results = model.train(
    data='data/yolo_dataset/data.yaml',
    epochs=100,
    batch=16,
    imgsz=1280,
    device=0,
    workers=16,
    patience=30,
    save=True,
    save_period=10,
    val=True,
    plots=True,
    optimizer='AdamW',
    lr0=0.002,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3,
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
    close_mosaic=15,
)

import shutil
best = Path(results.save_dir) / 'weights' / 'best.pt'
if best.exists():
    shutil.copy2(best, '/tmp/train/best_356class.pt')
    print(f'356-class model: {best.stat().st_size / 1024 / 1024:.1f} MB')
" 2>&1 | tee /tmp/train/training_356.log

echo "=== Uploading results ==="
gsutil cp /tmp/train/best_single_class.pt gs://ainm26osl-710-norgesgruppen/models/best_single_class.pt 2>&1
gsutil cp /tmp/train/best_356class.pt gs://ainm26osl-710-norgesgruppen/models/best_356class.pt 2>&1
gsutil -m cp -r /tmp/train/runs/ gs://ainm26osl-710-norgesgruppen/runs/ 2>&1

echo "=== TRAINING COMPLETE ==="
echo "Models uploaded:"
echo "  - best_single_class.pt  (for run_twostage.py)"
echo "  - best_356class.pt      (for run.py)"
