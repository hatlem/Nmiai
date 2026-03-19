#!/bin/bash
set -e

echo "============================================"
echo "=== YOLO26 + RT-DETR Training on A100    ==="
echo "============================================"

echo "=== Installing deps ==="
pip install "ultralytics>=8.3.0" opencv-python-headless onnx onnxsim 2>&1 | tail -5

echo "=== Downloading data ==="
mkdir -p /tmp/train/data
gsutil -m cp -r gs://ainm26osl-710-norgesgruppen/yolo_single_class/yolo_single_class /tmp/train/data/ 2>&1 | tail -3
gsutil -m cp -r gs://ainm26osl-710-norgesgruppen/yolo_dataset/yolo_dataset /tmp/train/data/ 2>&1 | tail -3

# Fix paths in data.yaml
sed -i 's|path:.*|path: /tmp/train/data/yolo_single_class|' /tmp/train/data/yolo_single_class/data.yaml
sed -i 's|path:.*|path: /tmp/train/data/yolo_dataset|' /tmp/train/data/yolo_dataset/data.yaml

cd /tmp/train

# ============================================================
# TRAIN 1: YOLO26-x single-class detector
# ============================================================
echo ""
echo ">>> [1/3] Training YOLO26-x single-class detector..."
python3 -c "
import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
torch.load = _patched

import cv2
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
model = YOLO('yolo26x.pt')
results = model.train(
    data='data/yolo_single_class/data.yaml',
    epochs=150,
    batch=16,
    imgsz=1280,
    device=0,
    workers=16,
    patience=40,
    save=True,
    save_period=25,
    val=True,
    plots=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
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
    close_mosaic=30,
)

import shutil
best = Path(results.save_dir) / 'weights' / 'best.pt'
if best.exists():
    shutil.copy2(best, '/tmp/train/yolo26x_single.pt')
    print(f'YOLO26-x single-class model: {best.stat().st_size / 1024 / 1024:.1f} MB')

    # Export to ONNX
    export_model = YOLO(str(best))
    onnx_path = export_model.export(format='onnx', imgsz=1280, simplify=True, opset=17)
    shutil.copy2(onnx_path, '/tmp/train/yolo26x_single.onnx')
    print(f'ONNX exported: {Path(onnx_path).stat().st_size / 1024 / 1024:.1f} MB')

# Print metrics
print('\\n=== YOLO26-x Single-Class Results ===')
print(f'mAP50: {results.results_dict.get(\"metrics/mAP50(B)\", \"N/A\")}')
print(f'mAP50-95: {results.results_dict.get(\"metrics/mAP50-95(B)\", \"N/A\")}')
" 2>&1 | tee /tmp/train/training_yolo26_single.log

# ============================================================
# TRAIN 2: YOLO26-x 357-class detector
# ============================================================
echo ""
echo ">>> [2/3] Training YOLO26-x 357-class detector..."
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
model = YOLO('yolo26x.pt')
results = model.train(
    data='data/yolo_dataset/data.yaml',
    epochs=150,
    batch=16,
    imgsz=1280,
    device=0,
    workers=16,
    patience=40,
    save=True,
    save_period=25,
    val=True,
    plots=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
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
    close_mosaic=30,
)

import shutil
best = Path(results.save_dir) / 'weights' / 'best.pt'
if best.exists():
    shutil.copy2(best, '/tmp/train/yolo26x_multi.pt')
    print(f'YOLO26-x 357-class model: {best.stat().st_size / 1024 / 1024:.1f} MB')

    # Export to ONNX
    export_model = YOLO(str(best))
    onnx_path = export_model.export(format='onnx', imgsz=1280, simplify=True, opset=17)
    shutil.copy2(onnx_path, '/tmp/train/yolo26x_multi.onnx')
    print(f'ONNX exported: {Path(onnx_path).stat().st_size / 1024 / 1024:.1f} MB')

print('\\n=== YOLO26-x 357-Class Results ===')
print(f'mAP50: {results.results_dict.get(\"metrics/mAP50(B)\", \"N/A\")}')
print(f'mAP50-95: {results.results_dict.get(\"metrics/mAP50-95(B)\", \"N/A\")}')
" 2>&1 | tee /tmp/train/training_yolo26_multi.log

# ============================================================
# TRAIN 3: RT-DETR-x single-class (for ensemble)
# ============================================================
echo ""
echo ">>> [3/3] Training RT-DETR-x single-class detector..."
python3 -c "
import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
torch.load = _patched

from pathlib import Path
from ultralytics import RTDETR

model = RTDETR('rtdetr-x.pt')
results = model.train(
    data='data/yolo_single_class/data.yaml',
    epochs=100,
    batch=16,
    imgsz=1280,
    device=0,
    workers=16,
    patience=30,
    save=True,
    save_period=25,
    val=True,
    plots=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=5,
)

import shutil
best = Path(results.save_dir) / 'weights' / 'best.pt'
if best.exists():
    shutil.copy2(best, '/tmp/train/rtdetr_best.pt')
    print(f'RT-DETR-x single-class model: {best.stat().st_size / 1024 / 1024:.1f} MB')

print('\\n=== RT-DETR-x Single-Class Results ===')
print(f'mAP50: {results.results_dict.get(\"metrics/mAP50(B)\", \"N/A\")}')
print(f'mAP50-95: {results.results_dict.get(\"metrics/mAP50-95(B)\", \"N/A\")}')
" 2>&1 | tee /tmp/train/training_rtdetr.log

# ============================================================
# Upload all results to GCS
# ============================================================
echo ""
echo "=== Uploading results to GCS ==="

# Models
gsutil cp /tmp/train/yolo26x_single.onnx gs://ainm26osl-710-norgesgruppen/models/yolo26x_single.onnx 2>&1
gsutil cp /tmp/train/yolo26x_multi.onnx gs://ainm26osl-710-norgesgruppen/models/yolo26x_multi.onnx 2>&1
gsutil cp /tmp/train/rtdetr_best.pt gs://ainm26osl-710-norgesgruppen/models/rtdetr_best.pt 2>&1

# Also upload .pt weights as backup
gsutil cp /tmp/train/yolo26x_single.pt gs://ainm26osl-710-norgesgruppen/models/yolo26x_single.pt 2>&1
gsutil cp /tmp/train/yolo26x_multi.pt gs://ainm26osl-710-norgesgruppen/models/yolo26x_multi.pt 2>&1

# Training logs and plots
gsutil -m cp -r /tmp/train/runs/ gs://ainm26osl-710-norgesgruppen/runs_yolo26/ 2>&1
gsutil cp /tmp/train/training_*.log gs://ainm26osl-710-norgesgruppen/runs_yolo26/ 2>&1

echo ""
echo "============================================"
echo "=== ALL TRAINING COMPLETE                ==="
echo "============================================"
echo ""
echo "Models uploaded to gs://ainm26osl-710-norgesgruppen/models/:"
echo "  - yolo26x_single.onnx  (single-class, for two-stage pipeline)"
echo "  - yolo26x_multi.onnx   (357-class, for single-stage pipeline)"
echo "  - rtdetr_best.pt       (RT-DETR single-class, for ensemble)"
echo "  - yolo26x_single.pt    (backup .pt weights)"
echo "  - yolo26x_multi.pt     (backup .pt weights)"
echo ""
echo "Training logs: gs://ainm26osl-710-norgesgruppen/runs_yolo26/"
