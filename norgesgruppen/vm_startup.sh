#!/bin/bash
# Startup script for GPU training VM
# Trains BOTH single-class (for two-stage) and 356-class (for single-stage)
# with CLAHE preprocessing to match inference
set -e

echo "=== Installing dependencies ==="
pip install ultralytics==8.1.0 opencv-python-headless 2>&1 | tail -3

echo "=== Downloading data from GCS ==="
mkdir -p /tmp/train/data
gsutil -m cp -r gs://ainm26osl-710-norgesgruppen/yolo_dataset/ /tmp/train/data/yolo_dataset/
gsutil -m cp -r gs://ainm26osl-710-norgesgruppen/yolo_single_class/ /tmp/train/data/yolo_single_class/ 2>/dev/null || true
gsutil cp gs://ainm26osl-710-norgesgruppen/scripts/train.py /tmp/train/

echo "=== Fixing data.yaml paths ==="
sed -i "s|path:.*|path: /tmp/train/data/yolo_dataset|" /tmp/train/data/yolo_dataset/data.yaml
if [ -f /tmp/train/data/yolo_single_class/data.yaml ]; then
    sed -i "s|path:.*|path: /tmp/train/data/yolo_single_class|" /tmp/train/data/yolo_single_class/data.yaml
fi

echo "=== Starting training ==="
cd /tmp/train

# Train single-class first (faster, needed for two-stage)
if [ -f data/yolo_single_class/data.yaml ]; then
    echo ">>> Training single-class detector..."
    python train.py \
      --data data/yolo_single_class/data.yaml \
      --model yolov8x.pt \
      --epochs 300 \
      --batch 4 \
      --imgsz 1280 \
      --device 0

    mv best.pt best_single_class.pt 2>/dev/null || true
fi

# Then train 356-class
echo ">>> Training 356-class detector..."
python train.py \
  --data data/yolo_dataset/data.yaml \
  --model yolov8x.pt \
  --epochs 300 \
  --batch 4 \
  --imgsz 1280 \
  --device 0

mv best.pt best_356class.pt 2>/dev/null || true

echo "=== Uploading results ==="
gsutil cp best_single_class.pt gs://ainm26osl-710-norgesgruppen/models/best_single_class.pt 2>/dev/null || true
gsutil cp best_356class.pt gs://ainm26osl-710-norgesgruppen/models/best_356class.pt 2>/dev/null || true
gsutil -m cp -r runs/ gs://ainm26osl-710-norgesgruppen/runs/

echo "=== TRAINING COMPLETE ==="
