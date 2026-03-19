#!/bin/bash
# Train both YOLO11x detector and EfficientNet-B3 classifier on GCP
#
# Usage:
#   1. SSH into GCP VM with GPU:
#      gcloud compute ssh <vm-name> --zone europe-north1-a
#
#   2. Clone repo + upload data:
#      git clone <repo-url> && cd nmiai/norgesgruppen
#      # Upload data/ directory (images + annotations)
#
#   3. Run this script:
#      bash train_on_gcp.sh
#
# Or deploy as Cloud Run Job for each training task.

set -e

echo "=== NorgesGruppen GCP Training ==="
echo "Start: $(date)"

# Install dependencies
pip install -q ultralytics torch torchvision timm opencv-python-headless numpy pillow

# Ensure data exists
if [ ! -d "data/yolo_dataset" ]; then
    echo "ERROR: data/yolo_dataset/ not found. Upload training data first."
    exit 1
fi

# --- Train YOLO11x detector ---
echo ""
echo "========================================="
echo "  STEP 1: Training YOLO11x detector"
echo "========================================="
python train_yolo11.py \
    --model yolo11x.pt \
    --data data/yolo_dataset/data.yaml \
    --epochs 300 \
    --batch 8 \
    --imgsz 1280 \
    --device 0

echo "YOLO11x training complete!"
echo "Outputs: best_yolo11x.pt, best_yolo11x.onnx"

# --- Train classifier ---
echo ""
echo "========================================="
echo "  STEP 2: Training EfficientNet-B3 classifier (ArcFace)"
echo "========================================="
python finetune_classifier.py

echo "Classifier training complete!"
echo "Outputs: models/efficientnet_b3_weights.pt, models/product_embeddings.npy"

echo ""
echo "=== All training complete! ==="
echo "End: $(date)"
echo ""
echo "Files to download:"
echo "  - best_yolo11x.pt (detector)"
echo "  - best_yolo11x.onnx (detector ONNX)"
echo "  - models/efficientnet_b3_weights.pt (classifier)"
echo "  - models/product_embeddings.npy (embeddings)"
echo "  - models/embedding_config.json (config)"
