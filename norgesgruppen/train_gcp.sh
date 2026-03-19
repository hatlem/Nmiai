#!/bin/bash
# Train YOLOv8x on GCP Compute Engine with GPU
# Run from: norgesgruppen/ directory
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project ainm26osl-710

set -e

PROJECT="ainm26osl-710"
ZONE="europe-north1-b"
INSTANCE="yolo-train"
BUCKET="gs://${PROJECT}-norgesgruppen"

echo "=== Step 1: Create GCS bucket and upload data ==="
gsutil mb -l europe-north1 "${BUCKET}" 2>/dev/null || true
gsutil -m cp -r data/yolo_dataset/ "${BUCKET}/yolo_dataset/"
gsutil cp train.py convert_coco_to_yolo.py "${BUCKET}/scripts/"

echo "=== Step 2: Create GPU VM ==="
gcloud compute instances create "${INSTANCE}" \
  --project="${PROJECT}" \
  --zone="${ZONE}" \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=100GB \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True" \
  --scopes=storage-full

echo "=== Step 3: Wait for VM to be ready ==="
sleep 60

echo "=== Step 4: Run training on VM ==="
gcloud compute ssh "${INSTANCE}" --zone="${ZONE}" --command="
  # Install deps
  pip install ultralytics==8.1.0

  # Download data from GCS
  mkdir -p /tmp/norgesgruppen/data
  gsutil -m cp -r ${BUCKET}/yolo_dataset/ /tmp/norgesgruppen/data/yolo_dataset/
  gsutil cp ${BUCKET}/scripts/train.py /tmp/norgesgruppen/

  # Update data.yaml path
  cd /tmp/norgesgruppen
  sed -i 's|path:.*|path: /tmp/norgesgruppen/data/yolo_dataset|' data/yolo_dataset/data.yaml

  # Train
  python train.py --data data/yolo_dataset/data.yaml --epochs 300 --batch 8 --imgsz 1280 --device 0

  # Upload results
  gsutil cp best.pt ${BUCKET}/models/best.pt
  gsutil -m cp -r runs/ ${BUCKET}/runs/
"

echo "=== Step 5: Download best model ==="
gsutil cp "${BUCKET}/models/best.pt" best.pt

echo "=== Step 6: Cleanup VM ==="
gcloud compute instances delete "${INSTANCE}" --zone="${ZONE}" --quiet

echo "=== Done! Model saved to best.pt ==="
ls -la best.pt
