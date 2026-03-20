#!/bin/bash
# Deploy training jobs to GCP Compute Engine with GPU
#
# Creates a spot/preemptible VM, uploads data, runs training, downloads results.
# Runs EfficientNet-B3 classifier first (fast ~30min), then YOLO11x (slow ~4-8h).
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project ainm26osl-710
#
# Usage:
#   cd norgesgruppen/
#   bash deploy_training.sh              # Run both jobs
#   bash deploy_training.sh classifier   # Only EfficientNet-B3
#   bash deploy_training.sh yolo         # Only YOLO11x
#   bash deploy_training.sh cleanup      # Delete VM and bucket data

set -euo pipefail

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT="ainm26osl-710"
ZONE="europe-north1-b"
INSTANCE="nmiai-train-gpu"
MACHINE_TYPE="g2-standard-8"         # 8 vCPU, 32 GB RAM, 1x NVIDIA L4
ACCELERATOR="type=nvidia-l4,count=1"
BOOT_DISK_SIZE="200GB"
BUCKET="gs://${PROJECT}-norgesgruppen"
REMOTE_DIR="/tmp/nmiai"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-all}"  # all, classifier, yolo, cleanup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }
err()  { echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
if [ "$MODE" = "cleanup" ]; then
    log "Deleting VM ${INSTANCE}..."
    gcloud compute instances delete "${INSTANCE}" \
        --project="${PROJECT}" --zone="${ZONE}" --quiet 2>/dev/null || warn "VM not found"
    log "Cleanup complete."
    exit 0
fi

# ---------------------------------------------------------------------------
# Step 1: Create GCS bucket and upload data
# ---------------------------------------------------------------------------
log "=== Step 1: Upload data to GCS ==="

gsutil mb -l europe-north1 "${BUCKET}" 2>/dev/null || true

# Upload training data (parallel, skip existing)
log "Uploading YOLO dataset..."
gsutil -m rsync -r "${LOCAL_DIR}/data/yolo_dataset/" "${BUCKET}/yolo_dataset/"

log "Uploading training images and annotations..."
gsutil -m rsync -r "${LOCAL_DIR}/data/train/" "${BUCKET}/train/"

log "Uploading product reference images..."
gsutil -m rsync -r "${LOCAL_DIR}/data/NM_NGD_product_images/" "${BUCKET}/NM_NGD_product_images/"

# Upload scripts
log "Uploading training scripts..."
gsutil cp "${LOCAL_DIR}/finetune_classifier.py" "${BUCKET}/scripts/"
gsutil cp "${LOCAL_DIR}/train_yolo11.py" "${BUCKET}/scripts/"

# Upload existing model weights if available (for fine-tuning)
if [ -f "${LOCAL_DIR}/models/efficientnet_b3_weights.pt" ]; then
    log "Uploading existing EfficientNet weights..."
    gsutil cp "${LOCAL_DIR}/models/efficientnet_b3_weights.pt" "${BUCKET}/models/"
fi
if [ -f "${LOCAL_DIR}/models/product_embeddings.npy" ]; then
    gsutil cp "${LOCAL_DIR}/models/product_embeddings.npy" "${BUCKET}/models/"
fi
if [ -f "${LOCAL_DIR}/models/embedding_config.json" ]; then
    gsutil cp "${LOCAL_DIR}/models/embedding_config.json" "${BUCKET}/models/"
fi

# ---------------------------------------------------------------------------
# Step 2: Create GPU VM (spot instance for cost savings)
# ---------------------------------------------------------------------------
log "=== Step 2: Create GPU VM (spot instance) ==="

# Check if VM already exists
if gcloud compute instances describe "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}" &>/dev/null; then
    warn "VM ${INSTANCE} already exists. Reusing it."
else
    gcloud compute instances create "${INSTANCE}" \
        --project="${PROJECT}" \
        --zone="${ZONE}" \
        --machine-type="${MACHINE_TYPE}" \
        --accelerator="${ACCELERATOR}" \
        --maintenance-policy=TERMINATE \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --boot-disk-size="${BOOT_DISK_SIZE}" \
        --boot-disk-type=pd-ssd \
        --image-family=pytorch-latest-gpu \
        --image-project=deeplearning-platform-release \
        --metadata="install-nvidia-driver=True" \
        --scopes=storage-full \
        --no-restart-on-failure

    log "VM created. Waiting for it to boot..."
    sleep 30
fi

# Wait until SSH is available
log "Waiting for SSH..."
for i in $(seq 1 30); do
    if gcloud compute ssh "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}" \
        --command="echo ready" 2>/dev/null; then
        break
    fi
    if [ "$i" -eq 30 ]; then
        err "SSH timeout after 5 minutes"
    fi
    sleep 10
done

log "VM is ready."

# ---------------------------------------------------------------------------
# Step 3: Install dependencies on VM
# ---------------------------------------------------------------------------
log "=== Step 3: Install dependencies ==="

gcloud compute ssh "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}" --command="
    set -e
    echo 'Installing Python packages...'
    pip install -q --upgrade pip
    pip install -q ultralytics torch torchvision timm opencv-python-headless numpy pillow

    echo 'Checking GPU...'
    python3 -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}\")'

    echo 'Dependencies installed.'
"

# ---------------------------------------------------------------------------
# Step 4: Download data from GCS to VM
# ---------------------------------------------------------------------------
log "=== Step 4: Download data to VM ==="

gcloud compute ssh "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}" --command="
    set -e
    mkdir -p ${REMOTE_DIR}/data ${REMOTE_DIR}/models

    echo 'Downloading YOLO dataset...'
    gsutil -m cp -r ${BUCKET}/yolo_dataset/ ${REMOTE_DIR}/data/yolo_dataset/

    echo 'Downloading training images...'
    gsutil -m cp -r ${BUCKET}/train/ ${REMOTE_DIR}/data/train/

    echo 'Downloading product images...'
    gsutil -m cp -r ${BUCKET}/NM_NGD_product_images/ ${REMOTE_DIR}/data/NM_NGD_product_images/

    echo 'Downloading scripts...'
    gsutil cp ${BUCKET}/scripts/finetune_classifier.py ${REMOTE_DIR}/
    gsutil cp ${BUCKET}/scripts/train_yolo11.py ${REMOTE_DIR}/

    # Download existing weights if available
    gsutil cp ${BUCKET}/models/efficientnet_b3_weights.pt ${REMOTE_DIR}/models/ 2>/dev/null || true
    gsutil cp ${BUCKET}/models/product_embeddings.npy ${REMOTE_DIR}/models/ 2>/dev/null || true
    gsutil cp ${BUCKET}/models/embedding_config.json ${REMOTE_DIR}/models/ 2>/dev/null || true

    # Fix YOLO data.yaml path
    sed -i 's|path:.*|path: ${REMOTE_DIR}/data/yolo_dataset|' ${REMOTE_DIR}/data/yolo_dataset/data.yaml

    echo 'Data ready.'
    du -sh ${REMOTE_DIR}/data/*/
"

# ---------------------------------------------------------------------------
# Step 5: Run EfficientNet-B3 classifier training
# ---------------------------------------------------------------------------
if [ "$MODE" = "all" ] || [ "$MODE" = "classifier" ]; then
    log "=== Step 5: Training EfficientNet-B3 classifier ==="

    gcloud compute ssh "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}" --command="
        set -e
        cd ${REMOTE_DIR}
        echo 'Starting EfficientNet-B3 fine-tuning...'
        python3 finetune_classifier.py 2>&1 | tee classifier_train.log

        echo 'Uploading classifier results to GCS...'
        gsutil cp models/efficientnet_b3_weights.pt ${BUCKET}/models/efficientnet_b3_weights.pt
        gsutil cp models/product_embeddings.npy ${BUCKET}/models/product_embeddings.npy
        gsutil cp models/embedding_config.json ${BUCKET}/models/embedding_config.json 2>/dev/null || true
        gsutil cp classifier_train.log ${BUCKET}/logs/classifier_train.log

        echo 'Classifier training complete!'
    "

    # Download classifier results locally
    log "Downloading classifier results..."
    mkdir -p "${LOCAL_DIR}/models"
    gsutil cp "${BUCKET}/models/efficientnet_b3_weights.pt" "${LOCAL_DIR}/models/efficientnet_b3_weights.pt"
    gsutil cp "${BUCKET}/models/product_embeddings.npy" "${LOCAL_DIR}/models/product_embeddings.npy"
    gsutil cp "${BUCKET}/models/embedding_config.json" "${LOCAL_DIR}/models/embedding_config.json" 2>/dev/null || true

    log "Classifier results downloaded to models/"
    ls -lh "${LOCAL_DIR}/models/efficientnet_b3_weights.pt" "${LOCAL_DIR}/models/product_embeddings.npy"
fi

# ---------------------------------------------------------------------------
# Step 6: Run YOLO11x training
# ---------------------------------------------------------------------------
if [ "$MODE" = "all" ] || [ "$MODE" = "yolo" ]; then
    log "=== Step 6: Training YOLO11x detector ==="
    warn "This will take 4-8 hours. The VM is a spot instance -- if preempted, re-run with: bash deploy_training.sh yolo"

    gcloud compute ssh "${INSTANCE}" --project="${PROJECT}" --zone="${ZONE}" --command="
        set -e
        cd ${REMOTE_DIR}
        echo 'Starting YOLO11x training...'
        python3 train_yolo11.py \
            --model yolo11x.pt \
            --data data/yolo_dataset/data.yaml \
            --epochs 300 \
            --batch 4 \
            --imgsz 1280 \
            --device 0 2>&1 | tee yolo_train.log

        echo 'Uploading YOLO results to GCS...'
        gsutil cp yolo11_best.onnx ${BUCKET}/models/yolo11_best.onnx 2>/dev/null || true
        gsutil cp yolo11_best.pt ${BUCKET}/models/yolo11_best.pt 2>/dev/null || true
        gsutil -m cp -r runs/ ${BUCKET}/runs/ 2>/dev/null || true
        gsutil cp yolo_train.log ${BUCKET}/logs/yolo_train.log

        echo 'YOLO training complete!'
    "

    # Download YOLO results locally
    log "Downloading YOLO results..."
    gsutil cp "${BUCKET}/models/yolo11_best.onnx" "${LOCAL_DIR}/yolo11_best.onnx" 2>/dev/null || warn "No ONNX file found"
    gsutil cp "${BUCKET}/models/yolo11_best.pt" "${LOCAL_DIR}/yolo11_best.pt" 2>/dev/null || warn "No PT file found"

    log "YOLO results downloaded."
    ls -lh "${LOCAL_DIR}/yolo11_best"* 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Step 7: Summary and optional cleanup
# ---------------------------------------------------------------------------
log "=== Training Complete ==="
echo ""
echo "Results:"
echo "  Classifier:  models/efficientnet_b3_weights.pt"
echo "               models/product_embeddings.npy"
echo "  Detector:    yolo11_best.onnx"
echo "               yolo11_best.pt"
echo ""
echo "GCS backup:    ${BUCKET}/models/"
echo "Training logs: ${BUCKET}/logs/"
echo ""
echo "To delete the VM (saves cost):"
echo "  bash deploy_training.sh cleanup"
echo ""
echo "To check training progress (if running in background):"
echo "  gcloud compute ssh ${INSTANCE} --zone=${ZONE} --command='tail -50 ${REMOTE_DIR}/yolo_train.log'"
