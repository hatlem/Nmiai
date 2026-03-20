#!/bin/bash
# Simpler alternative: submit training as Vertex AI Custom Training Jobs
#
# Benefits over deploy_training.sh:
#   - No VM management (auto-creates and destroys)
#   - Built-in GPU scheduling and retry
#   - Logs in Cloud Console
#   - No SSH needed
#
# Prerequisites:
#   gcloud auth login
#   gcloud config set project ainm26osl-710
#   Data already uploaded to GCS (run deploy_training.sh step 1, or manually)
#
# Usage:
#   bash train_vertex_jobs.sh upload      # Upload data to GCS only
#   bash train_vertex_jobs.sh classifier  # Submit classifier job
#   bash train_vertex_jobs.sh yolo        # Submit YOLO job
#   bash train_vertex_jobs.sh both        # Submit both sequentially
#   bash train_vertex_jobs.sh download    # Download results from GCS

set -euo pipefail

PROJECT="ainm26osl-710"
REGION="europe-north1"
BUCKET="gs://${PROJECT}-norgesgruppen"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-both}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $*"; }

# ---------------------------------------------------------------------------
# Upload data to GCS
# ---------------------------------------------------------------------------
upload_data() {
    log "Uploading data to GCS..."
    gsutil mb -l "${REGION}" "${BUCKET}" 2>/dev/null || true

    gsutil -m rsync -r "${LOCAL_DIR}/data/yolo_dataset/" "${BUCKET}/yolo_dataset/"
    gsutil -m rsync -r "${LOCAL_DIR}/data/train/" "${BUCKET}/train/"
    gsutil -m rsync -r "${LOCAL_DIR}/data/NM_NGD_product_images/" "${BUCKET}/NM_NGD_product_images/"
    gsutil cp "${LOCAL_DIR}/finetune_classifier.py" "${BUCKET}/scripts/"
    gsutil cp "${LOCAL_DIR}/train_yolo11.py" "${BUCKET}/scripts/"

    # Upload existing weights
    [ -f "${LOCAL_DIR}/models/efficientnet_b3_weights.pt" ] && \
        gsutil cp "${LOCAL_DIR}/models/efficientnet_b3_weights.pt" "${BUCKET}/models/"
    [ -f "${LOCAL_DIR}/models/product_embeddings.npy" ] && \
        gsutil cp "${LOCAL_DIR}/models/product_embeddings.npy" "${BUCKET}/models/"
    [ -f "${LOCAL_DIR}/models/embedding_config.json" ] && \
        gsutil cp "${LOCAL_DIR}/models/embedding_config.json" "${BUCKET}/models/"

    log "Upload complete."
}

# ---------------------------------------------------------------------------
# Submit classifier training job
# ---------------------------------------------------------------------------
submit_classifier() {
    local JOB_NAME="classifier-finetune-${TIMESTAMP}"
    log "Submitting classifier training job: ${JOB_NAME}"

    gcloud ai custom-jobs create \
        --project="${PROJECT}" \
        --region="${REGION}" \
        --display-name="${JOB_NAME}" \
        --worker-pool-spec="\
machine-type=n1-standard-8,\
accelerator-type=NVIDIA_TESLA_T4,\
accelerator-count=1,\
replica-count=1,\
container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py311:latest" \
        --args="\
-c,\
set -e && \
pip install -q timm torchvision pillow && \
mkdir -p /tmp/nmiai/data /tmp/nmiai/models && \
gsutil -m cp -r ${BUCKET}/train/ /tmp/nmiai/data/train/ && \
gsutil -m cp -r ${BUCKET}/NM_NGD_product_images/ /tmp/nmiai/data/NM_NGD_product_images/ && \
gsutil cp ${BUCKET}/scripts/finetune_classifier.py /tmp/nmiai/ && \
gsutil cp ${BUCKET}/models/efficientnet_b3_weights.pt /tmp/nmiai/models/ 2>/dev/null || true && \
gsutil cp ${BUCKET}/models/product_embeddings.npy /tmp/nmiai/models/ 2>/dev/null || true && \
gsutil cp ${BUCKET}/models/embedding_config.json /tmp/nmiai/models/ 2>/dev/null || true && \
cd /tmp/nmiai && python3 finetune_classifier.py && \
gsutil cp models/efficientnet_b3_weights.pt ${BUCKET}/models/efficientnet_b3_weights.pt && \
gsutil cp models/product_embeddings.npy ${BUCKET}/models/product_embeddings.npy && \
gsutil cp models/embedding_config.json ${BUCKET}/models/embedding_config.json 2>/dev/null || true"

    log "Classifier job submitted. Monitor at:"
    echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT}"
    echo ""
    echo "  gcloud ai custom-jobs list --project=${PROJECT} --region=${REGION} --filter='displayName:classifier'"
}

# ---------------------------------------------------------------------------
# Submit YOLO training job
# ---------------------------------------------------------------------------
submit_yolo() {
    local JOB_NAME="yolo11x-train-${TIMESTAMP}"
    log "Submitting YOLO11x training job: ${JOB_NAME}"

    gcloud ai custom-jobs create \
        --project="${PROJECT}" \
        --region="${REGION}" \
        --display-name="${JOB_NAME}" \
        --worker-pool-spec="\
machine-type=g2-standard-8,\
accelerator-type=NVIDIA_L4,\
accelerator-count=1,\
replica-count=1,\
container-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py311:latest" \
        --args="\
-c,\
set -e && \
pip install -q ultralytics opencv-python-headless && \
mkdir -p /tmp/nmiai/data && \
gsutil -m cp -r ${BUCKET}/yolo_dataset/ /tmp/nmiai/data/yolo_dataset/ && \
gsutil cp ${BUCKET}/scripts/train_yolo11.py /tmp/nmiai/ && \
cd /tmp/nmiai && \
sed -i 's|path:.*|path: /tmp/nmiai/data/yolo_dataset|' data/yolo_dataset/data.yaml && \
python3 train_yolo11.py --model yolo11x.pt --data data/yolo_dataset/data.yaml --epochs 300 --batch 4 --imgsz 1280 --device 0 && \
gsutil cp yolo11_best.onnx ${BUCKET}/models/yolo11_best.onnx 2>/dev/null || true && \
gsutil cp yolo11_best.pt ${BUCKET}/models/yolo11_best.pt 2>/dev/null || true && \
gsutil -m cp -r runs/ ${BUCKET}/runs/ 2>/dev/null || true"

    log "YOLO job submitted. Monitor at:"
    echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT}"
    echo ""
    echo "  gcloud ai custom-jobs list --project=${PROJECT} --region=${REGION} --filter='displayName:yolo'"
}

# ---------------------------------------------------------------------------
# Download results from GCS
# ---------------------------------------------------------------------------
download_results() {
    log "Downloading results from GCS..."
    mkdir -p "${LOCAL_DIR}/models"

    gsutil cp "${BUCKET}/models/efficientnet_b3_weights.pt" "${LOCAL_DIR}/models/" 2>/dev/null || warn "No classifier weights"
    gsutil cp "${BUCKET}/models/product_embeddings.npy" "${LOCAL_DIR}/models/" 2>/dev/null || warn "No embeddings"
    gsutil cp "${BUCKET}/models/embedding_config.json" "${LOCAL_DIR}/models/" 2>/dev/null || warn "No config"
    gsutil cp "${BUCKET}/models/yolo11_best.onnx" "${LOCAL_DIR}/" 2>/dev/null || warn "No ONNX model"
    gsutil cp "${BUCKET}/models/yolo11_best.pt" "${LOCAL_DIR}/" 2>/dev/null || warn "No PT model"

    log "Download complete. Files:"
    ls -lh "${LOCAL_DIR}/models/efficientnet_b3_weights.pt" "${LOCAL_DIR}/models/product_embeddings.npy" 2>/dev/null || true
    ls -lh "${LOCAL_DIR}/yolo11_best"* 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
case "$MODE" in
    upload)
        upload_data
        ;;
    classifier)
        submit_classifier
        ;;
    yolo)
        submit_yolo
        ;;
    both)
        upload_data
        submit_classifier
        echo ""
        log "Waiting 5s before submitting YOLO job..."
        sleep 5
        submit_yolo
        echo ""
        log "Both jobs submitted. They run in parallel on separate VMs."
        log "Download results when done: bash train_vertex_jobs.sh download"
        ;;
    download)
        download_results
        ;;
    *)
        echo "Usage: bash train_vertex_jobs.sh {upload|classifier|yolo|both|download}"
        exit 1
        ;;
esac
