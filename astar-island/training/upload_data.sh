#!/bin/bash
# Upload GT cache to GCS for training on Compute Engine / Vertex AI
set -e

PROJECT="ainm26osl-710"
BUCKET="gs://${PROJECT}-astar"
CACHE_DIR="$(dirname "$0")/../cache"

echo "Creating bucket if not exists..."
gsutil mb -p "$PROJECT" -l europe-north1 "$BUCKET" 2>/dev/null || true

echo "Uploading GT data..."
gsutil -m cp "${CACHE_DIR}"/r*_gt_s*.json "${BUCKET}/cache/"
gsutil -m cp "${CACHE_DIR}"/r*_init.json "${BUCKET}/cache/"

echo "Listing uploaded files..."
gsutil ls "${BUCKET}/cache/"

echo "Done. Files available at ${BUCKET}/cache/"
