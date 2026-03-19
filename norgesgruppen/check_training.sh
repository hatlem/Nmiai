#!/bin/bash
# Check training progress on GCP VM
export PATH="$HOME/google-cloud-sdk/bin:$PATH"
gcloud compute ssh yolo-train --zone=europe-west1-b --command="
  echo '=== GPU ==='
  nvidia-smi | grep -E 'MiB|%' | head -2
  echo ''
  echo '=== Training progress ==='
  tail -5 /tmp/train/training.log
  echo ''
  echo '=== Check if best.pt exists ==='
  ls -la /tmp/train/best.pt 2>/dev/null || echo 'Not yet'
  ls -la /tmp/train/runs/detect/*/weights/best.pt 2>/dev/null || echo 'No weights yet'
" 2>&1
