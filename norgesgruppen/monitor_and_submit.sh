#!/bin/bash
# Monitor GCP training and auto-download best checkpoints
# Run: bash monitor_and_submit.sh
# Checks every 5 min, downloads when mAP improves, packages submission

set -euo pipefail
cd "$(dirname "$0")"

PROJECT="ainm26osl-710"
TARGET_MAP=0.920
BEST_MAP=0
BEST_VM=""
CHECK_INTERVAL=300  # 5 min

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date +%H:%M:%S)]${NC} $*"; }
alert() { echo -e "${RED}[$(date +%H:%M:%S)] *** $* ***${NC}"; }

get_best_map() {
    local zone=$1 vm=$2 logfile=$3
    gcloud compute ssh "$vm" --project="$PROJECT" --zone="$zone" \
        --command="grep '^\s*all' /tmp/nmiai/$logfile 2>/dev/null | tail -1 | awk '{print \$6}'" 2>/dev/null || echo "0"
}

get_epoch() {
    local zone=$1 vm=$2 logfile=$3
    gcloud compute ssh "$vm" --project="$PROJECT" --zone="$zone" \
        --command="grep '^\s*all' /tmp/nmiai/$logfile 2>/dev/null | wc -l" 2>/dev/null || echo "0"
}

download_and_package() {
    local zone=$1 vm=$2 map=$3
    log "Downloading best.pt from $vm (mAP50=$map)..."

    # Find best.pt on VM
    local best_path
    best_path=$(gcloud compute ssh "$vm" --project="$PROJECT" --zone="$zone" \
        --command="ls -t /tmp/nmiai/runs/detect/train*/weights/best.pt 2>/dev/null | head -1" 2>/dev/null)

    if [ -z "$best_path" ]; then
        warn "No best.pt found on $vm"
        return 1
    fi

    # Upload to GCS first (faster than scp)
    gcloud compute ssh "$vm" --project="$PROJECT" --zone="$zone" \
        --command="gsutil cp $best_path gs://${PROJECT}-norgesgruppen/models/best_multiclass_latest.pt" 2>/dev/null

    # Download from GCS
    gsutil cp "gs://${PROJECT}-norgesgruppen/models/best_multiclass_latest.pt" best_multiclass.pt 2>/dev/null

    local size
    size=$(du -h best_multiclass.pt | awk '{print $1}')
    log "Downloaded: best_multiclass.pt ($size)"

    # Package submission with run_clean.py
    rm -rf submission_auto
    mkdir -p submission_auto/src
    cp run_clean.py submission_auto/run.py
    cp best_multiclass.pt submission_auto/best.pt
    cp src/wbf.py submission_auto/src/ 2>/dev/null || true
    cp src/__init__.py submission_auto/src/ 2>/dev/null || true

    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    rm -f "submission_auto_${ts}.zip"
    cd submission_auto
    zip -r "../submission_auto_${ts}.zip" . -x ".*" "__MACOSX/*" > /dev/null
    cd ..

    local zipsize
    zipsize=$(du -h "submission_auto_${ts}.zip" | awk '{print $1}')
    alert "SUBMISSION READY: submission_auto_${ts}.zip ($zipsize) — mAP50=$map"

    # Symlink for easy access
    ln -sf "submission_auto_${ts}.zip" submission_latest.zip

    return 0
}

log "=== NorgesGruppen Training Monitor ==="
log "Target: mAP50 >= $TARGET_MAP"
log "Checking every ${CHECK_INTERVAL}s..."
echo ""

while true; do
    echo -e "${CYAN}── $(date '+%Y-%m-%d %H:%M:%S') ──${NC}"

    # Check VM #2 (T4, best so far)
    map2=$(get_best_map "europe-west1-c" "nmiai-train-yolo" "yolo_train.log")
    ep2=$(get_epoch "europe-west1-c" "nmiai-train-yolo" "yolo_train.log")
    echo -e "  VM #2 (T4): Epoch ${ep2}, mAP50=${CYAN}${map2}${NC}"

    # Check VM #3 (L4)
    map3=$(get_best_map "us-central1-a" "nmiai-train-fast" "train3.log")
    ep3=$(get_epoch "us-central1-a" "nmiai-train-fast" "train3.log")
    echo -e "  VM #3 (L4): Epoch ${ep3}, mAP50=${CYAN}${map3}${NC}"

    # Check VM #1 (EfficientNet)
    eff_acc=$(gcloud compute ssh nmiai-train-gpu --project="$PROJECT" --zone=europe-west1-b \
        --command="grep 'val_acc' /tmp/nmiai/classifier_train.log 2>/dev/null | tail -1 | grep -oP 'val_acc=\K[0-9.]+'" 2>/dev/null || echo "?")
    echo -e "  VM #1 (T4): EfficientNet val_acc=${CYAN}${eff_acc}${NC}"

    # Find best across VMs
    current_best=0
    current_vm=""
    current_zone=""

    if (( $(echo "$map2 > $current_best" | bc -l 2>/dev/null || echo 0) )); then
        current_best=$map2; current_vm="nmiai-train-yolo"; current_zone="europe-west1-c"
    fi
    if (( $(echo "$map3 > $current_best" | bc -l 2>/dev/null || echo 0) )); then
        current_best=$map3; current_vm="nmiai-train-fast"; current_zone="us-central1-a"
    fi

    # Download if improved
    improved=$(python3 -c "print(1 if float('${current_best}') > float('${BEST_MAP}') + 0.005 else 0)" 2>/dev/null || echo 0)
    if [ "$improved" = "1" ]; then
        alert "NEW BEST: mAP50=$current_best (was $BEST_MAP) from $current_vm"
        BEST_MAP=$current_best
        BEST_VM=$current_vm
        download_and_package "$current_zone" "$current_vm" "$current_best"
    fi

    # Check if target reached
    target_reached=$(python3 -c "print(1 if float('${current_best}') >= ${TARGET_MAP} else 0)" 2>/dev/null || echo 0)
    if [ "$target_reached" = "1" ]; then
        alert "TARGET REACHED: mAP50=$current_best >= $TARGET_MAP"
        alert "Upload submission_latest.zip NOW!"
        # macOS notification
        osascript -e 'display notification "mAP50 target reached!" with title "NM i AI" sound name "Glass"' 2>/dev/null || true
    fi

    # Progress
    pct=$(python3 -c "print(f'{float(\"${current_best}\") / ${TARGET_MAP} * 100:.1f}')" 2>/dev/null || echo "?")
    echo -e "  ${GREEN}Best: ${current_best} (${pct}% of target)${NC}"
    echo ""

    sleep $CHECK_INTERVAL
done
