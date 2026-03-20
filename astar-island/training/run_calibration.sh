#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Run massive simulator parameter calibration on GCP Compute Engine.
#
# This script:
# 1. Creates a GCP VM with many CPUs (or uses existing)
# 2. Uploads code + cache data
# 3. Runs the calibration
# 4. Downloads results
#
# Usage:
#   ./run_calibration.sh [create|run|download|cleanup|all]
#
# Prerequisites:
#   - gcloud CLI authenticated with project ainm26osl-710
#   - Cache files in ../cache/ (r*_init.json, r*_gt_s*.json)
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT="ainm26osl-710"
ZONE="europe-north1-b"
VM_NAME="calibration-vm"
MACHINE_TYPE="c2-standard-30"  # 30 vCPUs, 120GB RAM — fast for CPU-bound work
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ASTAR_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="$ASTAR_DIR/cache"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[calibration]${NC} $*"; }
warn() { echo -e "${YELLOW}[calibration]${NC} $*"; }
err() { echo -e "${RED}[calibration]${NC} $*" >&2; }

# ── Create VM ────────────────────────────────────────────────────────────────

create_vm() {
    log "Creating VM: $VM_NAME ($MACHINE_TYPE) in $ZONE..."

    gcloud compute instances create "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --image-family=debian-12 \
        --image-project=debian-cloud \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-ssd \
        --scopes=cloud-platform \
        --metadata=startup-script='#!/bin/bash
apt-get update -qq
apt-get install -y -qq python3 python3-pip python3-venv > /dev/null 2>&1
python3 -m venv /opt/calibration-env
/opt/calibration-env/bin/pip install numpy > /dev/null 2>&1
echo "VM ready" > /tmp/vm-ready
'

    log "Waiting for VM to be ready..."
    for i in $(seq 1 60); do
        if gcloud compute ssh "$VM_NAME" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --command="test -f /tmp/vm-ready && echo ready" 2>/dev/null | grep -q ready; then
            log "VM is ready!"
            return 0
        fi
        sleep 5
        echo -n "."
    done
    err "VM did not become ready in time"
    return 1
}

# ── Upload code and data ────────────────────────────────────────────────────

upload_data() {
    log "Uploading simulator code and cache data..."

    # Create remote directory structure
    gcloud compute ssh "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --command="mkdir -p /home/\$USER/astar-island/cache /home/\$USER/astar-island/training"

    # Upload simulator
    gcloud compute scp "$ASTAR_DIR/simulator.py" \
        "$VM_NAME:/home/\$USER/astar-island/" \
        --project="$PROJECT" \
        --zone="$ZONE"

    # Upload calibration script
    gcloud compute scp "$SCRIPT_DIR/calibrate_sim.py" \
        "$VM_NAME:/home/\$USER/astar-island/training/" \
        --project="$PROJECT" \
        --zone="$ZONE"

    # Upload cache files (init + GT)
    log "Uploading cache files..."
    for f in "$CACHE_DIR"/r*_init.json "$CACHE_DIR"/r*_gt_s*.json; do
        if [ -f "$f" ]; then
            gcloud compute scp "$f" \
                "$VM_NAME:/home/\$USER/astar-island/cache/" \
                --project="$PROJECT" \
                --zone="$ZONE"
        fi
    done

    log "Upload complete."
}

# ── Run calibration ─────────────────────────────────────────────────────────

run_calibration() {
    log "Starting calibration on $VM_NAME..."
    log "This will take 2-4 hours with 30 vCPUs..."

    # Get number of CPUs on VM
    NCPUS=$(gcloud compute ssh "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --command="nproc" 2>/dev/null)
    log "VM has $NCPUS CPUs"

    # Run calibration in tmux so it survives SSH disconnects
    gcloud compute ssh "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --command="
        sudo apt-get install -y -qq tmux > /dev/null 2>&1

        # Kill any existing calibration session
        tmux kill-session -t calibration 2>/dev/null || true

        # Start new tmux session
        tmux new-session -d -s calibration '
            cd /home/\$USER/astar-island/training
            /opt/calibration-env/bin/python calibrate_sim.py \
                --workers $NCPUS \
                --mc-runs 50 \
                --rounds 1,2,4,5,6,7,8 \
                --output calibration_results.json \
                --batch-size 1000 \
                2>&1 | tee calibration.log
            echo \"CALIBRATION COMPLETE\" >> calibration.log
        '
        echo 'Calibration started in tmux session. Use: tmux attach -t calibration'
    "

    log "Calibration running in tmux on the VM."
    log "Monitor with:"
    log "  gcloud compute ssh $VM_NAME --project=$PROJECT --zone=$ZONE --command='tail -f /home/\$USER/astar-island/training/calibration.log'"
    log "Or attach to tmux:"
    log "  gcloud compute ssh $VM_NAME --project=$PROJECT --zone=$ZONE -- -t 'tmux attach -t calibration'"
}

# ── Check status ────────────────────────────────────────────────────────────

check_status() {
    log "Checking calibration status..."
    gcloud compute ssh "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --command="tail -30 /home/\$USER/astar-island/training/calibration.log 2>/dev/null || echo 'No log file yet'"
}

# ── Download results ────────────────────────────────────────────────────────

download_results() {
    log "Downloading results..."

    gcloud compute scp \
        "$VM_NAME:/home/\$USER/astar-island/training/calibration_results.json" \
        "$SCRIPT_DIR/calibration_results.json" \
        --project="$PROJECT" \
        --zone="$ZONE" 2>/dev/null && log "Downloaded calibration_results.json" || warn "calibration_results.json not found"

    gcloud compute scp \
        "$VM_NAME:/home/\$USER/astar-island/cache/calibrated_params_v2.json" \
        "$CACHE_DIR/calibrated_params_v2.json" \
        --project="$PROJECT" \
        --zone="$ZONE" 2>/dev/null && log "Downloaded calibrated_params_v2.json" || warn "calibrated_params_v2.json not found"

    gcloud compute scp \
        "$VM_NAME:/home/\$USER/astar-island/training/calibration.log" \
        "$SCRIPT_DIR/calibration.log" \
        --project="$PROJECT" \
        --zone="$ZONE" 2>/dev/null && log "Downloaded calibration.log" || warn "calibration.log not found"

    log "Download complete."
}

# ── Cleanup ─────────────────────────────────────────────────────────────────

cleanup() {
    warn "Deleting VM $VM_NAME..."
    gcloud compute instances delete "$VM_NAME" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --quiet
    log "VM deleted."
}

# ── Main ────────────────────────────────────────────────────────────────────

case "${1:-all}" in
    create)
        create_vm
        ;;
    upload)
        upload_data
        ;;
    run)
        run_calibration
        ;;
    status)
        check_status
        ;;
    download)
        download_results
        ;;
    cleanup)
        cleanup
        ;;
    all)
        create_vm
        upload_data
        run_calibration
        log ""
        log "VM is running calibration. Next steps:"
        log "  1. Check status:   $0 status"
        log "  2. Download:       $0 download"
        log "  3. Cleanup:        $0 cleanup"
        ;;
    *)
        echo "Usage: $0 [create|upload|run|status|download|cleanup|all]"
        exit 1
        ;;
esac
