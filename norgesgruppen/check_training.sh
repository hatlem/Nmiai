#!/bin/bash
# check_training.sh — Quick status of all GPU training jobs
# Usage: ./check_training.sh

export PATH="$HOME/google-cloud-sdk/bin:$PATH"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  NorgesGruppen Training Status — $(date +'%Y-%m-%d %H:%M')          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

VMS="yolo26-a100:us-central1-b:A100 yolo26-train:europe-west4-a:L4 nmiai-train-fast:us-central1-a:L4 yolo26-l4-3:europe-west4-c:L4 yolo26-t4-1:us-central1-a:T4"

for vm_info in $VMS; do
    vm=$(echo $vm_info | cut -d: -f1)
    zone=$(echo $vm_info | cut -d: -f2)
    gpu=$(echo $vm_info | cut -d: -f3)

    result=$(gcloud compute ssh $vm --zone=$zone --project=ainm26osl-710 --command='
        for f in /tmp/train/runs/detect/*/results.csv; do
            [ ! -f "$f" ] && continue
            epochs=$(wc -l < $f 2>/dev/null)
            [ "$epochs" -lt 5 ] 2>/dev/null && continue
            best=$(sort -t, -k7 -rn $f 2>/dev/null | head -1 | cut -d, -f7)
            last_epoch=$(tail -1 $f | cut -d, -f1)
            name=$(basename $(dirname $f))
            echo "${name}|${last_epoch}|${best}"
        done 2>/dev/null
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d " "
    ' --ssh-key-expire-after=20s 2>&1 | grep -E "train|%")

    echo "  $vm ($gpu)"
    if [ -z "$result" ]; then
        echo "    (ingen data)"
    else
        echo "$result" | while read line; do
            if echo "$line" | grep -q "%"; then
                echo "    GPU: $line"
            else
                name=$(echo $line | cut -d'|' -f1)
                epoch=$(echo $line | cut -d'|' -f2)
                best=$(echo $line | cut -d'|' -f3)
                echo "    $name: epoch $epoch, best mAP50=$best"
            fi
        done
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════════════"
echo "  Best submission: 0.6740 (20 mar 10:43)"
echo "  Target: 0.92+"
echo "  Max 3 submissions/dag — sjekk app.ainm.no"
echo "════════════════════════════════════════════════════════════════"
