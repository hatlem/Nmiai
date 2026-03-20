#!/bin/bash
# check_training_json.sh — Check all GPU training VMs, output JSON for dashboard
# Usage: ./check_training_json.sh > ../training_status.json
# Or:    watch -n 300 ./check_training_json.sh > ../training_status.json

export PATH="$HOME/google-cloud-sdk/bin:$PATH"

VMS="yolo26-a100:us-central1-b:A100 yolo26-train:europe-west4-a:L4 nmiai-train-fast:us-central1-a:L4 yolo26-l4-3:europe-west4-c:L4 yolo26-t4-1:us-central1-a:T4"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
BEST_GLOBAL=0

echo "{"
echo "  \"timestamp\": \"$TIMESTAMP\","
echo "  \"vms\": ["

first_vm=true
for vm_info in $VMS; do
    vm=$(echo $vm_info | cut -d: -f1)
    zone=$(echo $vm_info | cut -d: -f2)
    gpu=$(echo $vm_info | cut -d: -f3)

    result=$(gcloud compute ssh $vm --zone=$zone --project=ainm26osl-710 --command='
        for f in /tmp/train/runs/detect/*/results.csv; do
            [ ! -f "$f" ] && continue
            epochs=$(wc -l < $f 2>/dev/null)
            [ "$epochs" -lt 2 ] 2>/dev/null && continue
            best=$(sort -t, -k7 -rn $f 2>/dev/null | head -1 | cut -d, -f7)
            last_epoch=$(tail -1 $f | cut -d, -f1)
            total_epochs=$(head -1 $f | grep -c "")
            name=$(basename $(dirname $f))
            echo "RUN|${name}|${last_epoch}|${best}"
        done 2>/dev/null
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1
    ' --ssh-key-expire-after=20s 2>&1)

    gpu_util=""
    gpu_mem_used=""
    gpu_mem_total=""
    runs="[]"

    if [ -n "$result" ]; then
        # Parse GPU info
        gpu_line=$(echo "$result" | grep -v "RUN|" | grep -E "^[0-9]" | head -1)
        if [ -n "$gpu_line" ]; then
            gpu_util=$(echo "$gpu_line" | cut -d, -f1 | tr -d ' ')
            gpu_mem_used=$(echo "$gpu_line" | cut -d, -f2 | tr -d ' ')
            gpu_mem_total=$(echo "$gpu_line" | cut -d, -f3 | tr -d ' ')
        fi

        # Parse training runs
        run_lines=$(echo "$result" | grep "^RUN|")
        if [ -n "$run_lines" ]; then
            runs="["
            first_run=true
            while IFS= read -r line; do
                name=$(echo "$line" | cut -d'|' -f2)
                epoch=$(echo "$line" | cut -d'|' -f3 | tr -d ' ')
                best=$(echo "$line" | cut -d'|' -f4 | tr -d ' ')

                [ "$first_run" = true ] || runs="$runs,"
                first_run=false
                runs="$runs{\"name\":\"$name\",\"epoch\":$epoch,\"best_map50\":$best}"

                # Track global best
                if [ "$(echo "$best > $BEST_GLOBAL" | bc -l 2>/dev/null)" = "1" ]; then
                    BEST_GLOBAL=$best
                fi
            done <<< "$run_lines"
            runs="$runs]"
        fi
    fi

    [ "$first_vm" = true ] || echo ","
    first_vm=false

    cat <<VMJSON
    {
      "name": "$vm",
      "zone": "$zone",
      "gpu": "$gpu",
      "gpu_util": ${gpu_util:-null},
      "gpu_mem_used": ${gpu_mem_used:-null},
      "gpu_mem_total": ${gpu_mem_total:-null},
      "online": $([ -n "$gpu_util" ] && echo "true" || echo "false"),
      "runs": $runs
    }
VMJSON

done

echo "  ],"
echo "  \"best_map50\": $BEST_GLOBAL"
echo "}"
