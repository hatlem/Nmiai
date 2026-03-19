#!/bin/bash
# Package submission zip for NorgesGruppen task
#
# Modes:
#   ./package_submission.sh              — single-stage (run.py + best.pt + WBF)
#   ./package_submission.sh --twostage   — two-stage (run_twostage.py + classifier)
#   ./package_submission.sh --best       — full pipeline (run_best.py + all modules)
set -e

MODE="single"
if [ "$1" = "--twostage" ]; then
    MODE="twostage"
elif [ "$1" = "--best" ]; then
    MODE="best"
fi

echo "=== Packaging ${MODE} submission ==="

# Clean up
rm -rf submission_pkg
mkdir submission_pkg

# All modes need src/ modules
mkdir -p submission_pkg/src
cp src/__init__.py submission_pkg/src/
cp src/utils.py submission_pkg/src/    # CLAHE preprocessing
cp src/wbf.py submission_pkg/src/      # Weighted Boxes Fusion

if [ "$MODE" = "best" ]; then
    # Full pipeline: SAHI + Soft-NMS + Ensemble + Classifier
    cp src/sahi.py submission_pkg/src/
    cp src/soft_nms.py submission_pkg/src/
    cp src/ensemble.py submission_pkg/src/

    if [ ! -f "best.pt" ]; then
        echo "ERROR: best.pt not found. Train model first."
        exit 1
    fi

    # Copy run script as run.py (sandbox expects run.py)
    cp run_best.py submission_pkg/run.py
    cp best.pt submission_pkg/

    # Optional: classifier files
    for f in "models/product_embeddings.npy" "models/embedding_config.json" "models/efficientnet_b3_weights.pt"; do
        if [ -f "$f" ]; then
            cp "$f" submission_pkg/
            echo "  Included: $f"
        fi
    done

    # Optional: secondary model for ensemble
    if [ -f "rtdetr_best.pt" ]; then
        cp rtdetr_best.pt submission_pkg/
        echo "  Included: rtdetr_best.pt (ensemble)"
    fi

elif [ "$MODE" = "twostage" ]; then
    # Two-stage: detector + classifier + embeddings
    REQUIRED_FILES=(
        "best.pt"
        "models/product_embeddings.npy"
        "models/embedding_config.json"
    )
    OPTIONAL_FILES=(
        "models/efficientnet_b3_weights.pt"
    )

    for f in "${REQUIRED_FILES[@]}"; do
        if [ ! -f "$f" ]; then
            echo "ERROR: Required file not found: $f"
            exit 1
        fi
    done

    cp run_twostage.py submission_pkg/run.py
    cp best.pt submission_pkg/
    cp models/product_embeddings.npy submission_pkg/
    cp models/embedding_config.json submission_pkg/

    for f in "${OPTIONAL_FILES[@]}"; do
        if [ -f "$f" ]; then
            cp "$f" submission_pkg/
        fi
    done

else
    # Single-stage: detector + WBF
    if [ ! -f "best.pt" ]; then
        echo "ERROR: best.pt not found. Train model first."
        exit 1
    fi
    cp run.py submission_pkg/
    cp best.pt submission_pkg/
fi

# ── Validation ────────────────────────────────────────────────────────

# Check weight file count (max 3 allowed)
WEIGHT_COUNT=$(find submission_pkg -name "*.pt" -o -name "*.onnx" -o -name "*.safetensors" -o -name "*.npy" | wc -l | tr -d ' ')
echo "Weight files: ${WEIGHT_COUNT}/3"
if [ "$WEIGHT_COUNT" -gt 3 ]; then
    echo "ERROR: Max 3 weight files allowed!"
    exit 1
fi

# Check total weight size (max 420 MB)
TOTAL_SIZE=$(find submission_pkg \( -name "*.pt" -o -name "*.onnx" -o -name "*.safetensors" -o -name "*.npy" \) -exec du -cm {} + | tail -1 | awk '{print $1}')
echo "Total weight size: ${TOTAL_SIZE} MB / 420 MB"
if [ "$TOTAL_SIZE" -gt 420 ]; then
    echo "ERROR: Weights exceed 420 MB limit!"
    exit 1
fi

# Check Python file count
PY_COUNT=$(find submission_pkg -name "*.py" | wc -l | tr -d ' ')
echo "Python files: ${PY_COUNT}"

# Check no import os violation
if grep -rn "^import os$\|^from os import" submission_pkg/ --include="*.py" 2>/dev/null; then
    echo "ERROR: Found 'import os' — sandbox violation!"
    exit 1
fi
echo "Sandbox compliance: OK (no import os)"

# Create zip
cd submission_pkg
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
cd ..

# Verify
echo ""
echo "=== Zip contents ==="
unzip -l submission.zip

echo ""
ZIPSIZE=$(du -m submission.zip | awk '{print $1}')
echo "Zip size: ${ZIPSIZE} MB"
echo "Ready to upload: submission.zip"

# ── Report to dashboard ──
echo ""
echo "=== Reporting to dashboard ==="
curl -s -X POST http://localhost:8090/api/score \
  -H 'Content-Type: application/json' \
  -d "{\"task\":\"norgesgruppen\",\"raw\":0,\"note\":\"ZIP packaged (${MODE}, ${ZIPSIZE}MB)\"}" \
  2>/dev/null || echo "Dashboard not running, skipping report"
