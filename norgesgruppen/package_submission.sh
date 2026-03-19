#!/bin/bash
# Package submission zip for NorgesGruppen task
#
# Supports two modes:
#   ./package_submission.sh              — single-stage (run.py + best.pt)
#   ./package_submission.sh --twostage   — two-stage (run_twostage.py + best.pt + embeddings)
set -e

MODE="single"
if [ "$1" = "--twostage" ]; then
    MODE="twostage"
fi

echo "=== Packaging ${MODE}-stage submission ==="

# Clean up
rm -rf submission_pkg
mkdir submission_pkg

# Both modes need src/utils.py for CLAHE preprocessing
mkdir -p submission_pkg/src
cp src/utils.py submission_pkg/src/
touch submission_pkg/src/__init__.py

if [ "$MODE" = "twostage" ]; then
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

    # Copy run script as run.py (sandbox expects run.py)
    cp run_twostage.py submission_pkg/run.py

    # Copy weights
    cp best.pt submission_pkg/
    cp models/product_embeddings.npy submission_pkg/
    cp models/embedding_config.json submission_pkg/

    for f in "${OPTIONAL_FILES[@]}"; do
        if [ -f "$f" ]; then
            cp "$f" submission_pkg/
        fi
    done

    # Count weight files (max 3 allowed)
    WEIGHT_COUNT=$(find submission_pkg -name "*.pt" -o -name "*.onnx" -o -name "*.safetensors" -o -name "*.npy" | wc -l | tr -d ' ')
    echo "Weight files: ${WEIGHT_COUNT}/3"
    if [ "$WEIGHT_COUNT" -gt 3 ]; then
        echo "ERROR: Max 3 weight files allowed!"
        exit 1
    fi
else
    # Single-stage: just detector
    if [ ! -f "best.pt" ]; then
        echo "ERROR: best.pt not found. Train model first."
        exit 1
    fi
    cp run.py submission_pkg/
    cp best.pt submission_pkg/
fi

# Check total weight size (max 420 MB)
TOTAL_SIZE=$(find submission_pkg \( -name "*.pt" -o -name "*.onnx" -o -name "*.safetensors" -o -name "*.npy" \) -exec du -cm {} + | tail -1 | awk '{print $1}')
echo "Total weight size: ${TOTAL_SIZE} MB / 420 MB"
if [ "$TOTAL_SIZE" -gt 420 ]; then
    echo "ERROR: Weights exceed 420 MB limit!"
    exit 1
fi

# Check Python file count (max 10)
PY_COUNT=$(find submission_pkg -name "*.py" | wc -l | tr -d ' ')
echo "Python files: ${PY_COUNT}/10"

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
