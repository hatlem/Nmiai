#!/bin/bash
# Package the FINAL optimized submission for NorgesGruppen task
#
# Includes:
#   - run_final.py (SAHI + WBF + Soft-NMS + hybrid classification)
#   - best.pt (multi-class YOLOv8x FP16, 130 MB)
#   - models/efficientnet_b3_weights.pt (41 MB)
#   - models/product_embeddings.npy (2.1 MB)
#   - src/ modules (utils, wbf, soft_nms)
#
# Total: ~173 MB / 420 MB limit, 3/3 weight files, ~7/10 Python files
set -e

echo "=== Packaging FINAL optimized submission ==="

# Check required files
for f in "best.pt" "models/efficientnet_b3_weights.pt" "models/product_embeddings.npy" "models/embedding_config.json"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

# Clean up
rm -rf submission_pkg
mkdir -p submission_pkg/src submission_pkg/models

# ── Copy run script ──────────────────────────────────────────────────
cp run_final.py submission_pkg/run.py
echo "  run.py (from run_final.py)"

# ── Copy source modules ─────────────────────────────────────────────
cp src/__init__.py submission_pkg/src/
cp src/utils.py submission_pkg/src/       # CLAHE preprocessing
cp src/wbf.py submission_pkg/src/         # Weighted Boxes Fusion (fallback)
cp src/soft_nms.py submission_pkg/src/    # Soft-NMS for dense shelves
echo "  src/ modules: __init__, utils, wbf, soft_nms"

# ── Copy model weights ──────────────────────────────────────────────
cp best.pt submission_pkg/
echo "  best.pt (multi-class YOLOv8x FP16)"

cp models/efficientnet_b3_weights.pt submission_pkg/models/
cp models/product_embeddings.npy submission_pkg/models/
cp models/embedding_config.json submission_pkg/models/
echo "  models/ (EfficientNet-B3 + product embeddings)"

# ── Validation ───────────────────────────────────────────────────────

# Weight file count (max 3)
WEIGHT_COUNT=$(find submission_pkg \( -name "*.pt" -o -name "*.onnx" -o -name "*.safetensors" -o -name "*.npy" \) | wc -l | tr -d ' ')
echo ""
echo "Weight files: ${WEIGHT_COUNT}/3"
if [ "$WEIGHT_COUNT" -gt 3 ]; then
    echo "ERROR: Max 3 weight files allowed!"
    exit 1
fi

# Total weight size (max 420 MB)
TOTAL_SIZE=$(find submission_pkg \( -name "*.pt" -o -name "*.onnx" -o -name "*.safetensors" -o -name "*.npy" \) -exec du -cm {} + | tail -1 | awk '{print $1}')
echo "Total weight size: ${TOTAL_SIZE} MB / 420 MB"
if [ "$TOTAL_SIZE" -gt 420 ]; then
    echo "ERROR: Weights exceed 420 MB limit!"
    exit 1
fi

# Python file count (max 10)
PY_COUNT=$(find submission_pkg -name "*.py" | wc -l | tr -d ' ')
echo "Python files: ${PY_COUNT}/10"
if [ "$PY_COUNT" -gt 10 ]; then
    echo "ERROR: Max 10 Python files allowed!"
    exit 1
fi

# Sandbox compliance
if grep -rn "^import os$\|^from os import" submission_pkg/ --include="*.py" 2>/dev/null; then
    echo "ERROR: Found 'import os' — sandbox violation!"
    exit 1
fi
echo "Sandbox compliance: OK"

# List all files
echo ""
echo "=== Package contents ==="
find submission_pkg -type f | sort | while read f; do
    SIZE=$(du -h "$f" | awk '{print $1}')
    echo "  ${SIZE}  ${f#submission_pkg/}"
done

# Create zip
cd submission_pkg
zip -r ../submission.zip . -x ".*" "__MACOSX/*"
cd ..

echo ""
ZIPSIZE=$(du -m submission.zip | awk '{print $1}')
echo "=== submission.zip: ${ZIPSIZE} MB ==="
echo ""
echo "Verify zip structure:"
unzip -l submission.zip | head -20
echo ""
echo "Ready to upload: submission.zip"
