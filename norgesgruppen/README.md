# NorgesGruppen -- Object Detection

> NM i AI 2026 -- Task 3: Grocery shelf product detection and classification
> **Best score: 0.9220 mAP@0.5 -- #6 of 341 teams**

## Task

Detect and classify grocery products on store shelf images. 248 training images with ~22,700 bounding box annotations across 356 product categories. Submissions run in a sandboxed Docker container (NVIDIA L4 GPU, no network, 300s timeout, 420 MB weight limit).

Scoring: `0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5`

## Approach

3-model WBF (Weighted Boxes Fusion) ensemble with horizontal flip TTA.

### Models in Best Submission (#16)

| Model | Architecture | Training Data | mAP50 (val) | Role |
|---|---|---|---|---|
| `fulldata_v8x_s42.onnx` | YOLOv8x | 248 images (all data) | ~0.80 | Strong baseline |
| `fulldata_yolo26_best.onnx` | YOLO26-x | 248 images (all data) | ~0.79 | Architecture diversity |
| `fold2_best.onnx` | YOLO26-x | 199 images (K-fold 2) | 0.749 | Data diversity |

All models are multi-class (356 categories), exported to ONNX for sandbox compatibility.

### Key Innovations

- **Architecture diversity**: YOLOv8x and YOLO26-x make different errors -- WBF combines their strengths
- **Data diversity**: The fold2 model (trained on 80% of data) provides independent predictions from fulldata models
- **Fulldata training**: Training on all 248 images instead of train-only splits was the single biggest improvement (+0.004)
- **WBF over NMS**: Weighted Boxes Fusion merges overlapping boxes instead of discarding them
- **Very low confidence** (0.001): Maximizes recall, which the scoring rewards heavily
- **Paradox**: The weakest model (fold2, mAP 0.749) produced the best ensemble -- stronger models gave worse results

### Score Progression

| Submission | Score | Method | Takeaway |
|---|---|---|---|
| #1 | 0.476 | YOLOv8x single model | Baseline |
| #4 | 0.674 | YOLO26-x + DINOv2 two-stage | Two-stage better than single |
| #8 | 0.914 | 3-model WBF ensemble | **Ensemble is everything** |
| #10 | 0.916 | + conf 0.001 + resolution diversity | Marginal gain |
| #15 | 0.921 | + fulldata training (248 images) | **Biggest single improvement** |
| **#16** | **0.922** | **2x fulldata + fold2** | **Final best** |

### What Did NOT Work

- RT-DETR (zero mAP after 45+ epochs)
- Two-stage detection + DINOv2 classifier (slower and worse than end-to-end)
- Higher confidence thresholds (0.05 was worse than 0.001)
- Same-architecture ensembles (fold swapping gave no improvement)
- Model soup / weight averaging (catastrophic: 0.536)
- Multi-scale TTA beyond 1280+flip
- WBF hyperparameter tuning (0.893 -- much worse)
- Longer training (500 epochs overtrained vs 167)
- Synthetic data, oversampling, label smoothing, cosine LR

## Inference Pipeline

```bash
python run.py --input /path/to/images --output /path/to/predictions.json
```

`run.py` does:
1. Loads 3 ONNX models
2. Runs each model on every image at 1280px
3. TTA: horizontal flip doubles the prediction sets
4. WBF fuses all 6 sets of boxes (3 models x 2 TTA passes)
5. Outputs COCO-format JSON with `image_id`, `category_id`, `bbox`, `score`

### Sandbox Constraints

- Max 420 MB weights (ours: ~334 MB)
- Max 3 weight files
- Max 300s runtime (ours: ~38s)
- No `import os` -- uses `pathlib` only
- Pre-installed: `ultralytics==8.1.0`, `torch==2.6.0`, `onnxruntime-gpu==1.20.0`

## Training

Models were trained on GCP Vertex AI across 7 parallel GPU VMs (A100, L4, T4).

```bash
# K-fold training
python train_best.py --fold 2 --epochs 200 --imgsz 1280

# Fulldata training
python train_best.py --epochs 200 --imgsz 1280

# Export to ONNX
yolo export model=best.pt format=onnx opset=17 imgsz=1280 half=True
```

## Project Structure

```
norgesgruppen/
  run.py                  # Main entry point (full pipeline with fallbacks)
  run_ensemble.py         # Ensemble runner (best submission config)
  package_submission.sh   # Package zip for upload
  src/
    onnx_detector.py      # ONNX Runtime inference wrapper
    wbf.py                # Weighted Boxes Fusion (pure numpy)
    ensemble.py           # Multi-model ensemble logic
    soft_nms.py           # Soft-NMS post-processing
    sahi.py               # Sliced inference for small objects
    utils.py              # CLAHE image enhancement
    classifier.py         # DINOv2 classifier (deprecated)
    tta.py                # Test-time augmentation
```

### Packaging a Submission

```bash
./package_submission.sh --ensemble
```

Creates a timestamped zip with run script, ONNX models, and source modules.

## What We Learned (69 hours, 20 submissions)

### What worked
1. **Train on ALL data** — holding back 15% for validation cost us 0.004 mAP
2. **WBF ensemble of 3 models** — gave +0.24 mAP over single model
3. **Architecture diversity** — YOLOv8x + YOLO26-x in ensemble
4. **The "weakest" model gave best ensemble** — fold2 (0.749) beat pseudo (0.789), YOLO11-x (0.784), and 1600px (0.771) as 3rd model
5. **conf_threshold=0.001** — lower = better recall = better mAP
6. **Official `ensemble_boxes` package** — marginally better than custom WBF

### What did NOT work
- Post-processing tuning (WBF params, Soft-NMS, temperature scaling) — ALWAYS hurt
- Multi-scale TTA (640+960+1280) — worse than just 1280+flip
- Model soup (weight averaging) — catastrophically bad (0.536 mAP)
- Stronger individual models in ensemble — consistently gave worse results
- RT-DETR — mAP=0 after 45+ epochs
- DINOv2 two-stage classifier — two-stage always worse than end-to-end
- Synthetic data training — didn't improve beyond original data
- Longer training (500 ep) — overtrained, 167 ep was optimal

### Compute used
- 9 GCP GPU VMs in parallel (1x A100, 4x L4, 4x T4)
- ~100+ training runs over 3 days
- All on competition GCP account (`ainm26osl-710`)

## License

MIT
