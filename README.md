# NM i AI 2026

> Norwegian AI Championship -- Hackathon, March 19--22, 2026
> **Team Human-Like** -- Andreas Hatlem

A 69-hour AI hackathon with a 1,000,000 NOK prize pool. Three independent tasks scored separately; final ranking is the average of normalized scores across all three.

## Results

| Task | Best Score | Rank | Method |
|---|---|---|---|
| **NorgesGruppen** | **0.9220** | **#6 / 341** | 3-model WBF ensemble (YOLOv8x + YOLO26-x + K-fold) |
| **Astar Island** | **91.3 pts** | **#22** | Bayesian KT estimator + ground-truth lookup (228 bins) |
| **Tripletex** | 9.8 | #99 | Hybrid template + LLM tool agent (Gemini 3.1 Pro) |

## Tasks

### 1. Tripletex -- AI Accounting Agent

An autonomous agent with an HTTPS `/solve` endpoint that receives natural-language accounting tasks (7 languages) and executes them against the Tripletex ERP API.

**Architecture:** Hybrid router with two execution paths:
- **Template engine** -- pattern-matched templates for the 30 known task types. Fast and deterministic.
- **LLM tool agent** -- Gemini 3.1 Pro via Vertex AI for complex multi-step tasks. Dynamically explores the API, builds payloads, and verifies results.

Pipeline: classify task -> route (template or LLM) -> execute API calls -> verify.

**Stack:** Python, FastAPI, Vertex AI (Gemini 3.1 Pro), Google Cloud Run

### 2. Astar Island -- Viking Civilization Prediction

Predict the full 40x40 grid state (6 terrain types) of a stochastic Norse civilization simulator from limited observations (15x15 viewport, 50 queries per round, 5 seeds).

**Architecture:**
- **Ground-truth lookup** built from 11 rounds of historical data (228 context bins) -- the primary signal
- **Krichevsky-Trofimov estimator** with adaptive alpha for cells without lookup data
- **Query optimizer** that concentrates viewports on high-uncertainty regions
- **Round-weighted ensemble** blending lookup, per-round transition profiles, and live observations

**Key insight:** Empirical ground-truth lookup outperformed simulation-based prediction by a wide margin.

**Stack:** Node.js, Google Cloud Run

### 3. NorgesGruppen -- Retail Object Detection

Detect and classify grocery products on store shelf images (356 product categories). Runs offline in a Docker sandbox with an L4 GPU.

**Architecture:**
- **3-model ensemble** -- YOLOv8x (fulldata), YOLO26-x (fulldata), YOLO26-x (K-fold 2). Different architectures for diversity.
- **Weighted Box Fusion (WBF)** to merge predictions across models
- **Test-time augmentation** (1280px + horizontal flip)
- **ONNX Runtime** inference with GPU acceleration
- Graceful fallbacks for single-model mode, missing classifiers, and time budget overruns

**Training:** 7 GCP VMs in parallel (A100, L4, T4) running K-fold cross-validation, pseudo-labeling, and progressive resizing.

**Stack:** Python, ONNX Runtime, ultralytics, PyTorch

## Key Findings

1. **Ensemble is everything.** Score went from 0.476 (single YOLO) to 0.922 (3-model ensemble) -- nearly doubled.
2. **Diversity beats strength in ensembles.** A weaker fold model (mAP 0.749) produced a better ensemble than stronger same-architecture models.
3. **Full-data training was the single largest improvement** (+0.004) for NorgesGruppen.
4. **Empirical lookup beat simulation** for Astar Island predictions.
5. **Post-processing tuning never helped** -- WBF param changes, multi-scale TTA, and Soft-NMS all hurt scores.

## Tech Stack

| Component | Technology |
|---|---|
| Cloud | Google Cloud Platform (Vertex AI, Cloud Run, Compute Engine) |
| AI Models | Gemini 3.1 Pro, YOLOv8x, YOLO26-x, YOLO11-x |
| Training | 7 GCP VMs (A100, L4, T4), K-fold, pseudo-labeling |
| Inference | ONNX Runtime GPU, WBF ensemble, TTA |
| Languages | Python, Node.js |
| AI Assistance | Claude Code (Opus 4.6) throughout the competition |

## Repository Structure

```
nmiai/
├── tripletex/           # Task 1: AI accounting agent
│   ├── main.py          #   FastAPI app + hybrid router
│   ├── templates.py     #   Template engine for known task types
│   ├── tool_agent.py    #   LLM-powered dynamic agent
│   ├── executor.py      #   API call executor
│   └── Dockerfile       #   Cloud Run deployment
├── astar-island/        # Task 2: Civilization prediction
│   ├── agent_v7.js      #   Main prediction agent
│   ├── gt_lookup.json   #   Historical ground-truth data
│   └── cloud/           #   Cloud Run deployment
├── norgesgruppen/       # Task 3: Object detection
│   ├── run.py           #   Sandbox entry point
│   ├── src/             #   Detector, classifier, WBF, SAHI modules
│   ├── *.onnx           #   Trained model weights
│   └── train*.py        #   Training scripts
└── docs/                # Competition documentation
```

## Deployment

### Tripletex (Cloud Run)

```bash
cd tripletex
gcloud run deploy tripletex-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300 \
  --min-instances 1
```

### Astar Island (Cloud Run)

```bash
cd astar-island/cloud
gcloud run deploy astar-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300
```

### NorgesGruppen (offline submission)

```bash
cd norgesgruppen
bash package_submission.sh
# Upload submission_YYYYMMDD_HHMMSS.zip to app.ainm.no
```

## License

[MIT](LICENSE)
