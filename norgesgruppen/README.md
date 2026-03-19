# Task 3 - NorgesGruppen Object Detection

> NM i AI 2026 - Butikkhylle/planogram deteksjon og klassifisering

## Oversikt

Offline object detection-løsning for gjenkjenning av varer på butikkhyller.
Bruker YOLO26 trent på Vertex AI, eksportert til ONNX for rask offline inferens.

## Arkitektur

- **Modell:** YOLO26 (NMS-fri sanntidsdeteksjon)
- **Trening:** Vertex AI Custom Training (H100/A100 GPU)
- **Inferens:** ONNX Runtime (CPU/GPU offline)
- **Pre-prosessering:** CLAHE i LAB-fargerom for gjenskinn-fjerning

## Kjøring

```bash
pip install -r requirements.txt
python src/inference.py --input <bilder-mappe> --output detections.json
```

## Lokal testing

```bash
python test_local.py --model models/yolo26.onnx --image test_images/sample.jpg
```

## Struktur

```
task3-norgesgruppen/
├── README.md
├── requirements.txt
├── train_vertex.py          # Trening på GCP Vertex AI
├── src/
│   ├── inference.py          # Hovedinnlevering (offline eval)
│   └── utils.py              # CLAHE pre-prosessering
├── test_local.py             # Lokal verifikasjon
└── models/                   # .onnx modeller (gitignored)
```

## Lisens

MIT License - NM i AI 2026
