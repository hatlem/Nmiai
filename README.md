# NM i AI 2026

> Norgesmesterskapet i kunstig intelligens — Hackathon 19.–22. mars 2026

## Oppgaver

### 1. Tripletex — AI Accounting Agent (`tripletex/`)

AI-agent med HTTPS `/solve`-endpoint som mottar regnskapsoppgaver på naturlig språk og utforer dem via Tripletex API.

- **Arkitektur:** LLM-drevet planner → executor → verifier pipeline
- **Modell:** Gemini 3.1 Pro via Vertex AI
- **Deploy:** Google Cloud Run

### 2. Astar Island — Viking Civilisation Prediction (`astar-island/`)

Predikerer tilstanden til en stokastisk norrøn sivilisasjonssimulator gjennom begrenset viewport-observasjon.

- **Arkitektur:** Bayesian inferens + Monte Carlo-simulator + swarm query optimization
- **Metode:** Krichevsky-Trofimov estimator, loopy belief propagation, ABC parameter-inferens
- **Deploy:** Google Cloud Run

### 3. NorgesGruppen — Object Detection (`norgesgruppen/`)

Detekterer og klassifiserer dagligvarer på butikkhyllebilder i offline Docker-sandbox.

- **Modell:** YOLO11 trent på Vertex AI, eksportert til ONNX
- **Klassifisering:** DINOv2 embedding-basert nearest-neighbor
- **Inferens:** ONNX Runtime med GPU (L4), SAHI for small object detection

## Tech Stack

| Komponent | Teknologi |
|---|---|
| Cloud | Google Cloud Platform (Vertex AI, Cloud Run, GCS) |
| AI-modeller | Gemini 3.1 Pro, YOLO11, DINOv2 |
| Språk | Python |
| Inferens | ONNX Runtime, ultralytics |

## Struktur

```
nmiai/
├── tripletex/          # Oppgave 1: AI regnskapsagent
├── astar-island/       # Oppgave 2: Sivilisasjonsprediksjon
├── norgesgruppen/      # Oppgave 3: Objektdeteksjon
├── docs/               # Konkurransedokumentasjon
└── CLAUDE.md           # AI-agent instruksjoner
```

## Team

**Human-Like** — Andreas Hatlem

## Lisens

[MIT](LICENSE)
