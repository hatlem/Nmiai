# NM i AI 2026 — Norgesmesterskapet i kunstig intelligens

## Konkurranse
Vi deltar i NM i AI 2026 (hackathon-format). **1 000 000 NOK i premiepott.**
- **Start:** 19. mars kl. 18:00 CET
- **Frist:** søndag 22. mars kl. 15:00 CET
- **Totalt:** 69 timer, hvorav ~2 timer er brukt (startet 19. mars ~18:30)
- Samlet score = **gjennomsnitt av normaliserte scores på tvers av alle 3 oppgaver (33% hver)**
- Koden må være **åpen kildekode (MIT-lisens)** på offentlig GitHub-repo før fristen
- Alle lagmedlemmer må Vipps-verifiseres for premie
- Max 4 medlemmer per lag

## Google Cloud (ubegrenset)
Vi har en dedikert GCP-konto uten kreditgrenser — bruk det vi trenger.

| | |
|---|---|
| Prosjekt | `ainm26osl-710` |
| E-post | devstar7101@gcplab.me |
| Passord | ucyx4KaQu5 |
| Region | `europe-north1` (Finland, lavest latency til validatorer) |

### Tilgjengelige tjenester
- **Cloud Run** — deploy containeriserte API-er (Tripletex + Astar Island endpoints)
- **Vertex AI** — Gemini-modeller, Claude-modeller via Model Garden, ML-trening med GPU
- **Compute Engine** — full VM med GPU for tung trening
- **Cloud Storage** — datasett, modellvekter, logger
- **Cloud Shell** — gratis terminal med Python, Docker, gcloud CLI ferdig installert
- **Gemini Code Assist** — AI-kodehjelp i Cloud Shell Editor
- **AI Studio** (aistudio.google.com) — eksperimenter med Gemini direkte

### Deploy til Cloud Run (Tripletex + Astar Island)
```bash
gcloud run deploy my-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300 \
  --min-instances 1
```
Gir en URL som `https://my-agent-xxxxx-lz.a.run.app` — submit denne på app.ainm.no.

### Tilgjengelige AI-modeller (mars 2026)

**Via Vertex AI (vår GCP-konto):**

| Modell | Styrke | Bruk til |
|---|---|---|
| Gemini 3.1 Pro | Beste reasoning, 1M context | Komplekse Tripletex-oppgaver, Astar Island-analyse |
| Gemini 3.1 Flash-Lite | Raskest, billigst ($0.25/M input) | Enkel prompt-parsing, høyt volum |
| Gemini 3 Flash | Agentic, preview | Agentic workflows |
| Claude Opus 4.6 | Dypest reasoning, 1M context | Vanskelige multi-step oppgaver |
| Claude Sonnet 4.6 | 98% av Opus, 5x billigere | God default for det meste |

**Andre tilgjengelige (via API):**
- GPT-5.4 (1M context, Tool Search arkitektur)
- Qwen 3.5 Small (open-source, Apache 2.0, multimodal)
- NVIDIA Nemotron 3 Super (beste open-weight på SWE-Bench)

**For NorgesGruppen object detection:**
- YOLO26 (nyeste, jan 2026) — NMS-fri, 43% raskere CPU-inferens. MÅ eksporteres til ONNX for sandbox.
- YOLO11 — 22% færre parametere enn YOLOv8, bedre mAP. MÅ eksporteres til ONNX for sandbox.
- YOLOv8 — eneste som kjører native i sandbox (`ultralytics==8.1.0`). Enklest å starte med.
- **Anbefaling:** Tren med nyeste YOLO26/11, eksporter til ONNX (opset ≤ 20) for submission.

### Vertex AI fra Cloud Run

**VIKTIG: Gemini 3.1-modeller krever `location="global"`, IKKE `europe-north1`!**
- `gemini-3.1-pro-preview` og `gemini-3.1-flash-lite-preview` finnes KUN via `locations/global`
- `gemini-2.5-pro` og `gemini-2.5-flash` fungerer i `europe-north1`
- Cloud Run-tjenesten kan fortsatt ligge i `europe-north1` — bare Vertex AI init endres

```python
import vertexai
from vertexai.generative_models import GenerativeModel

# Gemini 3.1 (beste) — MÅ bruke location="global"
vertexai.init(project="ainm26osl-710", location="global")
model = GenerativeModel("gemini-3.1-pro-preview")  # eller "gemini-3.1-flash-lite-preview"
response = model.generate_content("Parse this accounting task: ...")

# Gemini 2.5 (fallback) — fungerer i europe-north1
# vertexai.init(project="ainm26osl-710", location="europe-north1")
# model = GenerativeModel("gemini-2.5-pro")  # eller "gemini-2.5-flash"
```

## Tre oppgaver

### 1. Tripletex — AI Accounting Agent (`tripletex/`)
Bygg en AI-agent med HTTPS `/solve`-endpoint som mottar regnskapsoppgaver på naturlig språk (7 språk) og utfører dem via Tripletex API. Scoring: korrekthet (felt-for-felt) × tier-multiplier + effektivitetsbonus. Max 5 min per oppgave. 30 oppgavetyper, 56 varianter hver.

### 2. Astar Island — Viking Civilisation Prediction (`astar-island/`)
Observer en stokastisk norrøn sivilisasjonssimulator gjennom begrenset viewport (15×15 av 40×40 kart). 50 queries per runde, 5 seeds. Prediker sannsynlighetsfordeling for terrengtyper (W×H×6 tensor). Scoring: entropy-vektet KL-divergens.

### 3. NorgesGruppen — Object Detection (`norgesgruppen/`)
Detekter og klassifiser dagligvarer på butikkhyllebilder. Upload `.zip` med `run.py` som kjører offline i Docker-sandbox (L4 GPU, 8GB RAM, 300s timeout, Python 3.11). Scoring: 70% detection mAP + 30% classification mAP. 356 produktkategorier. Bruk `ultralytics==8.1.0` eller ONNX.

## Dokumentasjon

### Generelt
- `docs/competition-overview.md` — Getting started, oppgaveoversikt
- `docs/competition-rules.md` — Fullstendig regelverk (18 seksjoner)
- `docs/google-cloud.md` — GCP-oppsett, Cloud Run deploy, tjenester, Gemini-verktøy

### Tripletex (`tripletex/`)
- `tripletex/docs/task1-tripletex.md` — Oppgavebeskrivelse, task categories, quick start
- `tripletex/docs/task1-endpoint.md` — /solve endpoint spec, request/response format, API-referanse
- `tripletex/docs/task1-scoring.md` — Scoring, tiers, effektivitetsbonus, rate limits
- `tripletex/docs/task1-examples.md` — Kodeeksempler, task patterns, feilhåndtering, optimalisering
- `tripletex/docs/task1-sandbox.md` — Sandbox-konto, API-autentisering, testing
- `tripletex/docs/tripletex-api-v2-oversikt.md` — Komplett Tripletex API v2 endepunktoversikt

### Astar Island (`astar-island/`)
- `astar-island/docs/task2-astar-island.md` — Alt: simuleringsmekanikk, API-spec, scoring, quickstart-kode

### NorgesGruppen (`norgesgruppen/`)
- `norgesgruppen/docs/task3-norgesgruppen.md` — Oppgavebeskrivelse, datasett, annotasjonsformat
- `norgesgruppen/docs/task3-submission-format.md` — Zip-struktur, run.py-kontrakt, sandbox-miljø, sikkerhet
- `norgesgruppen/docs/task3-scoring.md` — Scoring (70% detection + 30% classification), limits
- `norgesgruppen/docs/task3-examples-tips.md` — Random baseline, YOLOv8, ONNX-eksempler, vanlige feil

### Treningsdata (gitignored, kun lokalt)
- `norgesgruppen/data/train/images/` — 248 butikkhyllebilder
- `norgesgruppen/data/train/annotations.json` — COCO-format annotasjoner (~22 700 bbox, 356 kategorier)
- `norgesgruppen/data/NM_NGD_product_images/` — 345 produkter med 7 vinkler (main, front, back, left, right, top, bottom)
- `norgesgruppen/data/NM_NGD_coco_dataset.zip` — Komplett datasett (864 MB)
- `norgesgruppen/data/NM_NGD_product_images.zip` — Produktbilder (60 MB)

## Viktige begrensninger
- **NorgesGruppen sandbox:** Ingen `import os` — bruk `pathlib`. Ingen nettverkstilgang. Max 420 MB weights. Pre-installerte pakker: `ultralytics==8.1.0`, `torch==2.6.0`, `onnxruntime-gpu==1.20.0`, `timm==0.9.12`.
- **Tripletex:** Alle API-kall via proxy (`base_url` fra request). Basic Auth med `0` som brukernavn og `session_token` som passord.
- **Astar Island:** Aldri sett probability 0.0 — bruk minimum floor på 0.01. 50 queries totalt per runde, delt på 5 seeds.

## Konkurranseregler (ALLE AGENTER MÅ FØLGE DISSE)

Full regelverk: `docs/competition-rules.md`

### Kritiske regler
1. **Repo må være offentlig med MIT-lisens** og URL submittet på plattformen FØR fristen (22. mars kl. 15:00 CET)
2. **Ingen hardkodede/forhåndsberegnede svar** — alt må reflektere ekte AI/ML-arbeid
3. **Ingen deling av kode, modellvekter eller løsninger** med andre lag
4. **Ingen platform-abuse** — ikke omgå rate limits, ikke angrip infrastruktur, ikke prøv å trekke ut testdata/ground truth
5. **Ingen score-manipulasjon** — submissions skal maksimere task-performance, ikke manipulere normalisering
6. **AI-verktøy er tillatt og oppmuntret** — Claude, ChatGPT, Copilot, open-source modeller/datasett/papers er OK
7. **Koden blir reviewet** — organisatorene sjekker kodesimilaritet, submissionmønstre og API-logger
8. **Scoring normaliseres per oppgave** (0-100 basert på beste team), samlet = gjennomsnitt av 3 oppgaver

### Submission-rater (verifiserte lag)
- **Tripletex:** 3 concurrent, 5 per task per dag
- **Astar Island:** 50 queries per runde, 5 req/s simulate, 2 req/s submit
- **NorgesGruppen:** 3 per dag, max 2 in-flight

## MCP Docs Server
```
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```
