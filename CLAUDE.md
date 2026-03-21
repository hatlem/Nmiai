# NM i AI 2026 — Norgesmesterskapet i kunstig intelligens

## Konkurranse
Vi deltar i NM i AI 2026 (hackathon-format). **1 000 000 NOK i premiepott.**
- **Start:** 19. mars kl. 18:00 CET
- **Frist:** søndag 22. mars kl. 15:00 CET
- **Totalt:** 69 timer (startet 19. mars ~18:30)
- **GitHub-repo:** Settes offentlig med MIT-lisens FØR fristen (ikke før — konkurrenter skal ikke se koden)
- Samlet score = **gjennomsnitt av normaliserte scores på tvers av alle 3 oppgaver (33% hver)**
- Koden må være **åpen kildekode (MIT-lisens)** på offentlig GitHub-repo før fristen
- Alle lagmedlemmer må Vipps-verifiseres for premie
- Max 4 medlemmer per lag

## Arbeidsprinsipp
**Ingen quick fixes.** Hver løsning skal være best-in-class. Vi konkurrerer om 1 MNOK — bruk tid på å forstå problemet ordentlig, velg state-of-the-art metoder, og optimaliser grundig. Halvveis løsninger kaster bort submissions og tid.

## Submission-sporing (VIKTIG)
Alle submission-zips MÅ ha tidsstempel i filnavnet: `submission_YYYYMMDD_HHMMSS.zip`.
`package_submission.sh` gjør dette automatisk og lager en `submission.zip` symlink.
Hold oversikt over submissions og scores slik at vi kan spore fremgang og rulle tilbake om nødvendig.

## Astar Island API Token
Token lagret i `.env.ainm` (gitignored). Les med:
```bash
source .env.ainm && echo $AINM_TOKEN
```
Bruk som Bearer token: `Authorization: Bearer $AINM_TOKEN`
Eller i Node.js scripts: `node calibrate_gt.js --token $(cat .env.ainm | cut -d= -f2)`

## Google Cloud (ubegrenset)
Vi har en dedikert GCP-konto uten kreditgrenser — bruk det vi trenger.

| | |
|---|---|
| Prosjekt | `ainm26osl-710` |
| E-post | devstar7101@gcplab.me |
| Passord | Se `.env` eller team credentials (IKKE commit passord) |
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

## KRITISK: Astar Island — KUN agent_v7.js
**ALDRI start autopilot.py, auto_v5.js, auto_v6.js, agent_final.js eller noen annen Astar Island agent.**
Kun `astar-island/agent_v7.js` skal kjøre. Alle andre agenter stjeler queries fra det felles 50-query budsjettet og ødelegger scoren. autopilot.py er DISABLED (renamed til .DISABLED). Ikke rename den tilbake.

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

## Submission Tracking — NorgesGruppen

Hver submission MÅ logges her med tidskode, dato og innhold. Max 3 per dag.

| # | Dato | Tid | Zip-fil | Innhold | Score | Notater |
|---|------|-----|---------|---------|-------|---------|
| 1 | 2026-03-19 | ~21:00 | submission.zip | YOLOv8x multi-class, no classifier | 0.4759 | det~0.68, cls~0 |
| 2 | 2026-03-19 | ~22:00 | submission_v2.zip | YOLOv8x + WBF multi-scale | ? | Ukjent resultat |
| 3 | 2026-03-20 | 10:33 | submission_v3.zip | YOLOv8x single-class + EfficientNet-B3 classifier, score=det_conf only | — | Ikke submittet |
| 4 | 2026-03-20 | 10:38 | submission_20260320_103349.zip | YOLO26-x ONNX + DINOv2, score=det*cls^0.15 | **0.6740** | 287.9s, +41.6% vs forrige |
| 5 | 2026-03-20 | 11:52 | submission_20260320_115200.zip | YOLO26-x ONNX + DINOv2 + multi-class YOLO hybrid, score=det_score only | — | Ikke submittet (failed exit code 1) |
| 6 | 2026-03-20 | 14:16 | submission_20260320_141629.zip | YOLO26-x FP16 ONNX + DINOv2 only, score=det_score, CLAHE on crops only, conf=0.01, no SAHI, no multi-class | KLAR | 257MB, 2/3 weights. Fixes: no timeout, pure det ranking |
| 7 | 2026-03-21 | 15:30 | submission_20260321_153000.zip | YOLO26-x ONNX + DINOv2-Base v2 (FP16, epoch 26, val=91.5%, Focal+Mixup+EMA) | ? | 250MB, 2/3 weights. V2 classifier |
| 8 | 2026-03-20 | 18:41 | submission_20260320_182916.zip | 3-modell WBF ensemble (pseudo 0.789 + fold0 0.726 + fold2 0.749) + TTA, conf=0.01 | **0.9139** | 38.2s. Ensemble er game-changer |
| 9 | 2026-03-20 | 20:55 | submission_20260320_203746.zip | pseudo + fold4 + fold2, conf=0.05, TTA | 0.9119 | 38.5s. conf=0.05 VERRE enn 0.01 |
| 10 | 2026-03-20 | 22:49 | submission_20260320_224900.zip | pseudo + **1600px**(0.771) + fold2, conf=**0.001**, TTA | **0.9158** | 42.6s. NY BEST! 1600px + lavere conf hjelper |

### Nåværende status (oppdatert 21. mars 10:25)

#### Samlet konkurranse — Astar Island leaderboard
- **R13 = 91.3 pts (#22) — NY REKORD!** (forrige: R9=90.4)
- **R14 aktiv** — queries brukt av autopilot.py (nå drept), v7 submittet lookup-only
- **agent_v7.js er ENESTE agent** — autopilot.py drept for å unngå query-konflikter

#### Astar Island — Per-runde scores (API: /my-rounds)
| Runde | Raw Score | Rank | Queries | Agent | Notater |
|---|---|---|---|---|---|
| R1 | 55.4 | #23 | 50 | tidlig v1 | Lookup only, ingen calibration |
| R2 | 76.9 | #35 | 50 | v3? | Bedre, men langt fra topp |
| R3 | — | — | 0 | MISSING | Ingen submission! |
| R4 | 78.8 | #35 | 50 | ? | OK |
| R5 | 67.6 | #68 | 50 | ? | Dårlig — trolig query-bug |
| R6 | 60.6 | #83 | 50 | ? | Dårlig — trolig query-bug |
| R7 | 63.2 | #55 | 50 | ? | Dårlig |
| R8 | 64.8 | #115 | 50 | ? | Vår verste rank! |
| **R9** | **90.4** | **#29** | 50 | agent_final | Beste runde! |
| R10 | 59.1 | #133 | 50 | agent_final | Katastrofe — noe gikk galt |
| **R11** | **88.4** | **#18** | 50 | agent_final | Nest beste |
| R12 | 50.0 | #71 | 50 | agent_final | Veldig dårlig |
| **R13** | **91.3** | **#22** | 50 | agent_v7 (lookup only) | **NY REKORD!** Oppdatert lookup (228 bins, 11 runder) |
| R14 | ? | ? | 50 | v7 lookup + autopilot queries | Queries brukt av autopilot.py |

**Mønster:** Veksler mellom gode (88-90) og dårlige (50-65) runder. Bugfixene i v7 skal stabilisere dette.

#### Astar Island — agent_v7 fixes (deployed 21. mars 07:05)
1. Rate limit: bruker API budget response i stedet for rlCount heuristic
2. Survival: beregner fra observert grid (var alltid 100% fra metadata)
3. KT n>=1: adaptive alpha (0.25/0.35/0.5) i stedet for n>=3 cutoff
4. Port suppression FØR KT blend
5. Konsentrerte viewports: 1 per seed, gjentatt for sterk KT
6. SIM_DELAY 350ms (var 280ms)

#### NorgesGruppen — Beste score: 0.9158
- **Rank:** Ukjent (NorgesGruppen API ikke tilgjengelig via CLI)
- **Submissions brukt:** 10+ (se tabell under)
- **Submissions igjen:** 3 per dag, søndag er siste dag

#### Tripletex — Score: 23.4
- **Rank: #150** av 329 lag
- **Tasks touched:** 18/30
- **Submissions:** 152
- **Tier1: 13.09, Tier2: 10.32, Tier3: 0**
- **Topp 5:** Ave Christus Rex, Slop Overflow, Propulsion Optimizers, websecured.io, Proof Left to the Reader

### Hva vi har lært
| Endring | Effekt | Lærdom |
|---|---|---|
| Ensemble 3 modeller + TTA | +0.24 (0.674→0.914) | Ensemble er ALT |
| conf 0.01 → 0.05 | -0.002 (0.914→0.912) | Lavere conf = bedre (mer recall) |
| conf 0.01 → 0.001 | +0.002 (0.914→0.916) | Enda lavere conf hjelper |
| fold0 → fold4 (sterkere, same arch) | -0.0003 | Same arkitektur = null forbedring |
| fold0 → 1600px (annen oppløsning) | +0.002 | **Diversitet slår styrke** |
| Two-stage (YOLO+DINOv2) | 0.674 | Dårlig — multi-class direkte er bedre |
| Ren multi-class YOLO | 0.476 | For svak alene |

### Blindgater (IKKE prøv igjen)
- **RT-DETR-x:** mAP50=0 etter 45+ epochs. Feilet TWICE med forskjellig LR.
- **DINOv2 classifier:** Topper på 91% val_acc. Two-stage er alltid dårligere enn end-to-end.
- **conf=0.05:** Verre enn 0.001. ALDRI øk confidence threshold.
- **Fold-swapping (same arch):** Null effekt. fold0→fold4 ga -0.0003.
- **Model Soup (vekt-averaging):** 0.536 mAP — katastrofalt.
- **Sterkere individuelle modeller ≠ bedre ensemble:** pseudo_long(0.781) ga 0.9141 vs fold2(0.749) ga 0.9158.
- **Lengre trening (500ep):** Overtrent. 167ep var bedre enn 501ep.
- **Multi-scale TTA (640+960+1280):** 0.9149 vs 0.9158 med bare 1280+flip.
- **Soft-NMS etter WBF:** Ingen forbedring.
- **WBF tuning (iou=0.43/0.45, conf_type=max, model weighting, temperature scaling):** 0.8930 — MYYYE verre.
- **YOLO11-x i ensemble:** 0.9153 vs 0.9158 med fold2. Annen arkitektur hjalp ikke.
- **YOLOv8x(0.799)+YOLO11-x+pseudo ensemble:** 0.9153. FP16 konvertering kan ha skadet.
- **Syntetisk data trening:** Maks 0.788. Ikke bedre enn original data.
- **Oversampling:** Maks 0.774.
- **Label smoothing:** Maks 0.782.
- **Strong augmentation:** Maks 0.746.
- **Cosine LR:** Maks 0.717.

### Alle trente modeller (sortert etter mAP50)
| # | mAP50 | Arkitektur | Strategi | VM |
|---|---|---|---|---|
| 1 | **0.799** | YOLOv8x | Original trening | nmiai-train-yolo |
| 2 | 0.789 | YOLO26-x | Pseudo-labels | yolo26-train |
| 3 | 0.788 | YOLOv8x | Syntetisk data (A100) | yolo26-a100 |
| 4 | 0.784 | YOLO11-x | Standard | yolo26-a100 |
| 5 | 0.782 | YOLOv8x | Label smoothing | nmiai-train-gpu |
| 6 | 0.782 | YOLOv8x | Syntetisk data (T4) | yolo26-t4-2 |
| 7 | 0.781 | YOLO11-l | Standard | yolo26-train |
| 8 | 0.774 | YOLO26-x | Oversampled | yolo26-t4-2 |
| 9 | 0.773 | YOLOv8x | Label smooth+oversample | yolo26-l4-3 |
| 10 | 0.771 | YOLO26-x | 1600px resolution | yolo26-a100 |
| 11 | 0.758 | YOLO26-x | K-fold 4 | yolo26-a100 |
| 12 | 0.749 | YOLO26-x | K-fold 2 | yolo26-a100 |

### Ikke prøvd ennå (potensielt lovende)
- **Objects365 pretrain → fine-tune:** +3.1 AP for sjeldne objekter (forskning). Trenger ~4-6 timer.
- **HuggingFace shelf-detection modell som backbone:** foduucom/product-detection-in-shelf-yolov8
- **NVIDIA retail pretrained modell**

### Beste submission: 0.9158
- **Modeller:** pseudo(0.789) + 1600px(0.771) + fold2(0.749)
- **Config:** conf=0.001, WBF iou=0.55, TTA=1280+flip ONLY
- **Fil:** submission_20260320_224900.zip

### Regler for submission-logging
- **ALLTID** oppdater tabellen over når en ny submission lages
- Inkluder dato (YYYY-MM-DD), tidspunkt (HH:MM), zip-filnavn, kort beskrivelse av innhold
- Oppdater Score-kolonnen når resultatet er kjent
- NorgesGruppen har **max 3 submissions per dag**

### VIKTIG: Multi-class YOLO slår two-stage

**GCP eval på treningsdata (20. mars):**

| Tilnærming | Det mAP | Cls mAP | Score | Bilder | Tid |
|---|---|---|---|---|---|
| **Multi-class YOLOv8x** (end-to-end) | **0.9474** | **0.8758** | **0.9259** | 248/248 | 164s |
| Two-stage YOLO26+DINOv2 (sub #4) | ~0.82 | ~0.35 | 0.6740 | 117/248 | 288s |

**IKKE bruk two-stage (YOLO26+DINOv2).** Grunner:
1. DINOv2 classifier topper på 91.3% val_acc uansett trening (V1 og V2 ga identisk resultat)
2. Two-stage er tregere — bare 117/248 bilder ble prosessert
3. Category mapping mellom detektor og classifier skaper feil
4. Multi-class YOLO ser spatial context (hylleposisjon) for klassifisering
5. Score er naturlig kalibrert — riktige klasser får høyere confidence

**All fremtidig innsats bør gå til å forbedre multi-class YOLO:**
- Flere treningsepochs
- Bedre augmentation
- Større modell (YOLO26-x multi-class)
- Ensemble av multi-class modeller

### Classifier Training Tracker (DEPRECATED — bruk multi-class YOLO i stedet)

| Versjon | Modell | Val Acc | Mean/Cat | Top-5 | Epochs | Status |
|---|---|---|---|---|---|---|
| v1 | DINOv2 + CE + label_smoothing | 91.3% | ? | ? | 20 | DEPRECATED |
| v2 | DINOv2 + FocalLoss + Mixup + EMA | 91.3% (ep19) | 84.1% | 97.5% | 21/50 | DEPRECATED |

### GCP Training Fleet (oppdatert 20. mars 18:44)

| VM | GPU | Oppgave | Epochs | Best mAP50 | Status |
|---|---|---|---|---|---|
| yolo26-a100 | A100 40GB | K-fold 0,1,2 ✅ + fold 3 trener | 0:151, 1:113, 2:141, 3:62 | 0.749 (fold2) | Fold 3 trener |
| yolo26-train | L4 24GB | Pseudo-label multi-class | 132 | **0.789** | Trener videre |
| nmiai-train-fast | L4 24GB | Progressive resize pretrain | ? | ? | Trener |
| yolo26-l4-3 | L4 24GB | K-fold 1 (parallell) | 82 | 0.717 | Ferdig |
| yolo26-t4-1 | T4 16GB | K-fold 2 (parallell) | 91 | 0.736 | Trener |
| yolo26-t4-2 | T4 16GB | K-fold 3 (parallell) | 89 | 0.693 | Trener |
| classifier-train | L4 24GB | DINOv2 v2 | 44/50 | 91.8% val | Nesten ferdig |

### Tilgjengelige ONNX-modeller for ensemble

| Modell | mAP50 | Fil | Størrelse | Kilde |
|---|---|---|---|---|
| pseudo_best.onnx | **0.789** | ✅ Lastet ned | 109 MB | Pseudo-label (L4 train5) |
| fold2_best.onnx | **0.749** | ✅ Lastet ned | 108 MB | K-fold 2 (A100 kfold_2) |
| fold0_best.onnx | 0.726 | ✅ Lastet ned | 109 MB | K-fold 0 (A100 kfold_0) |
| fold1_best.onnx | 0.718 | ✅ Lastet ned | 113 MB | K-fold 1 (A100 kfold_1) |
| fold3 | ~0.678 | ⏳ Trener | ~108 MB | K-fold 3 (A100 kfold_3) |

**Max 3 filer × ~110 MB = 330 MB / 420 MB**

### Score-historikk — NorgesGruppen

| Sub | Score | Metode |
|---|---|---|
| #1 | 0.476 | YOLOv8x single model |
| #4 | 0.674 | YOLO26-x + DINOv2 (timeout) |
| **#8** | **0.914** | **3-modell WBF ensemble + TTA** |
| **#10** | **0.9158** | **pseudo+1600px+fold2, conf=0.001** |
| Topp | 0.920 | Havvind |

### Neste steg (oppdatert 21. mars 08:15)

**Astar Island (høyest prioritet — mest å hente):**
1. Overvåk agent_v7 på R14+ — target: 88+ raw konsistent
2. Hvis v7 fikser ustabiliteten: weighted_score stiger raskt (dårlige runder trekker oss ned)
3. Mulig forbedring: legg til ground truth fra R9/R11 (våre beste) i gt_lookup.json

**NorgesGruppen:**
1. Beste: 0.9158, nær toppen (0.920)
2. Sjekk om nye modeller er ferdigtrent på GCP
3. 3 submissions per dag — bruk dem klokt

**Tripletex:**
1. #150 — langt bak, men tier-multipliers gir stor oppside
2. Fokuser på tier 2/3 tasks for å øke score
3. Cloud Run agent kjører allerede

## MCP Docs Server
```
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```
