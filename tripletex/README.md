# Tripletex — AI Accounting Agent

> NM i AI 2026 — Task 1: AI-agent for regnskapsoppgaver via Tripletex API

## Løsning

LLM-drevet agent som mottar regnskapsoppgaver på naturlig språk (7 språk) og utfører dem automatisk via Tripletex API.

### Arkitektur

```
HTTP POST /solve
    │
    ▼
┌─────────────┐
│   Planner   │ ← Gemini 3.1 Pro: parser oppgave, identifiserer task type
└──────┬──────┘
       │
   ┌───┴───┐
   ▼       ▼
Template  Tool Agent
 Engine   (komplekse oppgaver)
   │       │
   ▼       ▼
┌─────────────┐
│  Executor   │ → Tripletex API-kall via proxy
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Verifier   │ → Validerer at oppgaven ble utført korrekt
└─────────────┘
```

### Komponenter

| Komponent | Fil | Beskrivelse |
|---|---|---|
| **Main** | `main.py` | FastAPI `/solve` endpoint, request parsing |
| **Planner** | `prompts/planner.py` | LLM prompt som klassifiserer oppgavetypen |
| **Template Engine** | `template_engine.py` | Forhåndsdefinerte API-sekvenser for vanlige oppgaver |
| **Templates** | `templates.py` | 30 oppgavetyper med felt-mapping og API-rekkefølge |
| **Tool Agent** | `tool_agent.py` | LLM-drevet agent for komplekse/ukjente oppgaver |
| **Executor** | `executor.py` | Utfører API-kall, håndterer autentisering og feil |
| **Tripletex Client** | `tripletex_client.py` | API-wrapper med retry, rate limiting, entity search |

### Nøkkelfunksjoner

- **30 oppgavetyper** (create_employee, create_invoice, register_payment, etc.)
- **7 språk** (NO, EN, ES, PT, NN, DE, FR)
- **Template engine** for raske, pålitelige operasjoner (ingen LLM-kall)
- **Tool agent fallback** for komplekse oppgaver som krever reasoning
- **Auto-entity lookup** — finner eksisterende kunder, leverandører, kontoer i sandbox
- **Efficiency bonus** — minimerer API-kall for høyere score

## Deploy

```bash
gcloud run deploy tripletex-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300 \
  --min-instances 1
```

### Endpoint

```
POST /solve
Content-Type: application/json

{
  "task_description": "Opprett en ny ansatt med navn Ola Nordmann...",
  "session_token": "...",
  "base_url": "https://proxy.ainm.no/..."
}
```

## Modell

- **Gemini 3.1 Pro** via Vertex AI (`location="global"`, VIKTIG: ikke `europe-north1`)
- Brukes til oppgave-klassifisering og tool agent reasoning
- Template engine trenger ikke LLM — direkte API-sekvenser

## Lisens

MIT
