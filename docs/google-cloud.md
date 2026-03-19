# Google Cloud

Google Cloud is an official partner of NM i AI 2026. Selected teams receive a free GCP account with a dedicated project — no credit limits, no billing setup, ready to use.

## What You Get

- A @gcplab.me Google account
- A dedicated GCP project with full access to Cloud services
- No credit limits — use what you need for the competition
- Access to Gemini models, Cloud Run, Vertex AI, and more
- Collaboration tools: Gmail, Google Docs, Google Chat, NotebookLM

## Who Gets an Account

Accounts are limited. We prioritize teams based on:

- Team verification — all members must be Vipps-verified
- Application — you must apply for a GCP account through the platform
- Competition activity — active teams are prioritized
- Younger participants — we prioritize students and early-career developers

You don't need a GCP account to compete — you can host your endpoint anywhere. But if you don't have your own cloud environment, this is a great way to get started for free.

## Account Setup

Once your team receives GCP credentials, follow these steps to get started.

### Log In

1. Open a new Chrome profile or an incognito window
2. Go to console.cloud.google.com
3. Sign in with your @gcplab.me email and password
4. Select your assigned project from the project dropdown

> **Tip:** Use a separate Chrome profile to keep your competition account separate from personal Google accounts.

### Cloud Shell

Cloud Shell is a free terminal built into the Google Cloud console. It comes with Python, git, gcloud CLI, and Docker pre-installed — everything you need to build and deploy.

Open it by clicking the terminal icon in the top-right corner of the console.

Cloud Shell gives you:

- A Linux VM with 5 GB persistent home directory
- Python 3, Node.js, Go, Java pre-installed
- Docker for building container images
- gcloud CLI already authenticated with your project
- Free to use — no compute charges

### Cloud Shell Editor

Need a full IDE? Click "Open Editor" in Cloud Shell or go directly to ide.cloud.google.com.

This is a VS Code-based editor in the browser with:

- File explorer, terminal, extensions
- Gemini Code Assist — AI coding companion built in
- Direct access to your Cloud Shell files

### Gemini Tools

Your GCP account includes several AI assistants:

| Tool | Where | What it does |
|---|---|---|
| Gemini Code Assist | Cloud Shell Editor | AI coding companion in the IDE |
| Gemini CLI | Cloud Shell terminal | Type `gemini` in the terminal for CLI-based AI help |
| Gemini Cloud Assist | Console sidebar | Ask questions about GCP services and configuration |
| AI Studio | aistudio.google.com | Experiment with Gemini models directly |

To enable Gemini Cloud Assist, click the Gemini icon in the console and hit "Enable".

## Deploy on Cloud Run

Two of the three competition tasks — Tripletex and Astar Island — require you to host a public HTTPS endpoint that our validators call. Cloud Run is the easiest way to deploy one.

### What is Cloud Run?

Cloud Run takes a Docker container and gives you a public HTTPS URL. You push your code, it handles scaling, TLS, and everything else. You only pay for actual requests (and with your GCP account, it's free).

### Step 1: Write Your Endpoint

Here's a minimal FastAPI endpoint that matches the competition format:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/solve")
async def solve(request: dict):
    prompt = request.get("prompt", "")
    credentials = request.get("tripletex_credentials", {})

    # Your AI agent logic here:
    # 1. Parse the prompt
    # 2. Call the Tripletex API using the provided credentials
    # 3. Complete the accounting task

    return {"status": "completed"}
```

### Step 2: Create a Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

And a requirements.txt:

```
fastapi
uvicorn[standard]
requests
```

### Step 3: Deploy

Open Cloud Shell and run:

```bash
# Clone your repo (or upload files via Cloud Shell Editor)
cd ~
git clone <your-repo-url>
cd your-project

# Deploy to Cloud Run (builds and deploys in one command)
gcloud run deploy my-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300
```

That's it. Cloud Run builds the Docker image, deploys it, and gives you a URL like:

```
https://my-agent-xxxxx-lz.a.run.app
```

### Step 4: Submit Your URL

1. Copy the Cloud Run URL
2. Go to the submission page for your task at app.ainm.no
3. Paste the URL and submit
4. Our validators will start calling your endpoint

## Tips

### Use europe-north1 Region

Deploy in `europe-north1` (Finland) — same region as our validators. Lower latency = faster scoring.

```bash
gcloud run deploy my-agent --region europe-north1 ...
```

### Handle Cold Starts

Cloud Run scales to zero when idle. The first request after idle may take a few seconds. To keep it warm:

```bash
gcloud run deploy my-agent --min-instances 1 ...
```

### Increase Memory for LLMs

If you're calling external LLM APIs (like Vertex AI), the default 512 MB is fine. If you're running a local model, increase memory:

```bash
gcloud run deploy my-agent --memory 2Gi --cpu 2 ...
```

### Update Your Deployment

After making changes, just run the same deploy command again:

```bash
gcloud run deploy my-agent --source . --region europe-north1 --allow-unauthenticated
```

### View Logs

```bash
gcloud run services logs read my-agent --region europe-north1 --limit 50
```

## Which Tasks Need Cloud Run?

| Task | Submission type | Cloud Run? |
|---|---|---|
| Tripletex | HTTPS endpoint (/solve) | Yes |
| Astar Island | HTTPS endpoint (/solve) | Yes |
| NorgesGruppen Data | Code upload (.zip) | No |

## Services & Tools

### Hosting Your Endpoint

| Service | Use case | When to use |
|---|---|---|
| Cloud Run | Deploy containerized APIs | Tripletex & Astar Island tasks — this is the go-to |
| Compute Engine | Full VM (any OS) | Need GPU or persistent server |

### AI & Machine Learning

| Service | Use case | When to use |
|---|---|---|
| Vertex AI | Managed ML platform | Access Gemini and other models via API |
| Model Garden | Pre-trained model catalog | Browse and deploy models (Gemini, Llama, Mistral) |
| AI Studio | Experiment with Gemini | Quick prototyping, prompt engineering |

### Using Vertex AI from Your Endpoint

**VIKTIG: Gemini 3.1-modeller krever `location="global"`, ikke region-spesifikk!**

```python
import vertexai
from vertexai.generative_models import GenerativeModel

# Gemini 3.1 (nyeste, beste) — MÅ bruke location="global"
vertexai.init(project="your-project-id", location="global")
model = GenerativeModel("gemini-3.1-pro-preview")
# Alternativer: "gemini-3.1-flash-lite-preview" (rask/billig)

# Gemini 2.5 (fallback) — fungerer med europe-north1
# vertexai.init(project="your-project-id", location="europe-north1")
# model = GenerativeModel("gemini-2.5-pro")

response = model.generate_content("Parse this accounting task: ...")
print(response.text)
```

Install with: `pip install google-cloud-aiplatform`

### Data & Storage

| Service | Use case | When to use |
|---|---|---|
| Cloud Storage | File storage (buckets) | Store datasets, model weights, logs |
| Cloud SQL | Managed PostgreSQL/MySQL | Need a relational database |
| BigQuery | Data warehouse | Analyze large datasets with SQL |

### Development Tools

| Tool | How to access | What it does |
|---|---|---|
| Cloud Shell | Console top-right icon | Free terminal with everything pre-installed |
| Cloud Shell Editor | "Open Editor" button | VS Code in the browser |
| Gemini Code Assist | Cloud Shell Editor sidebar | AI coding companion |
| Gemini CLI | `gemini` in Cloud Shell | AI assistant in the terminal |
| Cloud Build | Automatic with `gcloud run deploy --source .` | Builds your Docker images |

### Collaboration

Your @gcplab.me account also works with:

- Gmail — communicate with teammates
- Google Docs — shared documentation
- Google Chat — team messaging
- NotebookLM — AI-powered research notebook
