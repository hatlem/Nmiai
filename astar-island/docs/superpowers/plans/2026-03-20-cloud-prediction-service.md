# Astar Island: Cloud Prediction Service

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy a stable Cloud Run prediction service that polls for rounds, uses all 50 queries with explore-then-exploit strategy, and never crashes. Target: 85+ raw score.

**Architecture:** Single Python Cloud Run service with FastAPI. Polls Astar Island API every 30s. On new round: (1) submit lookup immediately, (2) explore with 5 queries to understand round params via settlement metadata, (3) compute shift, (4) exploit with 45 queries on high-value areas, (5) progressively resubmit with improved predictions. Uses gt_lookup.json baked into container. No local state dependencies.

**Tech Stack:** Python 3.11, FastAPI, numpy, Cloud Run (europe-north1), Cloud Storage for GT cache

---

## File Structure

```
astar-island/cloud/
├── main.py              # FastAPI app + polling loop
├── predictor.py         # Lookup + shift prediction logic
├── query_strategy.py    # Explore-then-exploit query planning
├── api_client.py        # Astar Island API wrapper with retry
├── metadata_analyzer.py # Settlement metadata → param inference
├── gt_lookup.json       # Baked-in lookup table (copied from local)
├── Dockerfile
├── requirements.txt
└── test_predictor.py    # Local tests against cached GT
```

### Task 1: API Client with Retry

**Files:**
- Create: `astar-island/cloud/api_client.py`
- Test: `astar-island/cloud/test_api_client.py`

A robust API client that NEVER crashes. All methods return results or None.

- [ ] **Step 1: Write test**

```python
# test_api_client.py
def test_retry_on_429():
    """Client retries on 429 with backoff"""
    client = AstarClient(token="test")
    # Mock 429 response then 200
    assert client._retry_request("GET", "/budget") is not None

def test_never_throws():
    """Client returns None on error, never throws"""
    client = AstarClient(token="invalid")
    assert client.get_rounds() is not None or client.get_rounds() is None
```

- [ ] **Step 2: Implement api_client.py**

```python
import httpx, time, logging
log = logging.getLogger(__name__)

class AstarClient:
    BASE = "https://api.ainm.no/astar-island"

    def __init__(self, token: str):
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0
        )

    def _req(self, method, path, json=None, retries=3):
        for attempt in range(retries):
            try:
                r = self.client.request(method, f"{self.BASE}{path}", json=json)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                log.warning(f"{method} {path} attempt {attempt}: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
        return None

    def get_rounds(self): return self._req("GET", "/rounds") or []
    def get_round(self, rid): return self._req("GET", f"/rounds/{rid}")
    def get_budget(self): return self._req("GET", "/budget")
    def get_my_rounds(self): return self._req("GET", "/my-rounds") or []
    def simulate(self, rid, si, x, y, w=15, h=15):
        return self._req("POST", "/simulate", {"round_id":rid,"seed_index":si,"viewport_x":x,"viewport_y":y,"viewport_w":w,"viewport_h":h})
    def submit(self, rid, si, pred):
        return self._req("POST", "/submit", {"round_id":rid,"seed_index":si,"prediction":pred})
    def get_analysis(self, rid, si):
        return self._req("GET", f"/analysis/{rid}/{si}")
```

- [ ] **Step 3: Run test, verify pass**
- [ ] **Step 4: Commit**

### Task 2: Predictor (Lookup + Shift)

**Files:**
- Create: `astar-island/cloud/predictor.py`
- Create: `astar-island/cloud/test_predictor.py`

Pure prediction logic. No API calls. Takes initial grid + shift → returns prediction tensor.

- [ ] **Step 1: Write test**

```python
# test_predictor.py
import json, numpy as np
from predictor import Predictor

def test_static_cells():
    """Mountains and ocean get near-deterministic predictions"""
    p = Predictor(json.load(open("gt_lookup.json")))
    grid = [[5, 10], [4, 1]]  # mountain, ocean, forest, settlement
    pred = p.predict(grid, 2, 2)
    assert pred[0][0][5] > 0.99  # mountain stays mountain
    assert pred[0][1][0] > 0.99  # ocean stays ocean

def test_shift_changes_predictions():
    """Applying shift modifies non-static cells"""
    p = Predictor(json.load(open("gt_lookup.json")))
    grid = [[1]]  # settlement
    pred_no_shift = p.predict(grid, 1, 1)
    shift = {1: [0.5, 2.0, 1.0, 1.0, 0.5, 1.0]}  # double settlement survival
    pred_shifted = p.predict(grid, 1, 1, shift=shift)
    assert pred_shifted[0][0][1] > pred_no_shift[0][0][1]

def test_probabilities_sum_to_one():
    """All cells sum to 1.0"""
    p = Predictor(json.load(open("gt_lookup.json")))
    # Load a real initial grid from cache
    data = json.load(open("cache/r1_gt_s0.json"))
    pred = p.predict(data["initial_grid"], data["height"], data["width"])
    for y in range(data["height"]):
        for x in range(data["width"]):
            assert abs(sum(pred[y][x]) - 1.0) < 0.01

def test_score_against_gt():
    """Lookup predictions score 80+ against GT (in-sample)"""
    p = Predictor(json.load(open("gt_lookup.json")))
    data = json.load(open("cache/r1_gt_s0.json"))
    pred = p.predict(data["initial_grid"], data["height"], data["width"])
    score = p.score_kl(pred, data["ground_truth"], data["height"], data["width"])
    assert score > 75, f"Expected 75+, got {score}"
```

- [ ] **Step 2: Implement predictor.py**

Key features:
- `predict(grid, H, W, shift=None, damping=0.7, floor=0.001)` → H×W×6 prediction
- `compute_shift(obs_transitions, lookup)` → per-class shift dict
- `score_kl(pred, gt, H, W)` → score (for local testing)
- Context key: `{ic}_{food}_{coastal}_{dist}_{nSett}` with fallback chain
- Shift applied as: `p[c] *= pow(shift[ic][c], damping)` with coastal awareness
- Clip shift to [0.5, 3.0]
- Floor 0.001, non-coastal port suppression

- [ ] **Step 3: Run tests, verify all pass**
- [ ] **Step 4: Score against ALL cached GT seeds, log results**

```bash
python -c "
from predictor import Predictor
import json, os
p = Predictor(json.load(open('gt_lookup.json')))
for f in sorted(os.listdir('cache')):
    if not f.endswith('.json') or 'gt_s' not in f: continue
    d = json.load(open(f'cache/{f}'))
    score = p.score_kl(p.predict(d['initial_grid'],d['height'],d['width']), d['ground_truth'],d['height'],d['width'])
    print(f'{f}: {score:.1f}')
"
```

- [ ] **Step 5: Commit**

### Task 3: Metadata Analyzer

**Files:**
- Create: `astar-island/cloud/metadata_analyzer.py`

Extract hidden parameter estimates from settlement metadata returned by /simulate.

- [ ] **Step 1: Write test**

```python
def test_extract_params():
    """Extract winter/aggression/trade from settlement metadata"""
    settlements = [
        {"population": 3.0, "food": 2.0, "wealth": 1.0, "defense": 0.5, "alive": True, "has_port": True, "owner_id": 0},
        {"population": 0.5, "food": 0.1, "wealth": 0.0, "defense": 0.2, "alive": False, "owner_id": 1},
    ]
    params = extract_params(settlements)
    assert "survival_rate" in params
    assert "avg_food" in params
    assert "faction_count" in params
```

- [ ] **Step 2: Implement**

```python
def extract_params(all_settlements_across_queries):
    """From observed settlement metadata, estimate round characteristics."""
    alive = [s for s in all_settlements_across_queries if s.get("alive")]
    dead = [s for s in all_settlements_across_queries if not s.get("alive")]

    total = len(alive) + len(dead)
    survival_rate = len(alive) / max(total, 1)
    avg_food = np.mean([s["food"] for s in alive]) if alive else 0
    avg_pop = np.mean([s["population"] for s in alive]) if alive else 0
    avg_wealth = np.mean([s["wealth"] for s in alive]) if alive else 0
    port_rate = sum(1 for s in alive if s.get("has_port")) / max(len(alive), 1)
    faction_count = len(set(s.get("owner_id", 0) for s in alive))

    return {
        "survival_rate": survival_rate,
        "avg_food": avg_food,
        "avg_population": avg_pop,
        "avg_wealth": avg_wealth,
        "port_rate": port_rate,
        "faction_count": faction_count,
        "death_rate": 1 - survival_rate,
    }
```

- [ ] **Step 3: Run test, commit**

### Task 4: Query Strategy (Explore → Exploit)

**Files:**
- Create: `astar-island/cloud/query_strategy.py`

- [ ] **Step 1: Write test**

```python
def test_explore_phase():
    """First 5 queries: 1 per seed, different viewports"""
    plan = QueryPlanner(initial_states, H=40, W=40, budget=50)
    explore = plan.explore_phase()
    assert len(explore) == 5
    seeds = set(q["seed"] for q in explore)
    assert len(seeds) == 5  # one per seed

def test_exploit_adapts_to_params():
    """After exploration, exploit focuses on high-value areas"""
    plan = QueryPlanner(initial_states, H=40, W=40, budget=50)
    plan.update_params({"survival_rate": 0.2})  # high death → query settlements
    exploit = plan.exploit_phase(remaining=45)
    sett_queries = sum(1 for q in exploit if q["reason"] == "settlement")
    assert sett_queries > 30  # mostly settlement queries
```

- [ ] **Step 2: Implement**

```python
class QueryPlanner:
    def explore_phase(self):
        """5 queries: 1 viewport per seed, centered on densest settlement cluster"""
        queries = []
        for si, state in enumerate(self.initial_states):
            # Find densest settlement cluster center
            setts = state["settlements"]
            cx = int(np.mean([s["x"] for s in setts]))
            cy = int(np.mean([s["y"] for s in setts]))
            queries.append({"seed": si, "x": max(0,min(W-15,cx-7)), "y": max(0,min(H-15,cy-7))})
        return queries

    def exploit_phase(self, remaining, params=None):
        """45 queries: repeat high-value viewports based on learned params"""
        # Score each possible viewport by expected information gain
        # Settlement viewports get 10 importance, expansion zones get 5
        # Repeat best viewports to get n≥5 per dynamic cell
        ...
```

- [ ] **Step 3: Run test, commit**

### Task 5: Main Service (FastAPI + Poll Loop)

**Files:**
- Create: `astar-island/cloud/main.py`

- [ ] **Step 1: Implement main.py**

```python
import asyncio, json, os, logging
from fastapi import FastAPI
from api_client import AstarClient
from predictor import Predictor
from query_strategy import QueryPlanner
from metadata_analyzer import extract_params

app = FastAPI()
log = logging.getLogger(__name__)
TOKEN = os.environ["AINM_TOKEN"]
client = AstarClient(TOKEN)
predictor = Predictor(json.load(open("gt_lookup.json")))
completed = set()

@app.get("/health")
def health():
    return {"status": "ok", "completed": len(completed)}

@app.on_event("startup")
async def startup():
    asyncio.create_task(poll_loop())

async def poll_loop():
    while True:
        try:
            await process_rounds()
        except Exception as e:
            log.error(f"Poll error: {e}")
        await asyncio.sleep(30)

async def process_rounds():
    rounds = client.get_rounds()
    for r in rounds:
        if r["status"] != "active" or r["id"] in completed:
            continue
        completed.add(r["id"])
        try:
            await process_round(r)
        except Exception as e:
            log.error(f"Round {r['round_number']} error: {e}")

async def process_round(round):
    log.info(f"R{round['round_number']} ACTIVE")
    detail = client.get_round(round["id"])
    H, W = detail["map_height"], detail["map_width"]

    # PHASE 1: Submit lookup immediately
    for si in range(detail["seeds_count"]):
        pred = predictor.predict(detail["initial_states"][si]["grid"], H, W)
        client.submit(round["id"], si, pred)
    log.info("Phase 1: lookup submitted")

    # PHASE 2: Explore (5 queries)
    budget = client.get_budget()
    left = (budget or {}).get("queries_max", 50) - (budget or {}).get("queries_used", 0)
    if left <= 0: return

    planner = QueryPlanner(detail["initial_states"], H, W, left)
    explore_qs = planner.explore_phase()

    all_obs_transitions = {}  # ic → [NC] counts
    all_settlements = []
    for q in explore_qs:
        result = client.simulate(round["id"], q["seed"], q["x"], q["y"])
        if not result: continue
        # Accumulate transitions
        _accumulate(result, detail["initial_states"][q["seed"]]["grid"], H, W, all_obs_transitions)
        all_settlements.extend(result.get("settlements", []))
        await asyncio.sleep(0.3)

    # PHASE 3: Analyze metadata + compute initial shift
    params = extract_params(all_settlements)
    shift = predictor.compute_shift(all_obs_transitions)
    log.info(f"Phase 2-3: explored, survival={params['survival_rate']:.2f}, shift computed")

    # Resubmit with shift
    for si in range(detail["seeds_count"]):
        pred = predictor.predict(detail["initial_states"][si]["grid"], H, W, shift=shift)
        client.submit(round["id"], si, pred)
    log.info("Phase 3: shifted predictions submitted")

    # PHASE 4: Exploit (remaining queries)
    exploit_qs = planner.exploit_phase(left - len(explore_qs), params)
    for i, q in enumerate(exploit_qs):
        result = client.simulate(round["id"], q["seed"], q["x"], q["y"])
        if not result: continue
        _accumulate(result, detail["initial_states"][q["seed"]]["grid"], H, W, all_obs_transitions)
        all_settlements.extend(result.get("settlements", []))
        await asyncio.sleep(0.3)

        # Every 10 queries: resubmit
        if (i + 1) % 10 == 0:
            shift = predictor.compute_shift(all_obs_transitions)
            for si in range(detail["seeds_count"]):
                pred = predictor.predict(detail["initial_states"][si]["grid"], H, W, shift=shift)
                client.submit(round["id"], si, pred)
            log.info(f"Phase 4: resubmit after {i+1} exploit queries")

    log.info(f"R{round['round_number']} COMPLETE")
```

- [ ] **Step 2: Write Dockerfile**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 3: Write requirements.txt**

```
fastapi
uvicorn
httpx
numpy
```

- [ ] **Step 4: Test locally**

```bash
cd astar-island/cloud
AINM_TOKEN=$(cat ../../.env.ainm | cut -d= -f2) uvicorn main:app --port 8080 &
curl localhost:8080/health
# Watch logs for round processing
```

- [ ] **Step 5: Commit**

### Task 6: Deploy to Cloud Run

- [ ] **Step 1: Copy gt_lookup.json into cloud/**

```bash
cp astar-island/gt_lookup.json astar-island/cloud/gt_lookup.json
```

- [ ] **Step 2: Deploy**

```bash
cd astar-island/cloud
gcloud run deploy astar-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300 \
  --min-instances 1 \
  --set-env-vars "AINM_TOKEN=$(cat ../../.env.ainm | cut -d= -f2)"
```

- [ ] **Step 3: Verify health endpoint**

```bash
curl https://astar-agent-xxxxx-lz.a.run.app/health
```

- [ ] **Step 4: Monitor first round**

```bash
gcloud run services logs read astar-agent --region europe-north1 --limit 50
```

- [ ] **Step 5: Kill ALL local agents**

```bash
pkill -f "auto_submit"
pkill -f "watch_and_cal"
pkill -f "continuous"
```

### Task 7: Local Test Suite

**Files:**
- Create: `astar-island/cloud/test_full_pipeline.py`

Run full pipeline against ALL cached GT with simulated observations.

- [ ] **Step 1: Implement**

```python
"""Test full pipeline against all 7 rounds of GT data."""
import json, os, numpy as np
from predictor import Predictor

def test_all_rounds():
    predictor = Predictor(json.load(open("gt_lookup.json")))
    cache = "../cache"

    scores = {}
    for f in sorted(os.listdir(cache)):
        if "gt_s" not in f: continue
        data = json.load(open(f"{cache}/{f}"))

        # Simulate observations
        obs_trans = simulate_queries(data, predictor, n_queries=10)
        shift = predictor.compute_shift(obs_trans)
        pred = predictor.predict(data["initial_grid"], data["height"], data["width"], shift=shift)
        score = predictor.score_kl(pred, data["ground_truth"], data["height"], data["width"])
        scores[f] = score

    avg = np.mean(list(scores.values()))
    print(f"Average: {avg:.1f}")
    for f, s in sorted(scores.items()):
        print(f"  {f}: {s:.1f}")
    assert avg > 80, f"Expected avg > 80, got {avg:.1f}"
```

- [ ] **Step 2: Run test suite**

```bash
cd astar-island/cloud
python test_full_pipeline.py
```

Expected: avg > 80 across all seeds.

- [ ] **Step 3: Commit**
