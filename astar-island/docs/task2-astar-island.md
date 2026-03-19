# Astar Island — Viking Civilisation Prediction

## What is this?

Astar Island is a machine learning challenge where you observe a black-box Norse civilisation simulator through a limited viewport and predict the final world state. The simulator runs a procedurally generated Norse world for 50 years — settlements grow, factions clash, trade routes form, alliances shift, forests reclaim ruins, and harsh winters reshape entire civilisations.

Your goal: observe, learn the world's hidden rules, and predict the probability distribution of terrain types across the entire map.

- **Task type:** Observation + probabilistic prediction
- **Platform:** app.ainm.no
- **API:** REST endpoints at api.ainm.no/astar-island/

## How It Works

1. **A round starts** — the admin creates a round with a fixed map, many hidden parameters, and 5 random seeds
2. **Observe through a viewport** — call `POST /astar-island/simulate` with viewport coordinates to observe one stochastic run through a window (max 15×15 cells). You have 50 queries total per round, shared across all 5 seeds.
3. **Learn the hidden rules** — analyze viewport observations to understand the forces that govern the world
4. **Generate predictions** — use your understanding to build probability distributions for the full map
5. **Submit predictions** — for each of the 5 seeds, submit a W×H×6 probability tensor predicting terrain type probabilities per cell
6. **Scoring** — your prediction is compared against the ground truth using entropy-weighted KL divergence

## The Core Challenge

The simulation is stochastic — the same map and parameters produce different outcomes every run. With only 50 queries shared across 5 seeds, and each query only revealing a 15×15 viewport of the 40×40 map, you must be strategic about what you observe and how you use that information.

The world is governed by many hidden forces that interact in complex ways. Teams that understand these interactions can build accurate models and generate predictions far beyond what raw observation provides.

## Quick Start

1. Sign in at app.ainm.no with Google
2. Create or join a team
3. Go to the Astar Island page
4. When a round is active, use the API to observe the simulator
5. Analyze results, build your model, submit predictions for all 5 seeds

## Key Concepts

| Concept | Description |
|---|---|
| Map seed | Determines terrain layout (fixed per seed, visible to you) |
| Sim seed | Random seed for each simulation run (different every query) |
| Hidden parameters | Values controlling the world's behavior (same for all seeds in a round) |
| 50 queries | Your budget per round, shared across all 5 seeds |
| Viewport | Each query reveals a max 15×15 window of the map |
| W×H×6 tensor | Your prediction — probability of each of 6 terrain classes per cell |
| 50 years | Each simulation runs for 50 time steps |

## Simulation Mechanics

### The World

The world is a rectangular grid (default 40×40) with 8 terrain types that map to 6 prediction classes:

| Internal Code | Terrain | Class Index | Description |
|---|---|---|---|
| 10 | Ocean | 0 (Empty) | Impassable water, borders the map |
| 11 | Plains | 0 (Empty) | Flat land, buildable |
| 0 | Empty | 0 | Generic empty cell |
| 1 | Settlement | 1 | Active Norse settlement |
| 2 | Port | 2 | Coastal settlement with harbour |
| 3 | Ruin | 3 | Collapsed settlement |
| 4 | Forest | 4 | Provides food to adjacent settlements |
| 5 | Mountain | 5 | Impassable terrain |

Ocean, Plains, and Empty all map to class 0 in predictions. Mountains are static (never change). Forests are mostly static but can reclaim ruined land. The interesting cells are those that can become Settlements, Ports, or Ruins.

### Map Generation

Each map is procedurally generated from a map seed:

- Ocean borders surround the map
- Fjords cut inland from random edges
- Mountain chains form via random walks
- Forest patches cover land with clustered groves
- Initial settlements placed on land cells, spaced apart

The map seed is visible to you — you can reconstruct the initial terrain layout locally.

### Simulation Lifecycle

Each of the 50 years cycles through multiple phases. The world goes through growth, conflict, trade, harsh winters, and environmental change — in that order.

#### Growth

Settlements produce food based on adjacent terrain. When conditions are right, settlements grow in population, develop ports along coastlines, and build longships for naval operations. Prosperous settlements expand by founding new settlements on nearby land.

#### Conflict

Settlements raid each other. Longships extend raiding range significantly. Desperate settlements (low food) raid more aggressively. Successful raids loot resources and damage the defender. Sometimes, conquered settlements change allegiance to the raiding faction.

#### Trade

Ports within range of each other can trade if not at war. Trade generates wealth and food for both parties, and technology diffuses between trading partners.

#### Winter

Each year ends with a winter of varying severity. All settlements lose food. Settlements can collapse from starvation, sustained raids, or harsh winters — becoming Ruins and dispersing population to nearby friendly settlements.

#### Environment

The natural world slowly reclaims abandoned land. Nearby thriving settlements may reclaim and rebuild ruined sites, establishing new outposts that inherit a portion of their patron's resources and knowledge. Coastal ruins can even be restored as ports. If no settlement steps in, ruins are eventually overtaken by forest growth or fade back into open plains.

### Settlement Properties

Each settlement tracks: position, population, food, wealth, defense, tech level, port status, longship ownership, and faction allegiance (owner_id).

Initial states expose settlement positions and port status. Internal stats (population, food, wealth, defense) are only visible through simulation queries.

## API Endpoint Specification

### Base URL

```
https://api.ainm.no/astar-island
```

All endpoints require authentication. The API accepts either:

- **Cookie:** `access_token` JWT cookie (set automatically when you log in at app.ainm.no)
- **Bearer token:** `Authorization: Bearer <token>` header

### Endpoints Overview

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | /astar-island/rounds | Public | List all rounds |
| GET | /astar-island/rounds/{round_id} | Public | Round details + initial states |
| GET | /astar-island/budget | Team | Query budget for active round |
| POST | /astar-island/simulate | Team | Observe one simulation through viewport |
| POST | /astar-island/submit | Team | Submit prediction tensor |
| GET | /astar-island/my-rounds | Team | Rounds with your scores, rank, budget |
| GET | /astar-island/my-predictions/{round_id} | Team | Your predictions with argmax/confidence |
| GET | /astar-island/analysis/{round_id}/{seed_index} | Team | Post-round ground truth comparison |
| GET | /astar-island/leaderboard | Public | Astar Island leaderboard |

### GET /astar-island/rounds

List all rounds with status and timing.

```json
[
  {
    "id": "uuid",
    "round_number": 1,
    "event_date": "2026-03-19",
    "status": "active",
    "map_width": 40,
    "map_height": 40,
    "prediction_window_minutes": 165,
    "started_at": "2026-03-19T10:00:00Z",
    "closes_at": "2026-03-19T10:45:00Z",
    "round_weight": 1,
    "created_at": "2026-03-19T09:00:00Z"
  }
]
```

#### Round Status

| Status | Meaning |
|---|---|
| pending | Round created but not yet started |
| active | Queries and submissions open |
| scoring | Submissions closed, scoring in progress |
| completed | Scores finalized |

### GET /astar-island/rounds/{round_id}

Returns round details including initial map states for all seeds.

```json
{
  "id": "uuid",
  "round_number": 1,
  "status": "active",
  "map_width": 40,
  "map_height": 40,
  "seeds_count": 5,
  "initial_states": [
    {
      "grid": [[10, 10, 10, ...], ...],
      "settlements": [
        {
          "x": 5, "y": 12,
          "has_port": true,
          "alive": true
        }
      ]
    }
  ]
}
```

#### Grid Cell Values

| Value | Terrain |
|---|---|
| 0 | Empty |
| 1 | Settlement |
| 2 | Port |
| 3 | Ruin |
| 4 | Forest |
| 5 | Mountain |
| 10 | Ocean |
| 11 | Plains |

### GET /astar-island/budget

Check your team's remaining query budget for the active round.

```json
{
  "round_id": "uuid",
  "queries_used": 23,
  "queries_max": 50,
  "active": true
}
```

### Rate Limits

| Endpoint | Limit |
|---|---|
| POST /simulate | 5 requests/second per team |
| POST /submit | 2 requests/second per team |

### POST /astar-island/simulate

Core observation endpoint. Each call runs one stochastic simulation and reveals a viewport window. Costs one query from your budget (50 per round).

#### Request

```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,
  "viewport_h": 15
}
```

| Field | Type | Description |
|---|---|---|
| round_id | string | UUID of the active round |
| seed_index | int (0–4) | Which of the 5 seeds to simulate |
| viewport_x | int (>=0) | Left edge of viewport (default 0) |
| viewport_y | int (>=0) | Top edge of viewport (default 0) |
| viewport_w | int (5–15) | Viewport width (default 15) |
| viewport_h | int (5–15) | Viewport height (default 15) |

#### Response

```json
{
  "grid": [[4, 11, 1, ...], ...],
  "settlements": [
    {
      "x": 12, "y": 7,
      "population": 2.8,
      "food": 0.4,
      "wealth": 0.7,
      "defense": 0.6,
      "has_port": true,
      "alive": true,
      "owner_id": 3
    }
  ],
  "viewport": {"x": 10, "y": 5, "w": 15, "h": 15},
  "width": 40,
  "height": 40,
  "queries_used": 24,
  "queries_max": 50
}
```

Each call uses a different random sim_seed, so you get a different stochastic outcome.

#### Error Codes

| Status | Meaning |
|---|---|
| 400 | Round not active, or invalid seed_index |
| 403 | Not on a team |
| 404 | Round not found |
| 429 | Query budget exhausted (50/50) or rate limit exceeded (max 5 req/sec) |

### POST /astar-island/submit

Submit your prediction for one seed. You must submit all 5 seeds for a complete score.

#### Request

```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "prediction": [
    [
      [0.85, 0.05, 0.02, 0.03, 0.03, 0.02],
      [0.10, 0.40, 0.30, 0.10, 0.05, 0.05],
      ...
    ],
    ...
  ]
}
```

#### Prediction Format

The prediction is a 3D array: `prediction[y][x][class]`

- Outer dimension: H rows (height)
- Middle dimension: W columns (width)
- Inner dimension: 6 probabilities (one per class)
- Each cell's 6 probabilities must sum to 1.0 (±0.01 tolerance)
- All probabilities must be non-negative

#### Class Indices

| Index | Class |
|---|---|
| 0 | Empty (Ocean, Plains, Empty) |
| 1 | Settlement |
| 2 | Port |
| 3 | Ruin |
| 4 | Forest |
| 5 | Mountain |

#### Response

```json
{
  "status": "accepted",
  "round_id": "uuid",
  "seed_index": 3
}
```

Resubmitting for the same seed overwrites your previous prediction. Only the last submission counts.

### GET /astar-island/my-rounds

Returns all rounds enriched with your team's scores, submission counts, rank, and query budget.

```json
[
  {
    "id": "uuid",
    "round_number": 1,
    "event_date": "2026-03-19",
    "status": "completed",
    "map_width": 40,
    "map_height": 40,
    "seeds_count": 5,
    "round_weight": 1,
    "round_score": 72.5,
    "seed_scores": [80.1, 65.3, 71.9, ...],
    "seeds_submitted": 5,
    "rank": 3,
    "total_teams": 12,
    "queries_used": 48,
    "queries_max": 50
  }
]
```

### GET /astar-island/my-predictions/{round_id}

Returns your submitted predictions with argmax and confidence grids.

```json
[
  {
    "seed_index": 0,
    "argmax_grid": [[0, 4, 5, ...], ...],
    "confidence_grid": [[0.85, 0.72, 0.93, ...], ...],
    "score": 78.2,
    "submitted_at": "2026-03-19T10:30:00+00:00"
  }
]
```

### GET /astar-island/analysis/{round_id}/{seed_index}

Post-round analysis. Returns your prediction alongside ground truth for comparison. Only available after round completion.

```json
{
  "prediction": [[[0.85, 0.05, 0.02, 0.03, 0.03, 0.02], ...], ...],
  "ground_truth": [[[0.90, 0.03, 0.01, 0.02, 0.02, 0.02], ...], ...],
  "score": 78.2,
  "width": 40,
  "height": 40,
  "initial_grid": [[10, 10, 10, ...], ...]
}
```

### GET /astar-island/leaderboard

```json
[
  {
    "team_id": "uuid",
    "team_name": "Vikings ML",
    "team_slug": "vikings-ml",
    "weighted_score": 72.5,
    "rounds_participated": 3,
    "hot_streak_score": 78.1,
    "rank": 1,
    "is_verified": true
  }
]
```

## Scoring

### Score Formula

Your score is based on entropy-weighted KL divergence between your prediction and the ground truth.

### Ground Truth

For each seed, the organizers pre-compute ground truth by running the simulation hundreds of times with the true hidden parameters. This produces a probability distribution for each cell.

For example, a cell might have ground truth `[0.0, 0.60, 0.25, 0.15, 0.0, 0.0]` — meaning 60% chance of Settlement, 25% Port, 15% Ruin, after 50 years.

### KL Divergence

For each cell:

```
KL(p || q) = Σ pᵢ × log(pᵢ / qᵢ)
```

Where p = ground truth, q = your prediction. Lower KL = better match.

### Entropy Weighting

Only dynamic cells (those that change between simulation runs) contribute to your score, weighted by their entropy:

```
entropy(cell) = -Σ pᵢ × log(pᵢ)
```

Cells with higher entropy count more toward your score.

### Final Score

```
weighted_kl = Σ entropy(cell) × KL(ground_truth[cell], prediction[cell])
              ─────────────────────────────────────────────────────────
                            Σ entropy(cell)

score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

- 100 = perfect prediction
- 0 = terrible prediction

### Common Pitfalls

**Never assign probability 0.0 to any class.** If the ground truth has pᵢ > 0 but your prediction has qᵢ = 0, KL divergence goes to infinity.

Always enforce a minimum probability floor:

```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

### Per-Round Score

```
round_score = (score_seed_0 + score_seed_1 + ... + score_seed_4) / 5
```

Always submit something for every seed — even a uniform prediction beats 0.

### Leaderboard

Your leaderboard score is your best round score of all time:

```
leaderboard_score = max(round_score) across all rounds
```

## Quickstart Code

### Authentication

```python
import requests

BASE = "https://api.ainm.no"

session = requests.Session()
session.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"
```

### Step 1: Get the Active Round

```python
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)

if active:
    round_id = active["id"]
    print(f"Active round: {active['round_number']}")
```

### Step 2: Get Round Details

```python
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

width = detail["map_width"]      # 40
height = detail["map_height"]    # 40
seeds = detail["seeds_count"]    # 5

for i, state in enumerate(detail["initial_states"]):
    grid = state["grid"]
    settlements = state["settlements"]
    print(f"Seed {i}: {len(settlements)} settlements")
```

### Step 3: Query the Simulator

```python
result = session.post(f"{BASE}/astar-island/simulate", json={
    "round_id": round_id,
    "seed_index": 0,
    "viewport_x": 10,
    "viewport_y": 5,
    "viewport_w": 15,
    "viewport_h": 15,
}).json()

grid = result["grid"]
settlements = result["settlements"]
```

### Step 4: Build and Submit Predictions

```python
import numpy as np

for seed_idx in range(seeds):
    prediction = np.full((height, width, 6), 1/6)  # uniform baseline

    # TODO: replace with your model's predictions

    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    })
    print(f"Seed {seed_idx}: {resp.status_code}")
```

## MCP Setup

```
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```
