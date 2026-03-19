#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
DRY_RUN=false
NO_MC=false
RESUME=false
MC_RUNS=100
MONITOR_ONLY=false
POLL=false
EXTRA_ARGS=()

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
Usage: run_round.sh [OPTIONS]

Run the Astar Island agent and optionally monitor scores.

Options:
  --dry-run        Run agent without submitting predictions
  --no-mc          Skip Monte Carlo simulation
  --resume         Resume from saved observations.json
  --mc-runs N      Monte Carlo runs per parameter sample (default: 200)
  --monitor        Only check scores (skip agent)
  --poll           Poll scores every 60s after submission
  --help           Show this help

Token is read from ASTAR_ISLAND_TOKEN env var or .env file.
USAGE
    exit 0
}

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --no-mc)      NO_MC=true; shift ;;
        --resume)     RESUME=true; shift ;;
        --mc-runs)    MC_RUNS="$2"; shift 2 ;;
        --monitor)    MONITOR_ONLY=true; shift ;;
        --poll)       POLL=true; shift ;;
        --help|-h)    usage ;;
        *)            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# ── Load token ────────────────────────────────────────────────────────────────
if [[ -z "${ASTAR_ISLAND_TOKEN:-}" ]]; then
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        # Source .env, handling KEY=VALUE format
        while IFS='=' read -r key value; do
            [[ -z "$key" || "$key" == \#* ]] && continue
            value="${value%\"}"
            value="${value#\"}"
            export "$key=$value"
        done < "$SCRIPT_DIR/.env"
    fi
fi

TOKEN="${ASTAR_ISLAND_TOKEN:-}"
if [[ -z "$TOKEN" ]]; then
    echo "ERROR: No token found."
    echo "Set ASTAR_ISLAND_TOKEN env var or create .env with:"
    echo "  ASTAR_ISLAND_TOKEN=your_jwt_token_here"
    exit 1
fi

echo "=== Astar Island Round Runner ==="
echo "Token: ${TOKEN:0:20}..."

# ── Monitor-only mode ─────────────────────────────────────────────────────────
if $MONITOR_ONLY; then
    echo ""
    echo "=== Checking scores ==="
    MONITOR_ARGS=(--token "$TOKEN")
    $POLL && MONITOR_ARGS+=(--poll)
    python3 "$SCRIPT_DIR/monitor.py" "${MONITOR_ARGS[@]}"
    exit 0
fi

# ── Run agent ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Running agent ==="
AGENT_ARGS=(--token "$TOKEN" --mc-runs "$MC_RUNS")
$DRY_RUN && AGENT_ARGS+=(--dry-run)
$NO_MC && AGENT_ARGS+=(--no-mc)
$RESUME && AGENT_ARGS+=(--resume)
AGENT_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

python3 "$SCRIPT_DIR/agent.py" "${AGENT_ARGS[@]}"

# ── Post-submission monitoring ────────────────────────────────────────────────
echo ""
echo "=== Checking scores ==="
MONITOR_ARGS=(--token "$TOKEN")
$POLL && MONITOR_ARGS+=(--poll)
python3 "$SCRIPT_DIR/monitor.py" "${MONITOR_ARGS[@]}"
