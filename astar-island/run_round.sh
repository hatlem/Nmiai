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

# Dashboard reporting helper
REPORT="$(cd "$(dirname "$0")/.." && pwd)/report.sh"

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

# Report start to dashboard
ASTAR_TEST_ID=$("$REPORT" test astar "Round run (mc=$MC_RUNS)" running 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('id',''))" 2>/dev/null || true)

python3 "$SCRIPT_DIR/agent.py" "${AGENT_ARGS[@]}"
AGENT_EXIT=$?

# ── Post-submission monitoring ────────────────────────────────────────────────
echo ""
echo "=== Checking scores ==="
MONITOR_ARGS=(--token "$TOKEN")
$POLL && MONITOR_ARGS+=(--poll)
python3 "$SCRIPT_DIR/monitor.py" "${MONITOR_ARGS[@]}"

# ── Report to dashboard ──
echo ""
echo "=== Reporting to dashboard ==="
# Try to extract best score from monitor output
SCORE=$(python3 "$SCRIPT_DIR/monitor.py" --token "$TOKEN" 2>/dev/null | grep -oP 'Score: \K[0-9.]+' | head -1)
if [[ -n "${SCORE:-}" ]]; then
    "$REPORT" score astar "$SCORE" "" "" "auto-reported from run_round.sh" 2>/dev/null || true
    if [[ -n "${ASTAR_TEST_ID:-}" ]]; then
        "$REPORT" update "$ASTAR_TEST_ID" passed "$SCORE" "Score: ${SCORE}" 2>/dev/null || true
    fi
    echo "Reported score: ${SCORE}"
elif [[ -n "${ASTAR_TEST_ID:-}" ]]; then
    if [[ "${AGENT_EXIT:-0}" -ne 0 ]]; then
        "$REPORT" update "$ASTAR_TEST_ID" failed "" "Agent exited with code ${AGENT_EXIT}" 2>/dev/null || true
    else
        "$REPORT" update "$ASTAR_TEST_ID" submitted "" "Submitted, score pending" 2>/dev/null || true
    fi
fi
