#!/bin/bash
# report.sh — Universal dashboard reporter for NM i AI 2026
#
# Usage:
#   ./report.sh test <task> <label> [status] [score] [details]
#   ./report.sh score <task> <raw_score> [rank] [total_teams] [note]
#   ./report.sh update <test_id> <status> [score] [details]
#
# All calls fail silently if the dashboard is unreachable.
# No dependencies beyond curl.

DASHBOARD="${DASHBOARD_URL:-http://localhost:8090}"

_curl_quiet() {
    curl -s -m 3 -o /dev/null -w "%{http_code}" "$@" 2>/dev/null || echo "000"
}

_curl_json() {
    curl -s -m 3 "$@" 2>/dev/null || true
}

case "${1:-}" in
    test)
        # test <task> <label> [status] [score] [details]
        TASK="${2:?missing task}"
        LABEL="${3:?missing label}"
        STATUS="${4:-running}"
        SCORE="${5:-}"
        DETAILS="${6:-}"

        BODY="{\"task\":\"${TASK}\",\"label\":\"${LABEL}\",\"status\":\"${STATUS}\""
        if [ -n "$SCORE" ]; then
            BODY="${BODY},\"score\":${SCORE}"
        fi
        if [ -n "$DETAILS" ]; then
            # Escape double quotes in details
            ESCAPED=$(echo "$DETAILS" | sed 's/"/\\"/g')
            BODY="${BODY},\"details\":\"${ESCAPED}\""
        fi
        BODY="${BODY}}"

        _curl_json -X POST "${DASHBOARD}/api/test" \
            -H 'Content-Type: application/json' \
            -d "$BODY"
        ;;

    score)
        # score <task> <raw_score> [rank] [total_teams] [note]
        TASK="${2:?missing task}"
        RAW="${3:?missing raw score}"
        RANK="${4:-}"
        TOTAL="${5:-}"
        NOTE="${6:-}"

        BODY="{\"task\":\"${TASK}\",\"raw\":${RAW}"
        if [ -n "$RANK" ]; then
            BODY="${BODY},\"rank\":${RANK}"
        fi
        if [ -n "$TOTAL" ]; then
            BODY="${BODY},\"total_teams\":${TOTAL}"
        fi
        if [ -n "$NOTE" ]; then
            ESCAPED=$(echo "$NOTE" | sed 's/"/\\"/g')
            BODY="${BODY},\"note\":\"${ESCAPED}\""
        fi
        BODY="${BODY}}"

        _curl_json -X POST "${DASHBOARD}/api/score" \
            -H 'Content-Type: application/json' \
            -d "$BODY"
        ;;

    update)
        # update <test_id> <status> [score] [details]
        TEST_ID="${2:?missing test_id}"
        STATUS="${3:?missing status}"
        SCORE="${4:-}"
        DETAILS="${5:-}"

        BODY="{\"status\":\"${STATUS}\""
        if [ -n "$SCORE" ]; then
            BODY="${BODY},\"score\":${SCORE}"
        fi
        if [ -n "$DETAILS" ]; then
            ESCAPED=$(echo "$DETAILS" | sed 's/"/\\"/g')
            BODY="${BODY},\"details\":\"${ESCAPED}\""
        fi
        BODY="${BODY}}"

        _curl_json -X PUT "${DASHBOARD}/api/test/${TEST_ID}" \
            -H 'Content-Type: application/json' \
            -d "$BODY"
        ;;

    *)
        echo "Usage: $0 {test|score|update} <args...>"
        echo ""
        echo "Commands:"
        echo "  test   <task> <label> [status] [score] [details]"
        echo "  score  <task> <raw_score> [rank] [total_teams] [note]"
        echo "  update <test_id> <status> [score] [details]"
        exit 1
        ;;
esac
