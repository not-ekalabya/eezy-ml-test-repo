#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SERVER_URL="${SERVER_URL:-http://localhost:5000}"
PROMPT="${1:-Tell me one short fact about model deployment.}"

echo "=== Sending one inference request ==="
echo "Server: $SERVER_URL"
echo "Prompt: $PROMPT"

curl -sS -X POST "$SERVER_URL/predict" \
  -H "Content-Type: application/json" \
  -d "$(printf '{\"features\": %s}' "$(printf '%s' "$PROMPT" | python -c 'import json,sys; print(json.dumps(sys.stdin.read()))')")"

echo ""