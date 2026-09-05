#!/usr/bin/env bash
# MerchantShield — one-command start: API on :8000, reviewer UI on :8501.
#
#   ./run.sh              API + Streamlit reviewer surface
#   ./run.sh --api-only   API only (http://localhost:8000/docs)
set -euo pipefail
cd "$(dirname "$0")"

VENV=.venv
PY=$VENV/bin/python
PIP=$VENV/bin/pip
API_PORT=${MERCHANTSHIELD_API_PORT:-8000}
UI_PORT=${MERCHANTSHIELD_UI_PORT:-8501}
API_ONLY=0
[ "${1:-}" = "--api-only" ] && API_ONLY=1

if [ ! -x "$PY" ]; then
  echo "==> Creating virtual environment"
  python3 -m venv --clear "$VENV"
  $PIP install -q --upgrade pip
  $PIP install -q -r requirements.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env (no API key set: running in LLM_DISABLED mode)"
fi

# Seed on first run so the review queue is not empty.
if [ ! -f merchantshield.db ]; then
  echo "==> Seeding ring cases"
  $PY scripts/seed_cases.py
fi

if [ "$API_ONLY" = "1" ]; then
  echo "==> MerchantShield API on http://localhost:$API_PORT   (docs: /docs)"
  exec $VENV/bin/uvicorn merchantshield.main:app \
    --host 0.0.0.0 --port "$API_PORT" --reload
fi

$VENV/bin/uvicorn merchantshield.main:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

echo "==> Waiting for the API on :$API_PORT"
for _ in $(seq 1 40); do
  if curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1; then break; fi
  kill -0 $API_PID 2>/dev/null || { echo "API failed to start"; exit 1; }
  sleep 0.5
done
curl -fsS "http://127.0.0.1:$API_PORT/health" >/dev/null 2>&1 || {
  echo "API did not become healthy on port $API_PORT"
  exit 1
}

echo "==> API  http://localhost:$API_PORT   (docs: /docs)"
echo "==> UI   http://localhost:$UI_PORT"
# Deliberately not `exec`: the shell must survive to run its EXIT trap, so the
# API is stopped with the UI rather than left orphaned.
MERCHANTSHIELD_API_URL="http://127.0.0.1:$API_PORT" \
  $VENV/bin/streamlit run merchantshield/ui/streamlit_app.py \
    --server.port "$UI_PORT" --server.headless true --browser.gatherUsageStats false
