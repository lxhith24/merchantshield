#!/usr/bin/env bash
# Cloud Run entrypoint: API stays internal on 127.0.0.1:8000; only the
# Streamlit UI is bound to the public $PORT, matching how run.sh wires the
# two processes together for local development.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f merchantshield.db ]; then
  echo "==> Seeding ring cases"
  python scripts/seed_cases.py
fi

python -m uvicorn merchantshield.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

echo "==> Waiting for the API"
for _ in $(seq 1 60); do
  if curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1; then break; fi
  kill -0 $API_PID 2>/dev/null || { echo "API failed to start"; exit 1; }
  sleep 0.5
done
curl -fsS "http://127.0.0.1:8000/health" >/dev/null 2>&1 || {
  echo "API did not become healthy"
  exit 1
}

echo "==> API healthy, starting Streamlit on port ${PORT:-7860}"
exec env MERCHANTSHIELD_API_URL="http://127.0.0.1:8000" \
  streamlit run merchantshield/ui/streamlit_app.py \
    --server.port "${PORT:-7860}" --server.address 0.0.0.0 \
    --server.headless true --browser.gatherUsageStats false
