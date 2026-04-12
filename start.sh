#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODE="${1:-backend}"

usage() {
  echo "Usage: $0 [backend|frontend|all]" >&2
  echo "  backend   - FastAPI on http://0.0.0.0:8000 (default)" >&2
  echo "  frontend  - Next.js dev on http://localhost:3000" >&2
  echo "  all       - backend and frontend in this shell (Ctrl+C stops both)" >&2
  exit 1
}

case "$MODE" in
  backend|frontend|all) ;;
  -h|--help|help) usage ;;
  *) usage ;;
esac

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH." >&2
  exit 1
fi

run_backend() {
  echo "Starting AceClaw backend (repo root: $ROOT)..."
  exec python3 -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
}

run_frontend() {
  if ! command -v npm >/dev/null 2>&1; then
    echo "ERROR: npm not found in PATH." >&2
    exit 1
  fi
  echo "Starting AceClaw frontend..."
  cd "$ROOT/frontend"
  exec npm run dev
}

if [[ "$MODE" == "backend" ]]; then
  run_backend
fi

if [[ "$MODE" == "frontend" ]]; then
  run_frontend
fi

# all: run both; stop both on Ctrl+C
if [[ "$MODE" == "all" ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "ERROR: npm not found in PATH." >&2
    exit 1
  fi
  python3 -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000 &
  backend_pid=$!
  (cd "$ROOT/frontend" && npm run dev) &
  frontend_pid=$!
  trap 'kill "$backend_pid" "$frontend_pid" 2>/dev/null || true' EXIT INT TERM
  echo "Backend: http://127.0.0.1:8000  Frontend: http://localhost:3000"
  echo "Press Ctrl+C to stop both."
  wait
fi
