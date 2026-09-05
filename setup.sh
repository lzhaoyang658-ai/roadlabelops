#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$PROJECT_DIR"

if [ ! -f .env ]; then
  install -m 600 .env.example .env
else
  chmod 600 .env
fi

ROADLABELOPS_PYTHON_BIN="${ROADLABELOPS_PYTHON_BIN:-python3.11}"

if ! command -v uv >/dev/null 2>&1; then
  echo "RoadLabelOps requires uv. Install it from https://docs.astral.sh/uv/."
  exit 1
fi

if ! command -v "$ROADLABELOPS_PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11 is required for the verified development environment."
  exit 1
fi

if ! "$ROADLABELOPS_PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'; then
  echo "ROADLABELOPS_PYTHON_BIN must point to Python 3.11."
  exit 1
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Node.js >=20.9 and npm are required for the frontend."
  exit 1
fi

if ! node -e 'const [major, minor] = process.versions.node.split(".").map(Number); process.exit(major > 20 || (major === 20 && minor >= 9) ? 0 : 1)'; then
  echo "Node.js >=20.9 is required; found $(node --version)."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "FFmpeg is missing. On macOS, run: brew install ffmpeg"
  exit 1
fi

uv sync --frozen --python "$ROADLABELOPS_PYTHON_BIN" --extra dev --extra detection
npm --prefix frontend ci

mkdir -p "$PROJECT_DIR/data" "$PROJECT_DIR/runtime"

echo "Setup complete. Provision a local YOLO .pt file, configure CVAT, then run:"
echo "  .venv/bin/roadlabelops doctor"
