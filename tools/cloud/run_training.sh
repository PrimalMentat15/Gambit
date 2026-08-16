#!/usr/bin/env bash
# Supervisor entrypoint for cloud training: always resumes from the newest
# checkpoint so crash/instance-restart recovery never loses more than one
# save interval. Config via env: TRAIN_CONFIG (default configs/m4.yaml),
# RUN_NAME (default m4).
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

cd /workspace/balatroagent/train
CONFIG="${TRAIN_CONFIG:-configs/m4.yaml}"
RUN="${RUN_NAME:-m4}"
mkdir -p "runs/$RUN"

# Newest checkpoint from THIS run only (multiple runs share the box since the
# 2x5090 A/B — cross-run pickup would let one run resume the other's weights),
# falling back to the shipped resume.pt on first launch.
CKPT="$(ls -t "runs/$RUN"/ckpt_*.pt runs/cloud/resume.pt 2>/dev/null | head -1 || true)"
RESUME_ARGS=()
[ -n "$CKPT" ] && RESUME_ARGS=(--resume "$CKPT")

echo "[run_training] config=$CONFIG resume=${CKPT:-none}"
echo $$ > "runs/$RUN/pid"
exec uv run python -m balatro_train.ppo --config "$CONFIG" "${RESUME_ARGS[@]}"
