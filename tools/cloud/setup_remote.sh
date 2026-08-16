#!/usr/bin/env bash
# Runs ON the cloud instance (as root) after deploy.sh has rsynced the repo
# to /workspace/balatroagent. Installs toolchains, builds the sim, and
# starts training resuming from train/runs/cloud/resume.pt.
set -euo pipefail

REPO=/workspace/balatroagent
RESUME="$REPO/train/runs/cloud/resume.pt"
CONFIG=configs/cloud.yaml

echo "== apt deps =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential curl pkg-config rsync tmux >/dev/null

echo "== rust =="
if ! command -v cargo >/dev/null; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q
fi
# shellcheck disable=SC1091
source "$HOME/.cargo/env"

echo "== uv =="
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -q
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== gpu sanity =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== build + sync (torch download ~2.5GB on first run) =="
cd "$REPO/train"
uv sync --extra sim

uv run python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available in torch"
print("torch", torch.__version__, "| cuda ok:", torch.cuda.get_device_name(0))
EOF

test -f "$RESUME" || { echo "ERROR: missing $RESUME (deploy.sh ships it)"; exit 1; }
test -f runs/pools/pool_v1.bin || { echo "ERROR: missing snapshot pool"; exit 1; }

echo "== launching training =="
mkdir -p runs/cloud
nohup uv run python -m balatro_train.ppo --config "$CONFIG" --resume "$RESUME" \
  > runs/cloud/train.log 2>&1 &
echo $! > runs/cloud/pid
sleep 20
ps -p "$(cat runs/cloud/pid)" >/dev/null || { echo "ERROR: training died on start:"; tail -30 runs/cloud/train.log; exit 1; }
echo "== training launched (pid $(cat runs/cloud/pid)); first iterations: =="
sleep 100
tail -8 runs/cloud/train.log
