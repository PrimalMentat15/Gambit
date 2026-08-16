#!/usr/bin/env bash
# Deploys the repo + latest checkpoint + snapshot pool to a Vast.ai instance
# and starts training there. Run from anywhere inside the repo.
#
# Usage: tools/cloud/deploy.sh <ssh-host> <ssh-port>
#   e.g. tools/cloud/deploy.sh ssh4.vast.ai 12345
#        (host/port come from the instance's Connect button in the Vast console)
set -euo pipefail

HOST="${1:?usage: deploy.sh <ssh-host> <ssh-port>}"
PORT="${2:?usage: deploy.sh <ssh-host> <ssh-port>}"
REPO_LOCAL="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
REMOTE=/workspace/balatroagent
export RSYNC_RSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"

echo "== rsync repo -> root@$HOST:$REMOTE =="
rsync -az --delete \
  --exclude 'target/' --exclude '.venv/' --exclude '__pycache__/' \
  --exclude 'runs/' --exclude 'reference/' --exclude 'tools/oracle*' \
  --exclude 'tools/crossval/reports/' \
  "$REPO_LOCAL/" "root@$HOST:$REMOTE/"

echo "== ship snapshot pool =="
ssh -p "$PORT" "root@$HOST" "mkdir -p $REMOTE/train/runs/pools $REMOTE/train/runs/cloud"
rsync -az "$REPO_LOCAL/train/runs/pools/pool_v1.bin" \
  "root@$HOST:$REMOTE/train/runs/pools/pool_v1.bin"

echo "== ship latest checkpoint =="
CKPT="$(ls -t "$REPO_LOCAL"/train/runs/m3_fast/ckpt_*.pt 2>/dev/null | head -1 || true)"
[ -z "$CKPT" ] && CKPT="$(ls -t "$REPO_LOCAL"/train/runs/m3/ckpt_*.pt | head -1)"
echo "   $CKPT"
rsync -az "$CKPT" "root@$HOST:$REMOTE/train/runs/cloud/resume.pt"

echo "== remote setup + launch (first run: ~5-10 min incl. torch download) =="
ssh -p "$PORT" "root@$HOST" "bash $REMOTE/tools/cloud/setup_remote.sh"

echo
echo "Deployed. Monitor with:"
echo "  ssh -p $PORT root@$HOST tail -f $REMOTE/train/runs/cloud/train.log"
echo "Pull checkpoints continuously with:"
echo "  tools/cloud/pull_checkpoints.sh $HOST $PORT"
