#!/usr/bin/env bash
# Continuously mirrors the cloud run's checkpoints/logs back to this PC so a
# host failure never costs more than a few minutes of training.
# Runs until killed; safe to Ctrl-C and restart anytime.
#
# Usage: tools/cloud/pull_checkpoints.sh <ssh-host> <ssh-port> [interval-sec] [run-name]
set -euo pipefail

HOST="${1:?usage: pull_checkpoints.sh <ssh-host> <ssh-port> [interval] [run-name]}"
PORT="${2:?usage: pull_checkpoints.sh <ssh-host> <ssh-port> [interval] [run-name]}"
INTERVAL="${3:-300}"
RUN="${4:-cloud}"
REPO_LOCAL="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
DEST="$REPO_LOCAL/train/runs/cloud_mirror/$RUN"
export RSYNC_RSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"

mkdir -p "$DEST"
echo "mirroring root@$HOST:/workspace/balatroagent/train/runs/$RUN/ -> $DEST every ${INTERVAL}s"
while true; do
  rsync -az \
    "root@$HOST:/workspace/balatroagent/train/runs/$RUN/" "$DEST/" \
    && date +"[%F %T] synced ok" \
    || echo "[warn] sync failed; retrying in ${INTERVAL}s"
  sleep "$INTERVAL"
done
