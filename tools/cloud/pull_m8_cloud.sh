#!/usr/bin/env bash
# Mirror a running cloud training run down to the local machine so the monitor
# (which only ever READS train/runs/<id>/events.jsonl) can follow it, and so the
# checkpoints are safe locally before the box is destroyed.
#
# Bandwidth-conscious (Vast bills egress): events.jsonl and train.log are fetched
# INCREMENTALLY -- only the bytes appended since last cycle, via `tail -c`, not
# the whole growing file each time. Checkpoints are pulled once each (skipped if
# already present locally).
#
# Run it locally in Git Bash:  bash tools/cloud/pull_m8_cloud.sh
# Stop with Ctrl-C. Re-running resumes cleanly (offsets derive from local size).
#
# Box coords change every time you rent a new instance -- edit HOST/PORT here.
set -u

# --- box coordinates (DIRECT route; the proxy route is broken on this box) -----
HOST=180.189.55.43
PORT=21546
KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR=/workspace/gambit/train/runs/m8_cloud
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)/train/runs/m8_cloud"
INTERVAL="${1:-300}"   # seconds between cycles (default 5 min); pass an arg to override

SSH=(ssh -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)
SCP=(scp -P "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30)

mkdir -p "$LOCAL_DIR"
echo "Mirroring $HOST:$REMOTE_DIR -> $LOCAL_DIR every ${INTERVAL}s (Ctrl-C to stop)"

# Append only the remote bytes beyond what we already hold locally.
fetch_incremental() {
  local name="$1"
  local local_path="$LOCAL_DIR/$name"
  local have=0
  [ -f "$local_path" ] && have=$(stat -c%s "$local_path" 2>/dev/null || echo 0)
  # tail -c +N is 1-indexed; +have+1 = first unseen byte. Empty if nothing new.
  "${SSH[@]}" "root@$HOST" "tail -c +$((have + 1)) '$REMOTE_DIR/$name' 2>/dev/null" >> "$local_path"
}

while true; do
  ts=$(date +%H:%M:%S)

  fetch_incremental events.jsonl
  fetch_incremental train.log

  # Checkpoints: pull any ckpt_*.pt we don't already have.
  remote_ckpts=$("${SSH[@]}" "root@$HOST" "cd '$REMOTE_DIR' 2>/dev/null && ls ckpt_*.pt 2>/dev/null" | tr -d '\r')
  new=0
  for f in $remote_ckpts; do
    if [ ! -f "$LOCAL_DIR/$f" ]; then
      "${SCP[@]}" "root@$HOST:$REMOTE_DIR/$f" "$LOCAL_DIR/$f" >/dev/null 2>&1 && new=$((new + 1))
    fi
  done

  ev=0; [ -f "$LOCAL_DIR/events.jsonl" ] && ev=$(stat -c%s "$LOCAL_DIR/events.jsonl" 2>/dev/null || echo 0)
  echo "[$ts] events.jsonl=${ev}B  +${new} new ckpt(s)"
  sleep "$INTERVAL"
done
