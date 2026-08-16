#!/usr/bin/env bash
# Bundles everything a new machine needs that is NOT in the public git repo:
# private oracle harnesses + extracted game Lua (copyrighted -> never via the
# public repo), snapshot pool, newest mirrored checkpoint, and Claude Code
# state (memory, plan file). Output: a single tar.gz to transfer DIRECTLY
# (scp/rsync/AirDrop) - do not upload to public/cloud storage.
set -euo pipefail
REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
OUT="${1:-$HOME/balatroagent-handoff-$(date +%Y%m%d).tar.gz}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE"/{private,claude-state,runs/pools,runs/latest-ckpt}

cp -r "$REPO/tools/oracle-harnesses" "$STAGE/private/" 2>/dev/null || echo "warn: no oracle-harnesses"
cp -r "$REPO/reference" "$STAGE/private/" 2>/dev/null || echo "warn: no reference/lua_source"
cp "$REPO/train/runs/pools/pool_v1.bin" "$STAGE/runs/pools/" 2>/dev/null || echo "warn: no pool"
CKPT="$(ls -t "$REPO"/train/runs/cloud_mirror/ckpt_*.pt 2>/dev/null | head -1 || true)"
[ -n "$CKPT" ] && cp "$CKPT" "$STAGE/runs/latest-ckpt/"

CLAUDE_PROJ="$HOME/.claude/projects/-home-teq-Desktop-balatroagent"
cp -r "$CLAUDE_PROJ/memory" "$STAGE/claude-state/" 2>/dev/null || echo "warn: no memory dir"
cp "$HOME/.claude/plans/i-want-to-build-eager-boole.md" "$STAGE/claude-state/plan.md" 2>/dev/null || true
# Session transcripts (optional reference; resuming cross-machine is best-effort)
mkdir -p "$STAGE/claude-state/transcripts"
cp "$CLAUDE_PROJ"/*.jsonl "$STAGE/claude-state/transcripts/" 2>/dev/null || true

cat > "$STAGE/README-HANDOFF.md" <<'EOF'
# Handoff bundle -> new machine

1. Clone the public repo: git clone https://github.com/jahankazimi078/balatroagent
2. private/oracle-harnesses -> <repo>/tools/oracle-harnesses  (gitignored)
   private/reference        -> <repo>/reference               (gitignored)
3. runs/pools/pool_v1.bin   -> <repo>/train/runs/pools/
   runs/latest-ckpt/*       -> <repo>/train/runs/cloud_mirror/ (optional; cloud has newer)
4. claude-state/memory      -> ~/.claude/projects/<encoded-repo-path>/memory/
   (encoded path = absolute repo path with / replaced by -; created on first
   claude session in the repo)
   claude-state/plan.md     -> ~/.claude/plans/ (optional reference)
   claude-state/transcripts -> same encoded dir (optional; lets `claude --resume`
   list old sessions, but they reference Linux paths - treat as read-only history)
5. New machine setup: rust (rustup), uv, then `cd sim && cargo test`,
   `cd train && uv sync --extra sim && uv run pytest`.
   Oracle LuaJIT: rebuild with tools/build_oracle_luajit.sh (binary is per-arch).
6. Cloud access: generate a NEW ssh key on the new machine (ssh-keygen -t ed25519),
   add the .pub to the Vast instance (console -> key icon) and to GitHub.
   Do NOT copy private keys between machines.
7. Restart the mirror + watcher there:
   tools/cloud/pull_checkpoints.sh <host> <port> 300 m4
EOF

tar -C "$STAGE" -czf "$OUT" .
echo "bundle: $OUT ($(du -h "$OUT" | cut -f1))"
