# Balatro RL - Windows Setup Walkthrough

## Status

| Step | Status | Details |
|------|--------|---------|
| Python venv | Done | Created at `venv/` (Python 3.12.4) |
| pip install requirements.txt | Done | All deps installed (torch, gymnasium, sb3, tensorboard, etc.) |
| pip install sb3-contrib | Done | Was missing from requirements.txt - added and installed |
| RLBridge mod symlink | Done | Junction: `%AppData%\Balatro\Mods\RLBridge` -> repo `RLBridge/` |
| Training directories | Done | Created `models/` and `tensorboard_logs/` |
| Lovely Injector | Done | `version.dll` in `C:\Games\Balatro.v1.0.1N\game\`, console window confirms it loads |
| Named pipes -> sockets | Done | Both sides ported to localhost TCP, verified with a round-trip test |
| CUDA PyTorch | Done | `torch` cu126 build installed, training defaults to the RTX 4050 |

---

## Communication: Localhost TCP Socket

The original project used Unix named pipes (`os.mkfifo`, `/tmp/balatro_*`), which do not exist on
Windows. Both sides now speak line-delimited JSON over a single localhost TCP connection.

| | |
|---|---|
| Address | `127.0.0.1:12345` |
| Server | Python — [ai/utils/communication.py](ai/utils/communication.py) (`BalatroSocketIO`) |
| Client | Lua mod — [RLBridge/communication.lua](RLBridge/communication.lua) |
| Framing | One JSON object per line, `\n` delimited |
| Latency | `TCP_NODELAY` on both ends, so no Nagle delay on the request/response ping-pong |
| Override | `BALATRO_RL_HOST` / `BALATRO_RL_PORT` env vars, read by both sides |

**Why sockets over Windows Named Pipes:** no new dependencies. Python uses stdlib `socket`; the Lua
side uses LuaSocket, which is already compiled into Balatro's `love.dll` (`require("socket")`).
Windows Named Pipes would have needed `pywin32` plus hand-rolled message framing, and a
shared-file-plus-polling scheme would have added latency to every single action. Sockets also keep
the code identical on Linux and macOS.

**Behaviour notes:**
- Python binds and blocks on `accept()` at startup — this is the "Waiting for Balatro to connect" message.
- The mod connects lazily, on its first action request (i.e. after you press `R`).
- If the connection drops (game closed/relaunched), the mod reconnects and Python re-accepts automatically.
- If the trainer is not running, the mod retries at most once per second so the game does not stall.

### Order of operations

1. Start the trainer first — it must own the listening port before the game connects.
2. Launch Balatro with Lovely.
3. Press `R` in Balatro.

---

## GPU

Installed `torch 2.13.0+cu126` / `torchvision 0.28.0+cu126`. Verified: `torch.cuda.is_available()` is
`True`, device reports as `NVIDIA GeForce RTX 4050 Laptop GPU` (sm_89), and a 4096x4096 matmul runs on
the GPU.

> [!WARNING]
> A plain `pip install -r ai/requirements.txt` will pull the CPU build back in and silently undo this.
> Reinstall with `--index-url https://download.pytorch.org/whl/cu126` if that happens.

Re-check any time with:

```powershell
.\venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`get_device()` in [ai/train_balatro.py](ai/train_balatro.py) picks CUDA when available and is passed
explicitly to `MaskablePPO`. Force CPU with `BALATRO_RL_DEVICE=cpu`.

> [!NOTE]
> Expect little speedup. The policy is a small `MlpPolicy` on a 216-dim observation, and the real
> bottleneck is the game itself — one environment, one blocking round-trip per action. SB3 will print
> a warning that PPO with `MlpPolicy` is usually faster on CPU; that warning is accurate here. The GPU
> matters only if the network is scaled up substantially or environments are parallelised.

---

## Running

```powershell
.\venv\Scripts\Activate.ps1
python -m ai.train_balatro
```

Then launch Balatro and press `R`.

Useful flags: `--no-prompt` (unattended), `--run-name <label>`, `--timesteps N`,
`--device cpu`, `--no-resume`.

---

## Telemetry (Phase 1, Stage 1)

Every run now writes a self-contained directory:

```
runs/2026-08-12_1543_<name>/
  meta.json      # git sha, hyperparams, device, PIDs, active flag
  events.jsonl   # durable event stream (source of truth)
  tb/            # per-run TensorBoard logs
  monitor.csv    # SB3 Monitor output
  training.log
  checkpoints/
```

After (or during) a run, see where the per-step time actually goes:

```bash
venv/Scripts/python.exe -m ai.tools.latency_report
```

With no argument it reads the latest run. It breaks each step into action mapping,
socket write, waiting on the game, observation build and reward calculation, then
uses the mod's frame counters to attribute the wait to *animations/blocking events*
versus *the game sitting idle while the state hash is unchanged* — two causes with
completely different fixes.

**Overhead:** `emit()` costs ~1.3 µs and total instrumentation is ~0.25 ms/step,
about 0.05% of the 483 ms budget. The emitter uses a bounded queue and drops events
rather than ever blocking the training loop.

Run the tests (no game needed) with `venv/Scripts/python.exe tests/test_e2e.py` —
see [tests/README.md](tests/README.md).

---

## Monitor UI (Phase 1, Stage 2)

```powershell
.\venv\Scripts\pip.exe install -r requirements-monitor.txt
.\venv\Scripts\python.exe -m monitor
```

Reads `runs/<id>/events.jsonl` only, so it never interferes with a training run —
start it, close it, restart it mid-session freely. It follows the newest run by
default; untick **Follow latest** to pin one.

> [!IMPORTANT]
> **PySide6 is pinned to 6.8.3 on this machine.** 6.11.x fails at import with
> `DLL load failed while importing QtCore: The specified procedure could not be
> found`. The cause is the Anaconda Python in use: it loads its own older
> `msvcp140.dll` into the process at startup, and Qt 6.11's `Qt6Core.dll` needs
> exports that version does not have. Clearing `PATH` does not help, because the
> conflicting DLL is already loaded in-process. Installing the current Microsoft
> Visual C++ Redistributable would also fix it if you want a newer PySide6.

Panels are dockable — drag, resize, stack or tear off, then **Save layout** to keep
the arrangement. Adding a chart is adding one file to `monitor/panels/`; the
registry finds it automatically.

---

## Project Structure Summary

```
balatro-rl/
+-- ai/                          # Python RL training
|   +-- requirements.txt         # Dependencies
|   +-- train_balatro.py         # Main training entry point
|   +-- environment/
|   |   +-- balatro_env.py       # Gymnasium environment
|   |   +-- reward.py            # Reward function
|   +-- utils/
|       +-- communication.py     # TCP socket server
|       +-- mappers.py           # State mapping
|       +-- replay.py            # Replay storage
|       +-- validation.py        # Action validation
+-- RLBridge/                    # Lua mod (symlinked to AppData)
|   +-- lovely/                  # Lovely patch definitions
|   |   +-- init.toml            # Module loading + patches
|   |   +-- ai.toml              # Update loop hook
|   +-- communication.lua        # TCP socket client
|   +-- ai.lua                   # Main AI loop
|   +-- actions.lua              # Game actions
+-- venv/                        # Python virtual environment
+-- models/                      # Training checkpoints
+-- tensorboard_logs/            # TensorBoard data
```

## Key Game Paths

| What | Path |
|------|------|
| Balatro exe | `C:\Games\Balatro.v1.0.1N\game\Balatro.exe` |
| Balatro AppData | `%AppData%\Balatro\` |
| Mods directory | `%AppData%\Balatro\Mods\` |
| RLBridge symlink | `%AppData%\Balatro\Mods\RLBridge` -> repo `RLBridge/` |

## Troubleshooting

| Symptom | Cause |
|---------|-------|
| `[COMM] ERROR: Cannot connect to AI at 127.0.0.1:12345` | Trainer not running, or started after the game and the port was never bound. Start Python first. |
| `[COMM] ERROR: LuaSocket unavailable` | `require("socket")` failed inside LÖVE. Check the Lovely console for the module load error. |
| Python hangs at "Waiting for Balatro to connect" | Expected until you press `R` in the game. |
| `OSError: [WinError 10048]` on bind | A previous trainer still holds port 12345. Kill it, or set `BALATRO_RL_PORT`. |
| Windows Firewall prompt on first run | Loopback-only; allow, or dismiss it — 127.0.0.1 traffic is not filtered. |
