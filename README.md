# Gambit

A reinforcement learning agent that plays Balatro, combining a Lua mod for game
instrumentation with a Python training environment.

## Credits

Gambit builds on [balatro-rl](https://github.com/angelvalentin80/balatro-rl) by
**Angel Valentin**, which established the RLBridge mod, the Gymnasium environment
and the reward design. That project is MIT licensed and its copyright is retained
in [LICENSE](LICENSE) alongside this work's.

## Overview

The mod extracts game state from Balatro and executes actions chosen by the agent.
Communication is line-delimited JSON over a localhost TCP socket, so the same code
path works on Windows, Linux and macOS.

## Features

- **Game state extraction**: hand cards, chips, hands/discards remaining, blind requirement
- **Action execution**: select cards, play hands, discard
- **Socket transport**: request/response over `127.0.0.1:12345`, `TCP_NODELAY`, auto-reconnect
- **Telemetry**: durable per-run event stream with full latency attribution
- **Replay system**: saves the top 10 winning games by chip score
- **Custom reward function**: rewards efficient play, big hands, and beating the blind
- **Automated training**: runs restart automatically after a win or loss

## Installation

### Prerequisites
- Balatro
- [Lovely Injector](https://github.com/ethangreen-dev/lovely-injector) for mod injection
- Python 3.8+

### Setup
1. Install Lovely Injector (`version.dll` next to `Balatro.exe`)
2. Symlink or copy `RLBridge/` into `%AppData%\Balatro\Mods\`
3. `pip install -r ai/requirements.txt`
   - For CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu126`
4. Start the trainer **first** (it owns the listening port), then launch Balatro and press `R`

```bash
python -m ai.train_balatro
```

See [walkthrough.md](walkthrough.md) for a detailed Windows setup guide.

## Architecture

- **RLBridge mod**: Lua, injected via Lovely's patching system
- **Localhost socket**: line-delimited JSON on `127.0.0.1:12345`; Python is the server, the
  mod is the client. Override with `BALATRO_RL_HOST` / `BALATRO_RL_PORT`.
- **Python environment**: Gymnasium environment trained with MaskablePPO (sb3-contrib)
- **Telemetry** (`ai/telemetry/`): stdlib-only NDJSON event stream. `emit()` never blocks the
  training loop — it drops events rather than adding latency to the hot path.

Each run writes a self-contained directory:

```
runs/<timestamp>_<name>/
  meta.json      # git sha, hyperparams, device, PIDs
  events.jsonl   # durable event stream
  tb/            # per-run TensorBoard logs
  monitor.csv    # SB3 Monitor output
  checkpoints/
```

## Performance

Training throughput is bound entirely by the game, not by Python — measurements put
Python at ~0.2 ms of a step. Profiling the round trip found two engine paths that
bypass Balatro's `GAMESPEED` multiplier:

- `EventManager:update` is driven by `real_dt` and rate-limits itself to one blocking
  event per 1/60 s, capping the game at ~60 blocking events per real second no matter
  how high `GAMESPEED` is set.
- `Moveable:move` is driven by `real_dt` capped at 1/20 s and never scaled at all.

Forcing the event queue to drain within a frame (`BALATRO_RL_DRAIN`, on by default)
gives a measured **4.6x** speedup:

| | ms/step | steps/sec | PLAY_HAND |
|---|---|---|---|
| Stock pacing | 296 | 3.4 | 995 ms |
| Forced event drain | **65** | **15.5** | **193 ms** |

Pumping card movement as well (`BALATRO_RL_PUMP`) was measured as a **regression**
(114 ms/step, one step reaching 13 s) and is off by default.

Inspect any run with:

```bash
python -m ai.tools.latency_report
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BALATRO_RL_HOST` / `BALATRO_RL_PORT` | `127.0.0.1` / `12345` | Socket address (both sides) |
| `BALATRO_RL_DEVICE` | auto | Force `cpu` or `cuda` |
| `BALATRO_RL_DRAIN` | on | Force-drain the event queue (the 4.6x speedup) |
| `BALATRO_RL_PUMP` | off | Also pump card movement (measured regression) |
| `BALATRO_RL_DIAG` | on | Sample which events block the queue |
| `BALATRO_RL_VERBOSE_REWARD` | on | Print per-episode reward breakdowns |

## Monitor

A live dashboard over the same event stream. It only ever *reads*
`runs/<id>/events.jsonl`, so it can be started, stopped or restarted mid-run
without affecting training.

```bash
pip install -r requirements-monitor.txt
python -m monitor
```

Panels sit in a dock area and can be dragged, resized, stacked or torn off; use
**Save layout** to keep the arrangement. Adding a chart means adding one file to
`monitor/panels/` — the registry discovers it and the shell docks it, with no
other code to touch.

Three properties keep it from competing with training for resources: one shared
timer repaints everything at 10 Hz rather than 60, hidden panels skip their
repaint entirely, and every series is a bounded ring buffer decimated before
drawing.

Colours come from a palette validated for colour-vision deficiency (worst
adjacent pair ΔE 9.4, all slots ≥3:1 against the surface). Action colours are
keyed to action id, so `PLAY_HAND` is the same hue in every panel.

## Tests

No game required — each test stands up a fake Balatro client over the real socket,
and the monitor renders offscreen to a screenshot. See [tests/README.md](tests/README.md).

## Future Work

- Analysis and replay tabs; process supervisor with a kill switch
- Optional remote web view for checking long runs away from the desk
- Training parallelization across multiple game instances
- Expand beyond ante 1 (jokers, shop, blind selection)

## License

MIT — see [LICENSE](LICENSE).
