# Gambit

A reinforcement learning agent that plays Balatro, combining a Lua mod for game
instrumentation with a Python training environment.

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

Separately, **15.3% of steps used to be discarded outright**. The action space was
8 independent binary bits, so the policy could select 6–8 cards, but Balatro caps
hand selection at 5 and the mod rejected anything outside 1–5 — burning the whole
step. This was first patched by clamping the selection after the fact, which
removed the waste but meant the executed action differed from the sampled one
(action projection), on 58.5% of `SELECT_HAND` steps. It is now fixed properly by
the autoregressive action space below, and clamping is retained only as an
assertion that should never fire.

Inspect any run with:

```bash
python -m ai.tools.latency_report
```

### Autoregressive card selection

The action space is `MultiDiscrete([3, 9, 9, 9, 9, 9])` — one action-type choice
followed by up to 5 card picks, each holding a hand position (0–7) or STOP (8).
`AutoregressiveCardPolicy` (`ai/policies/`) picks them sequentially inside a
**single forward pass**, masking each pick against the cards already taken and
gating STOP on the minimum. Illegal selections — too many cards, duplicates,
none at all — are therefore unrepresentable rather than corrected afterwards.

Two properties this preserves, both verified in `tests/test_autoregressive.py`:

- **One environment step per decision.** All 5 sub-decisions happen inside the
  policy, so the env still does exactly one socket round-trip per `step()`.
  Since the game round-trip is ~99% of wall-clock time, splitting a selection
  into 5 env steps would have been a large regression.
- **`log_prob` reproduces exactly** between sampling and PPO's recompute in
  `evaluate_actions` (measured difference: 0.0). Sampling and scoring share one
  code path specifically so they cannot drift — a mismatch would corrupt PPO's
  ratio silently, with no error to notice.

`PLAY_HAND`/`DISCARD_HAND` take no card parameters, so their slots are forced to
STOP and contribute no log-probability: the policy is not graded on a decision
the game ignores.

### Observation size is stable at any hand size

The observation is a fixed `Box(216,)`, built by encoding up to `MAX_CARDS` cards
into a fixed-size block. Hand size is not a constant — jokers/vouchers change it,
and consumables like Cryptid insert copies directly into hand, which the engine
does not cap the way it caps the deck — so an observation whose length depended on
however many cards happened to be in hand would silently violate the declared
space the moment the hand size drifted from a specific number. `max_cards` is now
passed into `BalatroStateMapper` at construction (`BalatroEnv.MAX_CARDS`) so it
tracks the environment's actual action space rather than being a second constant
that has to be kept in sync by hand, oversized hands are truncated with a visible
counter (`hand_truncated`) rather than changing the array length, and a final
length check acts as a safety net that logs loudly if it ever fires — which it
should not, if the padding logic above it is correct.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `BALATRO_RL_HOST` / `BALATRO_RL_PORT` | `127.0.0.1` / `12345` | Socket address (both sides) |
| `BALATRO_RL_DEVICE` | auto | Force `cpu` or `cuda` |
| `BALATRO_RL_DRAIN` | on | Force-drain the event queue (the 4.6x speedup) |
| `BALATRO_RL_PUMP` | off | Also pump card movement (measured regression) |
| `BALATRO_RL_DIAG` | on | Sample which events block the queue |
| `BALATRO_RL_VERBOSE_REWARD` | on | Print per-episode reward breakdowns |
| `BALATRO_RL_AUTOSTART` | off | Start training without pressing `R`. Set per-process by the supervisor; exporting it globally would make every manual launch start training |
| `BALATRO_RL_RECONNECT_TIMEOUT` | `300` | Seconds `reset()` waits for Balatro to reconnect before giving up, so closing the game mid-run no longer kills the trainer |

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

### Tabs

| Tab | What it does |
|---|---|
| **Live** | Dockable panels: throughput, latency by action, latency over time, episode reward, win rate, game state, action distribution, event log |
| **Control** | Launch a run (optionally starting Balatro too), watch process health, read trainer output |
| **Analysis** | Compare runs on any TensorBoard metric, plus reward-component and hand-type tables. Reads event files directly — no TensorBoard server |
| **Replays** | Browse saved winning games, see the seed each one was played on, and step through their actions |

### Starting and stopping runs

The Control tab launches the trainer and Balatro. Children are spawned to outlive
the monitor, so **closing the monitor leaves training running** and reopening it
re-attaches by reading the run directory.

Two toolbar buttons act on the same targets and differ only in cleanliness:

| | Trainer | Balatro | Monitor |
|---|---|---|---|
| **Stop** (`Ctrl+.`) | checkpoints, exits at next step boundary | closes after | stays open |
| **Kill** (`Ctrl+Shift+.`) | terminated immediately | terminated immediately | stays open |

Stop is cooperative, so it cannot help when the trainer is wedged in a socket read
waiting on the game — which is exactly why Kill is terminate-based. Kill also
closes Balatro, since a half-dead pair holding port 12345 is what blocks the next
run.

Neither depends on the GUI:

```bash
python -m ai.tools.killrun --stop    # graceful
python -m ai.tools.killrun --kill    # force
```

### Remote view

```bash
python -m monitor.web --host 0.0.0.0
```

A single self-contained page streaming live stats over Server-Sent Events, for
checking a multi-hour run from your phone. Stdlib only — no dependency and no JS
toolchain. Read-only by construction: no route writes, and the kill switch is
deliberately not exposed over the network. Binds to localhost unless you pass
`--host`.

## Tests

No game required — each test stands up a fake Balatro client over the real socket,
and the monitor renders offscreen to a screenshot. See [tests/README.md](tests/README.md).

## Future Work

- Training parallelization across multiple game instances
- Expand beyond ante 1 (jokers, shop, blind selection)

See [DECISIONS.md](DECISIONS.md) for the full engineering log — what was tried,
what was measured, and why each choice was made.

## License

MIT — see [LICENSE](LICENSE).
