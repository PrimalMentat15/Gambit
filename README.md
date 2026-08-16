# Gambit

A reinforcement learning agent that plays [Balatro](https://www.playbalatro.com/),
built on a seed-faithful Rust simulator, a PPO training stack, and a live monitoring
and run-control layer.

## Components

- **`sim/`** — a seed-faithful headless reimplementation of Balatro in Rust: bit-exact RNG
  (`pseudohash`/`pseudoseed` streams + LuaJIT's Tausworthe `math.random`), the full scoring
  pipeline, all 150 jokers, consumables, vouchers, tags, and boss blinds. ~300 tests, including
  oracle vectors generated with a pinned game-era LuaJIT and full-state cross-validation against
  the real game. Exposed to Python as a vectorized PyO3 env (`sim/py`) that steps thousands of
  games in parallel.
- **`train/`** — a CleanRL-style PPO stack: entity-token transformer policy (~1.1M params) with
  autoregressive masked pointer heads over a composite action space, ante-ladder curriculum,
  snapshot start-state mixing (backward curriculum), and throughput tuning (bf16 updates,
  `torch.compile`, GPU-resident buffers).
- **`bridge/`** — a client for [coder/balatrobot](https://github.com/coder/balatrobot) that drives
  the *real* game for evaluation and for the cross-validation harness (`balatro-crossval`) that
  diffs sim vs. game state after every action of seeded runs.
- **`monitor/`** — a live PyQt dashboard plus a phone-checkable SSE web view, both reading the
  run's NDJSON event stream. Read-only: start, stop or restart it mid-run without affecting
  training.

## Environment

One conda environment, `balatro`, covers the whole repo — trainer, simulator bindings, bridge and
monitor. There is no per-directory virtualenv and nothing needs a `uv run` prefix.

First-time setup (needs a Rust toolchain for the simulator bindings):

```bash
conda create -n balatro python=3.13 -y
```

```bash
conda activate balatro
```

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

```bash
pip install -e train -e bridge -r requirements-monitor.txt pytest pytest-timeout
```

```bash
pip install ./sim/py
```

Torch comes from the CUDA index explicitly, and goes in **first**: the default PyPI wheel is
CPU-only on Windows, so letting `train`'s own dependency resolve it silently costs you the GPU.
That wheel is ~2.6 GB and `pip` does not resume a dropped transfer, so on a flaky connection fetch
it with a resumable client and install from the file instead:

```bash
curl -L -C - --retry 30 --retry-delay 5 --retry-all-errors -o torch.whl "https://download.pytorch.org/whl/cu126/torch-2.13.0%2Bcu126-cp313-cp313-win_amd64.whl"
```

```bash
pip install torch.whl
```

Verify with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Rebuild the simulator bindings after any Rust change:

```bash
pip install --force-reinstall --no-deps ./sim/py
```

### Tuning throughput for your GPU

The single biggest throughput lever is `ppo.minibatch_size`, and it must be
tuned **by measurement on your actual card**, not left to the automatic probe.

Measured on a 6GB RTX 4050 laptop GPU, same run, same everything else:

| `minibatch_size` | steps/sec |
|---|---|
| 4096 | **~5,250** |
| 16384 | ~640 |

That is an **8x collapse**, and `perf.auto_minibatch: true` picks the slow one:
its probe checks whether an update step *fits* in VRAM (it measured 7,702 MiB
as fitting), not whether it runs *fast*. At 16384 the card sits at ~5.8/6.1 GB
and thrashes rather than cleanly OOMing. So `m3_fast.yaml` ships
`auto_minibatch: false` with a hand-tuned `minibatch_size: 4096`.

If you move to a different GPU, re-tune by running a few iterations at a couple
of values and comparing `charts/sps` — don't assume a bigger minibatch is
faster.

`perf.compile` (`torch.compile`) needs Triton, which has no official Windows
build; `pip install triton-windows` is a working unofficial one. On this card
it was worth only ~5% (610 → 639 steps/s), so it ships **off** — it is not what
gates throughput here. If you do enable it, note the first run spends ~10
minutes compiling before the first iteration completes, and without Triton
installed it crashes with `torch._inductor.exc.TritonMissing` rather than
falling back to eager.

## Quick start

```bash
cd sim && cargo test
```

```bash
cd train && pytest
```

```bash
cd train && python -m balatro_train.ppo --config configs/m3_fast.yaml
```

```bash
cd train && tensorboard --logdir runs/
```

`CLAUDE.md` documents the repo conventions; `train/README.md` covers the training stack and
throughput toggles; `bridge/README.md` covers real-game evaluation and cross-validation.

## Run directories

Each run owns a directory named after its config's `run_name`:

```
train/runs/<run_name>/
  meta.json      # git sha, hyperparams, device, trainer pid
  events.jsonl   # durable event stream
  ckpt_*.pt      # checkpoints
  events.out.*   # TensorBoard
```

The name is fixed rather than timestamped so a resumed run keeps one identity: `--resume` paths
stay valid, and TensorBoard draws one continuous curve across restarts instead of a new disconnected
run per launch.

`emit()` never blocks the training loop — it drops events rather than adding latency to the
hot path.

## Monitor

```bash
python -m monitor
```

Panels sit in a dock area and can be dragged, resized, stacked or torn off; use **Save layout**
to keep the arrangement. Adding a chart means adding one file to `monitor/panels/` — the registry
discovers it and the shell docks it, with no other code to touch.

Three properties keep it from competing with training for resources: one shared timer repaints
everything at 10 Hz rather than 60, hidden panels skip their repaint entirely, and every series is
a bounded ring buffer decimated before drawing.

Colours come from a palette validated for colour-vision deficiency (worst adjacent pair ΔE 9.4,
all slots ≥3:1 against the surface).

### Remote view

```bash
python -m monitor.web --host 0.0.0.0
```

A single self-contained page streaming live stats over Server-Sent Events, for checking a
multi-hour run from your phone. Stdlib only — no dependency and no JS toolchain. Read-only by
construction: no route writes, and the kill switch is deliberately not exposed over the network.
Binds to localhost unless you pass `--host`.

### Stopping a run

Stop is cooperative — the trainer writes a final checkpoint and exits at the next iteration
boundary. Kill is terminate-based, for when that is not enough.

```bash
python -m balatro_train.tools.killrun --stop
```

```bash
python -m balatro_train.tools.killrun --kill
```

## A note on game code and assets

Balatro is © LocalThunk, published by Playstack. This project is an independent, noncommercial
research/educational effort with **no affiliation to or endorsement by** either. The repo contains
**no game code or assets** (no art, audio, fonts, or shaders):

- The simulator is a clean-room-style *reimplementation* (each behavior cites the game source
  location it mirrors, but the code is original Rust).
- The Lua oracle harnesses used to generate some test vectors replicate game functions verbatim and
  are therefore **excluded from this repository** (gitignored, local-only), as is the extracted game
  source they validate against. The generated numeric test vectors (`sim/core/tests/data/*.tsv`) are
  included.
- To work on sim fidelity you need to own Balatro (the source ships inside the game's executable —
  see `CLAUDE.md`). The simulator is headless — no UI, art, or content — and is not a substitute for
  the game; if you want to play Balatro, [buy it](https://www.playbalatro.com/).
- If you are the rights holder and have any concern about this repository, please open an issue and
  it will be addressed promptly.

See [DECISIONS.md](DECISIONS.md) for the engineering log — what was tried, what was measured, and
why each choice was made.

## License

MIT — see [LICENSE](LICENSE).
