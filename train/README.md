# train/ — Balatro PPO training stack (P6+)

Custom CleanRL-style PPO (PyTorch, single GPU) with an entity-token
transformer policy and masked autoregressive composite actions. Built
against the env **contract** in `balatro_train/env_api.py` +
`balatro_train/encoding.py`; it runs on the pure-Python `MockBalatroEnv`
("mock") or the **Rust PyO3 vec-env** ("sim", crate `sim/py`, module
`balatro_sim`) with zero changes to any training code.

## Setup

```sh
cd train
conda activate balatro   # torch (CUDA), numpy, tensorboard, pyyaml — see repo README
```

## Build the Rust vec-env (`balatro_sim`)

The extension is a local path dependency built by maturin (release profile
by default, see `sim/py/pyproject.toml`):

```sh
cd train
pip install ./sim/py                                 # first build + install
pip install --force-reinstall --no-deps ./sim/py     # after ANY Rust change
```

(Equivalent manual flow: `maturin develop --release -m ../sim/py/Cargo.toml`.)
Pure-Rust tests of the binding (encoder invariants, action mapping,
determinism, snapshot round-trip) live in `sim/py/tests/` and run with
`cargo test` in `sim/` — no Python needed.

Then train on the real env:

```sh
python -m balatro_train.ppo --config configs/sim_debug.yaml
```

### Gates / benchmarks

```sh
pytest                                          # incl. tests/test_sim_env.py
python scripts/sim_soak.py --envs 256 --total-steps 5000000   # M0 soak
python scripts/sim_soak.py --bench              # steps/s at N=256/1024/2048
python scripts/sim_bot_eval.py --seeds 1000     # heuristic-bot baseline
```

### Binding decisions (P6, summarized; full docs in `sim/py/src/*.rs`)

* **Seeds**: env *i* owns a SplitMix64 stream seeded with `seeds[i]`; every
  episode draws the next u64 → 8-char base-36 (`[0-9A-Z]{8}`) Balatro seed
  (valid in the real game — `SimBalatroEnv.run_seed(i)` exposes it for P5
  live replays). Auto-reset seeds come from the same stream, so trajectories
  are pure functions of the reset seeds.
* **Joker state scalars** (`joker_feats[6:10]`, sign·log1p):
  `[state.mult, state.chips, state.x_mult − 1, state.extra]` — the effect
  engine's four accumulators (Ride the Bus mult, Runner chips, Vampire-style
  x_mult scalers, named counters like Seltzer hands / Turtle Bean size).
* **PLAY under The Psychic**: the sim scores a <5-card play as 0 (debuffed
  hand) rather than rejecting it, exactly like the Lua `evaluate_play` — so
  PLAY_HAND stays legal whenever the hand is non-empty and no subset-size
  masking is needed.
* **Cerulean Bell**: the forced card is auto-included in every play/discard
  (appended, or replacing the last pick at 5) — per-card masks cannot
  express "must include".
* **USE_CONSUMABLE leniency** (contract): targets truncated to
  `max_highlighted`, padded from the lowest hand indices to
  `min_highlighted`, Aura re-pointed at an edition-less card; if the
  consumable is still unusable (shared USE/SELL target mask cannot express
  usability) the action is a **no-op**, never an error.
* **PICK_PACK** of a targeted consumable auto-picks targets the same way
  (the action space has no card head for packs).
* **Hand clamp**: hands can exceed 10 (Turtle Bean, The Serpent); the obs
  shows the first 10 in display order and only those are selectable.
* **Not expressible in the v1 action space** (documented frictions):
  boss reroll (Director's Cut/Retcon voucher action), buy-and-use from the
  shop, joker reordering (MOVE_JOKER reserved; inventory order is kept
  as-is), To Do List's target hand in obs, and flipped-joker id hiding
  (Amber Acorn ids stay visible).
* **Rewards** (env-side, Rust): per-hand `β·Δmin(chips/req, 1)`; blind clear
  `β·0.5·(1+0.15·ante)` (whatever action cleared it); win +15 unscaled;
  no loss penalty; `set_shaping_beta` scales only the shaping terms.
  Episode `info["episode"] = {r, l, ante, won}`; a win reports the
  post-boss ante (9).
* **snapshot(i) / restore(i, bytes)**: full bincode serialization of the
  `Run` (RNG stream states included) + episode stats + the env's seed
  stream — restorable across processes, bit-identical continuation.
* **redeterminize(i, seed)**: replaces ALL future RNG streams with a fresh
  seed's and reshuffles the undrawn deck; every observable (incl. drawpile
  counts) is unchanged. For determinized search.

## Run debug training on the mock env

```sh
python -m balatro_train.ppo --config configs/debug.yaml
tensorboard --logdir runs        # curves
```

`configs/debug.yaml` is a tiny CPU config (16 envs, d_model 64).
`configs/default.yaml` is the plan's full-size setup (2048 envs × 128
rollout, d_model 128, γ=0.999) — it runs against the mock too, but only
makes real sense once the Rust vec-env lands (`env.name: sim`).
`configs/m1.yaml` is the M1 run: full-size on the sim with the ante-limit
curriculum; TB events + checkpoints land in `runs/m1/` (gitignored).

## Ante-limit curriculum

Envs take a `win_ante` ctor kwarg (1..=8, default 8) and expose
`set_win_ante(n)`: the episode ends with the +15 win bonus (unscaled by β)
and `info["episode"]["won"] = True` once the boss of ante `win_ante` is
defeated.  `set_win_ante` applies at each env's NEXT (auto-)reset —
in-flight episodes keep their goalpost (a full `reset()` applies it
everywhere).  With `curriculum.enabled: true` the PPO loop starts at
`start_ante`, runs an argmax eval on a held-out seed block every
`eval_every_steps` (`promotion_eval_episodes` episodes), and promotes along
`ladder` (default 1→2→3→5→8) when the win rate reaches `promote_winrate`;
a full-game (win_ante 8) milestone eval runs every `milestone_every_steps`.
TB: `curriculum/win_ante`, `curriculum/promo_win_rate`,
`curriculum/win_rate_ante_<k>`, `curriculum/promotion`, `eval/*`,
`charts/ep_win_rate`.  β annealing stays a manual hook
(`PPOTrainer.set_shaping_beta`, β=1.0 for M1).

## Snapshot start-state mixing (M3 backward curriculum)

Envs expose `load_snapshot_pool(path)` + `set_snapshot_fraction(f)`: with
probability `f` an AUTO-reset starts from a uniformly sampled mid-run
snapshot out of a packed pool file instead of a fresh seed (a full
`reset()` always starts fresh).  Pool file format:
`balatro_train/snapshot_pool.py` (writer) / `sim/py/src/pool.rs` (validated
Rust reader; ONE shared in-memory copy per vec-env).  Semantics
(`sim/py/src/env.rs::begin_episode`): every draw (include decision, pool
index, redeterminize re-seed of the restored run's future) comes from the
env's own per-env seed stream — deterministic given (seeds, pool, fraction
schedule), and `f == 0` is bit-identical to a pool-less env; only snapshots
with `ante < win_ante` are eligible (fresh fallback if none);
`info["episode"]` gains `"from_snapshot"`.

Generate a pool (bot-only skews antes 3-4; add `--ckpt` to co-drive envs
with a policy for deeper coverage; manifest json written next to the pool):

```sh
python -m balatro_train.gen_snapshots --out runs/pools/pool_v1.bin
```

Trainer (`snapshots:` config block, see `configs/m3.yaml`): loads the pool,
anneals the fraction linearly (`fraction` → `anneal.final_fraction` over
`anneal.start_step..end_step` — take it to 0 before the end of the run so
the final regime is fresh-start).  Promotion/milestone evals build separate
envs that never load a pool — eval win rates are always fresh-start.  TB:
`snapshots/snapshot_fraction`, `snapshots/ep_from_snapshot_share`,
`snapshots/ep_{return,win_rate}_{fresh,snapshot}` (per-source — snapshot
episodes start mid-run, so never compare their win rates to fresh ones).
Resume support: `python -m balatro_train.ppo --config ... --resume ckpt.pt`
restores policy + optimizer + return normalizer + step counters +
curriculum `win_ante` (backward compatible with M1 checkpoints).

Fixed-seed evaluation (stochastic + argmax, held-out seeds):

```sh
python -m balatro_train.eval --checkpoint checkpoints/debug/ckpt_*.pt
```

Tests (masking invariants, logprob/entropy consistency, GAE, buffer
round-trip, PPO overfit + determinism):

```sh
pytest
```

## Throughput (perf toggles)

`configs/m3_fast.yaml` = `m3.yaml` + the `perf:` block (see
`config.PerfConfig`).  Every toggle defaults OFF and the defaults are
**bit-identical** to the pre-perf trainer (gated by seeded A/B in
`tests/test_perf.py` and a saved-loss-history comparison at review time).

| toggle | what | numerics |
|---|---|---|
| `perf.bf16_update` | bf16 autocast for the update forward/backward (fp32 master weights, fp32 losses; logprobs stay fp32 via autocast's fp32 logsumexp). Attention already runs fused SDPA (mem-efficient backend) on torch 2.13 even in fp32 — bf16 halves the gemm/elementwise cost and the activation VRAM. | reduced precision |
| `perf.compile` | `torch.compile` on `policy._forward` (rollout + update + eval shapes each specialize once; `dynamic=False` — all shapes here are static, incl. the 5 GRU card sub-steps). **Needs Triton**; no official Windows build exists, `pip install triton-windows` is a working unofficial one. Without it, `compile: true` raises `torch._inductor.exc.TritonMissing` and crashes rather than falling back to eager. Measured worth only **~5%** on a 6GB RTX 4050 (610 → 639 steps/s) and cost ~10 min of first-run compile warmup, so `m3_fast.yaml` ships it **off** — on this card `minibatch_size` dominates, not codegen. | codegen (validated) |
| `perf.buffer_device` | `cpu` / `cuda` / `auto`. `cuda` keeps the rollout buffer GPU-resident: per-step stores are device-side copies queued **before** `env.step` (they overlap the Rust step), and minibatch fetch is a pure GPU gather instead of a per-epoch CPU-gather + H2D of the whole batch. `auto` picks cuda when the buffer fits next to the measured update peak. | bit-exact |
| `perf.pin_memory` | pinned staging + `non_blocking` H2D for per-step obs/masks and (cpu-buffer) minibatch fetches (stage reuse is guarded by a CUDA event). | bit-exact |
| `perf.auto_minibatch` | startup VRAM probe on a throwaway policy copy: halves `ppo.minibatch_size` until a worst-case synthetic fwd/bwd/Adam step fits (floor 1024). **⚠️ Makes `minibatch_size` portable, not fast.** It optimises for *fitting*, and the largest size that fits can be far from the fastest: on a 6GB RTX 4050 it chose 16384 (probe: 7,702 MiB peak, "fits") and ran at ~640 steps/s, where a hand-set 4096 ran at ~5,250 — an 8x difference, because 16384 leaves the card at ~5.8/6.1 GB and thrashes instead of cleanly OOMing. Prefer `auto_minibatch: false` plus a value tuned by measuring `charts/sps` on the target GPU; use the probe only as a first guess on unfamiliar hardware. | bit-exact (RNG untouched) |
| `perf.fused_adam` | single-kernel fused CUDA Adam. | bit-exact state, fused kernel |

Also unconditional (bit-exact): rollout `act()` skips entropy kernels
(`need_entropy=False`; entropy was discarded), GAE always runs on CPU fp32
regardless of buffer device, and update stat `.item()` syncs are deferred to
the end of the update.

What is *not* overlapped, deliberately: policy inference for step *t+1*
cannot overlap `env.step(t)` (hard data dependency), and collection for
iteration *k+1* never overlaps update *k* — on-policy collection must use
post-update weights.

Equivalence evidence (2026-07): all perf-off and bit-exact-toggle runs
reproduce the pre-perf losses bit-for-bit (seeded mock-env, cpu+cuda);
bf16_update+compile tracked a 30-iteration win_ante=1 sim run within 0.2σ
on win-rate/return/loss/entropy/KL (identical 0.882 last-10 win rate).
Measured on the local 4070 *while the m3 run occupied the GPU*: ~6.8k →
~10.2k steps/s end-to-end at N=2048 (expect a much larger factor on an idle
GPU; per-phase breakdown: `scripts/bench_ppo.py`).

```sh
# phase breakdown / throughput bench (contention-aware; see --help)
python scripts/bench_ppo.py --config configs/m3_fast.yaml \
    --ckpt <ckpt-copy>.pt --iters 2 --warmup-iters 1 --no-sync
```

## Layout

| file | what |
|---|---|
| `balatro_train/encoding.py` | **Single source of truth**: obs/mask/action array specs, enums, feature index layouts. The Rust binding must match it exactly. |
| `balatro_train/env_api.py`  | `VecEnv` protocol + full contract doc (auto-reset, mask semantics, reward shaping scheme, β anneal hook) + env registry. |
| `balatro_train/mock_env.py` | Pure-numpy fake env with correct mask semantics; strict action validator; `shaped` and `bandit` reward modes. |
| `balatro_train/policy.py`   | Entity-token transformer (3×128 pre-LN), pointer heads, autoregressive card head w/ GRU, value head; exact −inf masking; one shared code path for sampling and PPO recompute. |
| `balatro_train/buffer.py`   | Rollout buffer (dict obs + stored masks + composite actions), GAE, running return normalizer. |
| `balatro_train/ppo.py`      | Train loop (`python -m balatro_train.ppo`), TB logging, checkpointing (policy+optimizer+normalizer), `--resume`, snapshot-fraction anneal. |
| `balatro_train/eval.py`     | Fixed-seed eval protocol skeleton. |
| `balatro_train/snapshot_pool.py` | Snapshot pool file writer/reader + manifest helpers (Rust reader: `sim/py/src/pool.rs`). |
| `balatro_train/gen_snapshots.py` | Pool generation tool (bot and/or policy-checkpoint driven, stratified per-ante quotas). |
| `balatro_train/telemetry/` | NDJSON event stream (`events.jsonl`) + `RunSession` (run directory, `meta.json`, stop file). Stdlib only. |
| `balatro_train/tools/`     | `killrun` (graceful stop / force kill from the terminal), process helpers, event-stream reader. |
| `configs/`                  | `debug.yaml` (tiny/CPU), `default.yaml` (plan hypers), `m1.yaml` (first real run), `m3.yaml` (m1 + snapshot mixing + resume), `m3_fast.yaml` (m3 + perf toggles; see "Throughput"). |
| `scripts/bench_ppo.py`      | Wall-clock phase breakdown + torch.profiler bench of collect/update (supports `--set path=value` config overrides). |

## Event stream and stopping a run

Alongside TensorBoard, the trainer appends a newline-delimited JSON event stream
to `runs/<run_name>/events.jsonl`. It is the contract the monitor, the web view
and `monitor/analysis.py` read; none of them import this package's trainer.

| event | when | carries |
|---|---|---|
| `session_start` | trainer startup | run id, device, env count, target steps, policy size |
| `rollout` | every iteration | the same `stats` dict that goes to TensorBoard, plus per-rollout `action_counts` |
| `episode_end` | each finished episode | return, length, terminal ante, won, whether it started from a snapshot |
| `promotion_eval` / `curriculum_promotion` | curriculum checks | win rate at the current goal, threshold, ladder move |
| `milestone_eval` | full-game eval | ante-8 argmax win rate, ante histogram |
| `checkpoint_saved`, `session_end` | as named | path, step, exit status |

Two rules keep it cheap. `emit()` never blocks — a background thread drains a
queue to disk and drops events rather than adding latency. And **nothing
per-step goes in**: at ~10k env-steps/s a per-step event would swamp the stream,
so anything per-step is aggregated in the hot path and emitted once per
iteration (see `action_counts` in `collect()`).

Adding a field is safe; renaming or removing one breaks those readers silently.
New metrics belong in the `stats` dict `train()` already builds, so TensorBoard
and the event stream stay in agreement by construction.

**Stopping a run** is a file, not a signal — it works regardless of who started
the trainer or whether they share a process tree. The trainer checks once per
iteration and then falls through to the same post-loop checkpoint save that a
completed run uses, so a stop still lands a final checkpoint:

```bash
python -m balatro_train.tools.killrun --stop
```

```bash
python -m balatro_train.tools.killrun --kill
```

`--kill` terminates immediately and loses the in-flight rollout. Both act on the
newest active run in `runs/`, or on a directory you name.

## What plugs in later

1. **Rust PyO3 vec-env** (`balatro_sim`): implement `env_api.VecEnv`
   (reset/step/num_envs/set_shaping_beta, auto-reset, rewards env-side),
   emit arrays per `encoding.py`, then register it in
   `env_api.make_vec_env("sim", ...)` and set `env.name: sim` in the config.
   Contract decisions the binding must honor are listed in `encoding.py` /
   `env_api.py` docstrings.
2. **Curriculum / snapshot starts** — DONE (ante-limit curriculum + snapshot
   start-state mixing above); β annealing remains a manual hook
   (`PPOTrainer.set_shaping_beta`).
3. **wandb** — optional (`log.wandb: true`); TensorBoard is the default.
4. **Heuristic bot + M0 gates** (100M-step invalid-action counter, ≥20k
   steps/s) run against the real binding; the same masking tests here are
   the template.
