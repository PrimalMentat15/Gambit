"""CleanRL-style PPO training loop for the Balatro vec-env contract.

Run:  python -m balatro_train.ppo --config configs/debug.yaml

Key properties (see project plan):
* Joint logprob/entropy over the composite autoregressive action (sum of
  conditional terms) — PPO consumes joint quantities.
* Minibatch recompute uses the masks STORED in the rollout buffer.
* GAE(gamma=0.999, lambda=0.95), advantage normalization per minibatch,
  running return normalization, lr annealed linearly to 0.
* bf16 autocast for rollout inference; updates in fp32.
* TensorBoard always (unless disabled); wandb optional, off by default.
* Checkpoints carry policy + optimizer + return-normalizer state + config.
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from balatro_train.buffer import ReturnNormalizer, RolloutBuffer
from balatro_train.config import (
    ShapingConfig,
    SnapshotConfig,
    TrainConfig,
    config_to_dict,
    load_config,
)
from balatro_train.encoding import N_ACTION_TYPES
from balatro_train.env_api import make_vec_env
from balatro_train.policy import BalatroPolicy, actions_to_numpy, obs_to_torch
from balatro_train.telemetry import (
    EventType,
    RunSession,
    TelemetryEmitter,
    close_emitter,
    emit,
    set_emitter,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def snapshot_fraction_at(step: int, cfg: SnapshotConfig) -> float:
    """The snapshot start-state fraction at a global step (linear anneal).

    ``cfg.fraction`` until ``anneal.start_step``, linear to
    ``anneal.final_fraction`` at ``anneal.end_step``, constant after.
    ``end_step <= start_step`` disables the anneal (constant fraction).
    """
    if not cfg.enabled:
        return 0.0
    a = cfg.anneal
    if a.end_step <= a.start_step or step <= a.start_step:
        return float(cfg.fraction)
    if step >= a.end_step:
        return float(a.final_fraction)
    t = (step - a.start_step) / (a.end_step - a.start_step)
    return float(cfg.fraction + t * (a.final_fraction - cfg.fraction))


def shaping_beta_at(step: int, cfg: ShapingConfig) -> float:
    """The reward-shaping scale β at a global step (linear anneal).

    ``cfg.beta`` until ``anneal.start_step``, linear to ``anneal.final_beta``
    at ``anneal.end_step``, constant after.  ``end_step <= start_step``
    disables the anneal (constant β).  Same shape as
    :func:`snapshot_fraction_at`.
    """
    a = cfg.anneal
    if a.end_step <= a.start_step or step <= a.start_step:
        return float(cfg.beta)
    if step >= a.end_step:
        return float(a.final_beta)
    t = (step - a.start_step) / (a.end_step - a.start_step)
    return float(cfg.beta + t * (a.final_beta - cfg.beta))


class PPOTrainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        seed_everything(cfg.seed)
        self.device = torch.device(
            cfg.device if cfg.device != "cuda" or torch.cuda.is_available() else "cpu"
        )

        self.env = make_vec_env(cfg.env.name, cfg.env.num_envs, **cfg.env.kwargs)
        self.policy = BalatroPolicy(cfg.policy).to(self.device)
        use_fused_adam = cfg.perf.fused_adam and self.device.type == "cuda"
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=cfg.ppo.lr, eps=1e-5,
            fused=use_fused_adam or None,  # None = default (non-fused) impl
        )

        # ---- perf toggles (see config.PerfConfig; defaults = classic path) --
        perf = cfg.perf
        self._probe_peak = None
        if perf.auto_minibatch and self.device.type == "cuda":
            cfg.ppo.minibatch_size = self._probe_minibatch_size()
        self._pin_h2d = bool(perf.pin_memory and self.device.type == "cuda")
        self._h2d_stage: dict[str, torch.Tensor] = {}
        buffer_device = self._resolve_buffer_device()
        if perf.compile and self.device.type != "cpu":
            self.policy._forward = torch.compile(
                self.policy._forward, mode=perf.compile_mode, dynamic=False
            )

        self.buffer = RolloutBuffer(
            cfg.ppo.rollout_len, cfg.env.num_envs, self.device,
            storage_device=buffer_device, pin_memory=perf.pin_memory,
        )
        self.ret_norm = (
            ReturnNormalizer(cfg.env.num_envs, cfg.ppo.gamma) if cfg.ppo.norm_ret else None
        )

        self.global_step = 0
        self.iteration = 0
        self.ep_returns: deque[float] = deque(maxlen=100)
        self.ep_lengths: deque[float] = deque(maxlen=100)
        self.ep_antes: deque[float] = deque(maxlen=100)
        self.ep_wins: deque[float] = deque(maxlen=100)
        # Per-source episode stats (snapshot start-state mixing): fresh vs
        # from-snapshot episodes are logged separately — snapshot episodes
        # start mid-run, so their returns/win-rates are NOT comparable to
        # fresh ones (key for interpreting win rates while mixing is on).
        self.ep_from_snapshot: deque[float] = deque(maxlen=100)
        self.ep_returns_by_src = {False: deque(maxlen=100), True: deque(maxlen=100)}
        self.ep_wins_by_src = {False: deque(maxlen=100), True: deque(maxlen=100)}
        self.history: list[dict] = []  # per-iteration stats (tests consume this)
        self.action_counts = np.zeros(N_ACTION_TYPES, dtype=np.int64)

        # Ante-limit curriculum (see config.CurriculumConfig): set the win
        # ante BEFORE reset so the very first episodes use start_ante.
        cur = cfg.curriculum
        self.win_ante: int | None = None
        if cur.enabled:
            if cur.start_ante not in cur.ladder:
                raise ValueError(
                    f"curriculum.start_ante {cur.start_ante} not in ladder {cur.ladder}"
                )
            self.win_ante = int(cur.start_ante)
            self.env.set_win_ante(self.win_ante)
            self._next_promo_eval = cur.eval_every_steps
            self._next_milestone_eval = cur.milestone_every_steps

        # Snapshot start-state mixing (see config.SnapshotConfig): load the
        # pool once (one shared in-memory copy env-side) and drive the
        # fraction from the anneal schedule.  TRAINING envs only — the
        # promotion/milestone eval envs (_run_eval) are built separately
        # and never load a pool, so evals always use fresh starts.
        self.snapshot_fraction = 0.0
        self.shaping_beta = -1.0  # sentinel: first _apply_shaping_beta always pushes
        if cfg.snapshots.enabled:
            if not cfg.snapshots.pool_path:
                raise ValueError("snapshots.enabled requires snapshots.pool_path")
            n = self.env.load_snapshot_pool(cfg.snapshots.pool_path)
            print(f"[snapshots] pool {cfg.snapshots.pool_path}: {n} entries",
                  flush=True)

        seeds = [cfg.env.seed + i for i in range(cfg.env.num_envs)]
        self.obs, self.masks = self.env.reset(seeds)
        self._apply_snapshot_fraction()
        self._apply_shaping_beta()

        self.writer = None
        self.wandb = None
        run_dir = Path(cfg.log.log_dir) / cfg.log.run_name
        self.ckpt_dir = Path(cfg.log.ckpt_dir) / cfg.log.run_name

        # NDJSON event stream alongside TensorBoard, not instead of it: the
        # monitor tails events.jsonl live, TB keeps the long-run curves.
        self.session = RunSession.for_run(
            name=cfg.log.run_name,
            runs_dir=cfg.log.log_dir,
            config=config_to_dict(cfg),
        )
        if cfg.log.telemetry:
            set_emitter(TelemetryEmitter(self.session.events_path))
        emit(
            EventType.SESSION_START,
            run_id=self.session.run_id,
            device=str(self.device),
            num_envs=cfg.env.num_envs,
            rollout_len=cfg.ppo.rollout_len,
            total_timesteps=cfg.ppo.total_timesteps,
            policy_params=self.policy.num_params(),
            win_ante=self.win_ante,
        )

        if cfg.log.tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(str(run_dir))
        if cfg.log.wandb:  # optional, off by default
            import wandb

            wandb.init(project=cfg.log.wandb_project, name=cfg.log.run_name,
                       config=config_to_dict(cfg))
            self.wandb = wandb

    # ------------------------------------------------------------------ hooks

    def set_shaping_beta(self, beta: float) -> None:
        """Shaping anneal hook (plan: 1.0 -> 0.1 once ante-8 win-rate > 10%).

        The annealing *schedule* is a later curriculum concern; the plumbing
        is contract-complete today.
        """
        self.env.set_shaping_beta(beta)

    def _apply_snapshot_fraction(self) -> float:
        """Push the schedule's fraction for the current global step to the
        training env (no-op when snapshot mixing is disabled)."""
        f = snapshot_fraction_at(self.global_step, self.cfg.snapshots)
        self.snapshot_fraction = f
        if self.cfg.snapshots.enabled:
            self.env.set_snapshot_fraction(f)
        return f

    def _apply_shaping_beta(self) -> float:
        """Push the schedule's β for the current global step to the training
        env.  TRAINING env only: eval envs judge win/loss, not returns, so
        their β is irrelevant — but shaped ep_return logs from training
        episodes shrink as β anneals (expected, not a regression)."""
        b = shaping_beta_at(self.global_step, self.cfg.shaping)
        if b != self.shaping_beta:
            self.env.set_shaping_beta(b)
            self.shaping_beta = b
        return b

    # ---------------------------------------------------------------- rollout

    def _autocast(self):
        if self.cfg.bf16_rollout:
            return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    def _update_autocast(self):
        if self.cfg.perf.bf16_update and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        import contextlib

        return contextlib.nullcontext()

    # ------------------------------------------------------------ perf plumbing

    def _synthetic_minibatch(self, mb: int) -> tuple[dict, dict, dict]:
        """Worst-case-shaped dummy batch for the VRAM probe (max hand length,
        5 card picks, all masks legal — activation peak matches training)."""
        import balatro_train.encoding as E

        dev = self.device
        obs = {k: torch.zeros((mb, *shape), dtype=torch.float32, device=dev)
               if dt == np.dtype(np.float32)
               else torch.zeros((mb, *shape), dtype=torch.int64, device=dev)
               for k, (shape, dt) in E.OBS_SPEC.items()}
        obs["hand_len"].fill_(E.HAND_MAX)
        obs["joker_ids"].fill_(1)
        obs["consumable_ids"].fill_(1)
        obs["shop_ids"].fill_(1)
        masks = {k: torch.ones((mb, *shape), dtype=torch.bool, device=dev)
                 for k, (shape, _dt) in E.MASK_SPEC.items()}
        actions = {k: torch.zeros((mb, *shape), dtype=torch.int64, device=dev)
                   for k, (shape, _dt) in E.ACTION_SPEC.items()}
        actions["cards"] = torch.arange(E.MAX_CARD_PICKS, device=dev).expand(mb, -1).clone()
        actions["n_cards"].fill_(E.MAX_CARD_PICKS)
        return obs, masks, actions

    def _probe_minibatch_size(self) -> int:
        """Find the largest ``minibatch_size`` <= the configured one that fits
        in free VRAM (halving on OOM, floor 1024).  Runs a synthetic
        forward/backward/Adam step on a throwaway policy copy, so training
        state and RNG streams are untouched."""
        import copy as _copy

        cfg = self.cfg.ppo
        mb = min(cfg.minibatch_size, self.cfg.env.num_envs * cfg.rollout_len)
        while True:
            probe_policy = probe_opt = None
            try:
                probe_policy = _copy.deepcopy(self.policy)
                probe_opt = torch.optim.Adam(probe_policy.parameters(), lr=cfg.lr, eps=1e-5)
                torch.cuda.reset_peak_memory_stats(self.device)
                obs, masks, actions = self._synthetic_minibatch(mb)
                with self._update_autocast():
                    logp, ent, value, _ = probe_policy.evaluate_actions(obs, masks, actions)
                loss = (logp.float().pow(2).mean() + value.float().pow(2).mean()
                        - ent.float().mean())
                loss.backward()
                probe_opt.step()
                torch.cuda.synchronize(self.device)
                self._probe_peak = torch.cuda.max_memory_allocated(self.device)
                if mb != cfg.minibatch_size:
                    print(f"[perf] auto_minibatch: {cfg.minibatch_size} OOMs, "
                          f"using {mb}", flush=True)
                print(f"[perf] update peak (mb {mb}, "
                      f"bf16_update={self.cfg.perf.bf16_update}): "
                      f"{self._probe_peak / 2**20:,.0f} MiB", flush=True)
                return mb
            except torch.OutOfMemoryError:
                if mb <= 1024:
                    raise
                mb //= 2
            finally:
                del probe_policy, probe_opt
                torch.cuda.empty_cache()

    def _resolve_buffer_device(self) -> str:
        """Resolve ``perf.buffer_device`` ("auto" -> cuda when the buffer fits
        next to the update working set with headroom)."""
        choice = self.cfg.perf.buffer_device
        if self.device.type != "cuda":
            return "cpu"
        if choice in ("cpu", "cuda"):
            return choice
        if choice != "auto":
            raise ValueError(f"perf.buffer_device: unknown value {choice!r}")
        import balatro_train.encoding as E

        per_step = sum(
            int(np.prod(shape)) * dt.itemsize
            for spec in (E.OBS_SPEC, E.MASK_SPEC, E.ACTION_SPEC)
            for shape, dt in spec.values()
        ) + 16  # + logprob/value
        buf_bytes = per_step * self.cfg.env.num_envs * self.cfg.ppo.rollout_len
        free, _total = torch.cuda.mem_get_info(self.device)
        # headroom: measured update peak (if probed) + 25%, else 4 GiB
        headroom = int(self._probe_peak * 1.25) if self._probe_peak else 4 << 30
        dev = "cuda" if free - buf_bytes > headroom else "cpu"
        print(f"[perf] buffer_device auto -> {dev} (buffer "
              f"{buf_bytes / 2**20:,.0f} MiB, free {free / 2**20:,.0f} MiB, "
              f"headroom {headroom / 2**20:,.0f} MiB)", flush=True)
        return dev

    def _to_device(self, batch: dict) -> dict[str, torch.Tensor]:
        """numpy batch dict -> device tensors; with ``perf.pin_memory`` the
        copies stage through reusable pinned buffers with non_blocking H2D.
        Safe to reuse the stage each step: the action D2H sync (and any
        ``.item()``/``.cpu()``) drains the stream before the next overwrite."""
        if not self._pin_h2d:
            return obs_to_torch(batch, self.device)
        out = {}
        for k, v in batch.items():
            src = torch.from_numpy(np.ascontiguousarray(v))
            stage = self._h2d_stage.get(k)
            if stage is None or stage.shape != src.shape or stage.dtype != src.dtype:
                stage = torch.empty_like(src, pin_memory=True)
                self._h2d_stage[k] = stage
            stage.copy_(src)
            out[k] = stage.to(self.device, non_blocking=True)
        return out

    def collect(self) -> None:
        gpu_buffer = self.buffer.storage_device.type == "cuda"
        self.buffer.reset()
        # Aggregated here rather than emitted per step: at ~10k steps/s a
        # per-step event would swamp the stream, and a bincount per step is
        # noise next to the sim step it sits beside.
        self.action_counts = np.zeros(N_ACTION_TYPES, dtype=np.int64)
        for _ in range(self.cfg.ppo.rollout_len):
            obs_t = self._to_device(self.obs)
            masks_t = self._to_device(self.masks)
            with self._autocast():
                actions_t, logprob, _ent, value, _aux = self.policy.act(
                    obs_t, masks_t, need_entropy=False
                )
            actions_np = actions_to_numpy(actions_t)
            self.action_counts += np.bincount(
                actions_np["action_type"], minlength=N_ACTION_TYPES
            )

            # Store the policy-side half BEFORE stepping the env.  The stored
            # data is already final (obs/masks/actions/logp/value for step t),
            # so this is semantics-neutral; with a GPU-resident buffer the
            # copies are device-side and overlap the Rust env step below.
            if gpu_buffer:
                self.buffer.store_policy(
                    obs_t, masks_t, actions_t, logprob.float(), value.float()
                )
            else:
                self.buffer.store_policy(
                    self.obs, self.masks, actions_np, logprob.float(), value.float()
                )

            next_obs, next_masks, reward, done, infos = self.env.step(actions_np)
            self.global_step += self.env.num_envs

            for info in infos:
                if "episode" in info:
                    ep = info["episode"]
                    self.ep_returns.append(ep["r"])
                    self.ep_lengths.append(ep["l"])
                    self.ep_antes.append(ep.get("ante", 0))
                    self.ep_wins.append(float(ep.get("won", False)))
                    src = bool(ep.get("from_snapshot", False))
                    self.ep_from_snapshot.append(float(src))
                    self.ep_returns_by_src[src].append(ep["r"])
                    self.ep_wins_by_src[src].append(float(ep.get("won", False)))
                    emit(
                        EventType.EPISODE_END,
                        step=self.global_step,
                        r=ep["r"],
                        l=ep["l"],
                        ante=ep.get("ante", 0),
                        won=bool(ep.get("won", False)),
                        from_snapshot=src,
                    )

            stored_reward = self.ret_norm(reward, done) if self.ret_norm else reward
            self.buffer.store_env(stored_reward, done)
            self.obs, self.masks = next_obs, next_masks

        with torch.no_grad(), self._autocast():
            last_value = self.policy.get_value(
                self._to_device(self.obs), self._to_device(self.masks)
            )
        self.buffer.compute_gae(last_value.float(), self.cfg.ppo.gamma, self.cfg.ppo.gae_lambda)

    # ----------------------------------------------------------------- update

    def lr_at(self, step: int) -> float:
        """The annealed learning rate for a global step.

        The ramp's denominator is ``lr_anneal_total`` when set, else
        ``total_timesteps``.  Keeping them separable matters on resume:
        ``total_timesteps`` is also the stop condition, so extending a budget
        mid-run rescales the ramp and jumps lr at the resume point unless the
        schedule is pinned."""
        cfg = self.cfg.ppo
        if not cfg.anneal_lr:
            return cfg.lr
        total = cfg.lr_anneal_total or cfg.total_timesteps
        frac = 1.0 - step / max(total, 1)
        return max(frac, 0.0) * cfg.lr

    def update(self) -> dict:
        cfg = self.cfg.ppo
        lr = self.lr_at(self.global_step)
        if cfg.anneal_lr:
            for group in self.optimizer.param_groups:
                group["lr"] = lr

        stats_t = {k: [] for k in
                   ("loss", "pg_loss", "v_loss", "entropy", "approx_kl", "clipfrac")}
        for _ in range(cfg.update_epochs):
            for mb in self.buffer.minibatches(cfg.minibatch_size):
                # bf16_update: autocast covers the policy recompute only; the
                # loss terms below are fp32 (head outputs upcast — logsumexp
                # already runs fp32 under autocast so logprobs are fp32).
                with self._update_autocast():
                    new_logprob, entropy, new_value, _aux = self.policy.evaluate_actions(
                        mb["obs"], mb["masks"], mb["actions"]
                    )
                new_logprob = new_logprob.float()
                entropy = entropy.float()
                new_value = new_value.float()
                logratio = new_logprob - mb["logprobs"]
                ratio = logratio.exp()

                adv = mb["advantages"]
                if cfg.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                pg_loss = torch.max(
                    -adv * ratio,
                    -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef),
                ).mean()
                v_loss = 0.5 * (new_value - mb["returns"]).pow(2).mean()
                ent = entropy.mean()
                loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # Defer the .item() syncs to the end of the update: keep the
                # scalars as 0-d tensors here (values are identical).
                with torch.no_grad():
                    stats_t["loss"].append(loss.detach())
                    stats_t["pg_loss"].append(pg_loss.detach())
                    stats_t["v_loss"].append(v_loss.detach())
                    stats_t["entropy"].append(ent.detach())
                    stats_t["approx_kl"].append(((ratio - 1) - logratio).mean())
                    stats_t["clipfrac"].append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
                    )
        out = {k: float(np.mean([t.item() for t in v])) for k, v in stats_t.items()}
        out["lr"] = lr
        return out

    # ------------------------------------------------------------------ train

    def train(self, total_timesteps: int | None = None) -> list[dict]:
        total = total_timesteps or self.cfg.ppo.total_timesteps
        batch = self.cfg.env.num_envs * self.cfg.ppo.rollout_len
        start = time.time()
        # Steps completed BEFORE this process started, i.e. whatever --resume
        # restored. Throughput must be measured against work this session
        # actually did: dividing the absolute global_step by this session's
        # elapsed time reports a resumed run as impossibly fast (a run resumed
        # at 277M steps "achieves" 277M/1s in its first second) and poisons the
        # ETA that reads it.
        resumed_from = self.global_step
        last_t, last_step = start, self.global_step
        stopped = False
        while self.global_step < total:
            # One check per iteration is enough: an iteration is
            # num_envs * rollout_len steps against the Rust sim, so a stop
            # takes effect within seconds. The post-loop checkpoint below then
            # runs exactly as it does on normal completion.
            if self.session.stop_requested():
                stopped = True
                print(
                    f"[stop] requested at step {self.global_step:,} — "
                    f"saving final checkpoint and exiting",
                    flush=True,
                )
                break
            self.iteration += 1
            self._apply_snapshot_fraction()
            self._apply_shaping_beta()
            self.collect()
            stats = self.update()
            stats["iteration"] = self.iteration
            stats["global_step"] = self.global_step
            now = time.time()
            stats["sps"] = int((self.global_step - resumed_from) / (now - start))
            # Instantaneous rate over the last iteration. The session average
            # above still takes minutes to shed a slow first iteration (compile
            # warmup, cache fill), so anything asking "how fast is it right
            # now" -- the monitor's ETA especially -- wants this one.
            stats["sps_inst"] = int(
                (self.global_step - last_step) / max(now - last_t, 1e-9)
            )
            last_t, last_step = now, self.global_step
            if self.ep_returns:
                stats["ep_return_mean"] = float(np.mean(self.ep_returns))
                stats["ep_length_mean"] = float(np.mean(self.ep_lengths))
                stats["ep_ante_mean"] = float(np.mean(self.ep_antes))
                stats["ep_win_rate"] = float(np.mean(self.ep_wins))
            if self.win_ante is not None:
                stats["win_ante"] = self.win_ante
            stats["shaping_beta"] = self.shaping_beta
            if self.cfg.snapshots.enabled:
                stats["snapshot_fraction"] = self.snapshot_fraction
                if self.ep_from_snapshot:
                    stats["ep_from_snapshot_share"] = float(
                        np.mean(self.ep_from_snapshot)
                    )
                # Fresh vs snapshot episodes are different populations
                # (snapshot ones start mid-run) — log them separately.
                for src, name in ((False, "fresh"), (True, "snapshot")):
                    if self.ep_returns_by_src[src]:
                        stats[f"ep_return_{name}"] = float(
                            np.mean(self.ep_returns_by_src[src])
                        )
                        stats[f"ep_win_rate_{name}"] = float(
                            np.mean(self.ep_wins_by_src[src])
                        )
            self.history.append(stats)
            self._log(stats)
            # Kept out of `stats` so the history/TensorBoard path stays scalars.
            emit(EventType.ROLLOUT, **stats,
                 action_counts=self.action_counts.tolist())
            if self.cfg.curriculum.enabled:
                self._curriculum_step()
            if (
                self.cfg.log.save_every_iters
                and self.iteration % self.cfg.log.save_every_iters == 0
            ):
                self.save_checkpoint()
        ckpt = self.save_checkpoint()
        emit(
            EventType.SESSION_END,
            step=self.global_step,
            iteration=self.iteration,
            status="stopped" if stopped else "completed",
            final_checkpoint=str(ckpt),
            elapsed=time.time() - start,
        )
        self.session.mark_finished("stopped" if stopped else "completed")
        close_emitter()
        if self.writer:
            self.writer.close()
        return self.history

    def _log(self, stats: dict) -> None:
        if self.iteration % self.cfg.log.log_every_iters == 0:
            parts = [f"iter {stats['iteration']:4d}", f"step {stats['global_step']:>10d}",
                     f"sps {stats['sps']:>6d}"]
            if "ep_return_mean" in stats:
                parts.append(f"ep_ret {stats['ep_return_mean']:7.3f}")
                parts.append(f"ante {stats['ep_ante_mean']:4.2f}")
                parts.append(f"win {stats['ep_win_rate']:4.2f}")
            if "win_ante" in stats:
                parts.append(f"goal {stats['win_ante']}")
            parts += [f"loss {stats['loss']:7.4f}", f"ent {stats['entropy']:6.3f}",
                      f"kl {stats['approx_kl']:7.5f}"]
            print("  ".join(parts), flush=True)
        if self.writer:
            for key in ("loss", "pg_loss", "v_loss", "entropy", "approx_kl",
                        "clipfrac", "lr"):
                self.writer.add_scalar(f"losses/{key}", stats[key], self.global_step)
            for key in ("ep_return_mean", "ep_length_mean", "ep_ante_mean",
                        "ep_win_rate", "sps", "sps_inst"):
                if key in stats:
                    self.writer.add_scalar(f"charts/{key}", stats[key], self.global_step)
            if "win_ante" in stats:
                self.writer.add_scalar("curriculum/win_ante", stats["win_ante"],
                                       self.global_step)
            for key in ("snapshot_fraction", "ep_from_snapshot_share",
                        "ep_return_fresh", "ep_win_rate_fresh",
                        "ep_return_snapshot", "ep_win_rate_snapshot"):
                if key in stats:
                    self.writer.add_scalar(f"snapshots/{key}", stats[key],
                                           self.global_step)
        if self.wandb:
            self.wandb.log(stats, step=self.global_step)

    # ------------------------------------------------------------- curriculum

    def _run_eval(self, win_ante: int, episodes: int) -> dict:
        """Argmax eval on the held-out seed block at the given win ante.

        Builds a fresh eval env (cheap for the Rust sim) so training env
        state is untouched; restores policy train mode afterwards.

        FRESH STARTS ONLY: the eval env constructed here never loads a
        snapshot pool, so its snapshot fraction is structurally 0 — every
        eval episode is a full run from a held-out seed.  Curriculum
        promotions and milestone win-rates are therefore never inflated by
        snapshot mid-run starts.
        """
        from balatro_train.eval import evaluate

        cur = self.cfg.curriculum
        env_kwargs = dict(self.cfg.env.kwargs)
        env_kwargs["win_ante"] = win_ante
        stats = evaluate(
            self.policy,
            env_name=self.cfg.env.name,
            num_envs=cur.eval_num_envs,
            episodes=episodes,
            deterministic=True,
            seed_base=cur.eval_seed_base,
            device=self.device,
            env_kwargs=env_kwargs,
        )
        self.policy.train()
        return stats

    def _curriculum_step(self) -> None:
        """Periodic promotion eval (cheap) + full-game milestone eval."""
        cur = self.cfg.curriculum
        assert self.win_ante is not None

        if self.global_step >= self._next_promo_eval:
            self._next_promo_eval = (
                self.global_step // cur.eval_every_steps + 1
            ) * cur.eval_every_steps
            t0 = time.time()
            stats = self._run_eval(self.win_ante, cur.promotion_eval_episodes)
            win_rate = stats["win_rate"]
            print(
                f"[curriculum] step {self.global_step}: ante-{self.win_ante} "
                f"argmax win-rate {win_rate:.3f} over {stats['episodes']} eps "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
            if self.writer:
                self.writer.add_scalar("curriculum/promo_win_rate", win_rate,
                                       self.global_step)
                self.writer.add_scalar(
                    f"curriculum/win_rate_ante_{self.win_ante}", win_rate,
                    self.global_step,
                )
            emit(
                EventType.PROMOTION_EVAL,
                step=self.global_step,
                win_ante=self.win_ante,
                win_rate=win_rate,
                episodes=stats["episodes"],
                threshold=cur.promote_winrate,
                elapsed=time.time() - t0,
            )
            ladder = list(cur.ladder)
            if win_rate >= cur.promote_winrate and self.win_ante != ladder[-1]:
                new_ante = ladder[ladder.index(self.win_ante) + 1]
                print(
                    f"[curriculum] PROMOTION {self.win_ante} -> {new_ante} at "
                    f"step {self.global_step} (win-rate {win_rate:.3f} >= "
                    f"{cur.promote_winrate})",
                    flush=True,
                )
                if self.writer:
                    self.writer.add_scalar("curriculum/promotion", new_ante,
                                           self.global_step)
                    self.writer.add_text(
                        "curriculum/events",
                        f"step {self.global_step}: promoted win_ante "
                        f"{self.win_ante} -> {new_ante} (win-rate {win_rate:.3f})",
                        self.global_step,
                    )
                emit(
                    EventType.CURRICULUM_PROMOTION,
                    step=self.global_step,
                    from_ante=self.win_ante,
                    to_ante=new_ante,
                    win_rate=win_rate,
                    threshold=cur.promote_winrate,
                )
                self.win_ante = new_ante
                # Applies at each training env's next auto-reset.
                self.env.set_win_ante(new_ante)

        if self.global_step >= self._next_milestone_eval:
            self._next_milestone_eval = (
                self.global_step // cur.milestone_every_steps + 1
            ) * cur.milestone_every_steps
            t0 = time.time()
            stats = self._run_eval(8, cur.milestone_episodes)
            print(
                f"[milestone] step {self.global_step}: full-game argmax "
                f"win-rate {stats['win_rate']:.3f}  ante_mean "
                f"{stats['ante_mean']:.2f}  return_mean {stats['return_mean']:.2f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )
            if self.writer:
                for key in ("win_rate", "ante_mean", "return_mean", "length_mean"):
                    self.writer.add_scalar(f"eval/{key}", stats[key], self.global_step)
                for ante, count in stats["ante_hist"].items():
                    self.writer.add_scalar(
                        f"eval/ante_hist_{ante}", count / max(stats["episodes"], 1),
                        self.global_step,
                    )
            emit(
                EventType.MILESTONE_EVAL,
                step=self.global_step,
                win_ante=8,
                elapsed=time.time() - t0,
                **stats,
            )

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self, path: str | Path | None = None) -> Path:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = Path(path) if path else self.ckpt_dir / f"ckpt_{self.global_step}.pt"
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "ret_norm": self.ret_norm.state_dict() if self.ret_norm else None,
                "config": config_to_dict(self.cfg),
                "global_step": self.global_step,
                "iteration": self.iteration,
                "win_ante": self.win_ante,
            },
            path,
        )
        emit(
            EventType.CHECKPOINT_SAVED,
            step=self.global_step,
            iteration=self.iteration,
            path=str(path),
            win_ante=self.win_ante,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume: policy + optimizer + return-normalizer + step counters +
        curriculum win_ante.  Backward compatible with pre-M3 checkpoints
        (M1's): ``win_ante`` is read with ``.get`` and snapshot mixing is
        pure config — nothing snapshot-related lives in the checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.ret_norm and ckpt["ret_norm"] is not None:
            self.ret_norm.load_state_dict(ckpt["ret_norm"])
        self.global_step = ckpt["global_step"]
        self.iteration = ckpt["iteration"]
        if self.cfg.curriculum.enabled and ckpt.get("win_ante") is not None:
            self.win_ante = int(ckpt["win_ante"])
            self.env.set_win_ante(self.win_ante)
            # Re-reset so the restored win_ante applies to every env NOW
            # (set_win_ante alone only lands at each env's next auto-reset,
            # which would leave the first episodes on the ctor goalpost).
            seeds = [self.cfg.env.seed + i for i in range(self.cfg.env.num_envs)]
            self.obs, self.masks = self.env.reset(seeds)
            every = self.cfg.curriculum.eval_every_steps
            self._next_promo_eval = (self.global_step // every + 1) * every
            every = self.cfg.curriculum.milestone_every_steps
            self._next_milestone_eval = (self.global_step // every + 1) * every
        # Snapshot fraction follows the restored global step's schedule.
        self._apply_snapshot_fraction()
        self._apply_shaping_beta()
        # The lr ramp is the one schedule that is not a pure function of
        # global_step: extending total_timesteps to buy more training also
        # rescales it, so a resume can silently jump lr by orders of
        # magnitude. Report both sides of the discontinuity.
        if self.cfg.ppo.anneal_lr:
            saved = ckpt.get("config", {}).get("ppo", {})
            prev_total = saved.get("lr_anneal_total") or saved.get("total_timesteps")
            cur = self.cfg.ppo
            total = cur.lr_anneal_total or cur.total_timesteps
            lr = self.lr_at(self.global_step)
            print(f"[lr] resume at {lr:.3e} (anneal over {total:,} steps)", flush=True)
            if prev_total and prev_total != total:
                prev_lr = max(1.0 - self.global_step / max(prev_total, 1), 0.0) * (
                    saved.get("lr", cur.lr)
                )
                print(
                    f"[lr] WARNING: the checkpoint annealed over {prev_total:,} "
                    f"steps ({prev_lr:.3e} here) — the ramp was rescaled. Pin "
                    f"ppo.lr_anneal_total to keep it continuous.",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--device", default=None, help="override cfg.device")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--run-name", default=None, help="override cfg.log.run_name")
    parser.add_argument("--resume", default=None, metavar="CKPT",
                        help="resume from a ppo.py checkpoint (.pt): restores "
                             "policy + optimizer + return normalizer + "
                             "global step + curriculum win_ante")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device:
        cfg.device = args.device
    if args.total_timesteps:
        cfg.ppo.total_timesteps = args.total_timesteps
    if args.run_name:
        cfg.log.run_name = args.run_name

    trainer = PPOTrainer(cfg)
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"resumed {args.resume} at step {trainer.global_step:,} "
              f"(win_ante {trainer.win_ante})", flush=True)
    print(f"policy params: {trainer.policy.num_params():,}  device: {trainer.device}")
    trainer.train()


if __name__ == "__main__":
    main()
