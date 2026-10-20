"""Generate the Kaggle notebook that set up, benchmarked and trained m12.

    python train/kaggle/make_notebook.py <out.ipynb>

Push it with `kaggle kernels push` next to a kernel-metadata.json that attaches the
private dataset (source snapshot + prebuilt balatro_sim wheel + resume checkpoint).
See train/README.md "Training on Kaggle".
"""
import json
import sys

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "id": f"cell-{len(cells)}", "metadata": {}, "source": src})

def code(src):
    cells.append({"cell_type": "code", "id": f"cell-{len(cells)}", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src})

md("""# Gambit m12 on Kaggle

Inputs: the private `gambit-bundle` dataset (source snapshot, prebuilt `balatro_sim`
wheel, and the m11b checkpoint).

**Setup is offline and takes seconds** — no internet needed and nothing downloaded:
torch/numpy/yaml/tensorboard come with Kaggle's image, the Rust binding installs from the
prebuilt wheel in the dataset, and the trainer runs from its source tree. So start
directly on the RTX PRO 6000; there is no separate setup machine to switch away from.
(Only a changed `sim/` source needs Internet, to rebuild the binding with Rust.)

**Benchmark** — GPU RTX PRO 6000, `RUN_TRAINING = False`, Run All. Reports rollout
steps/sec at several `num_envs` so `OVERRIDES` can be tuned.

**First training session** — GPU **RTX PRO 6000** (bf16 required; T4/P100 are refused),
set `RUN_TRAINING = True` and `FIRST_SESSION = True`, then **Save Version → Save & Run
All**. It resumes from the bundle's m11b seed. Training writes `runs/m12/STOP` after
`STOP_AFTER_HOURS`, so it checkpoints and exits cleanly before Kaggle's 12 h limit.

**Every later session** — Add Input → this notebook's output from the previous
*training* version, set `FIRST_SESSION = False`, `RUN_TRAINING = True`, Save & Run All.
The newest `runs/m12/ckpt_*.pt` across all inputs is resumed automatically. If no m12
checkpoint is attached the run stops within seconds rather than retraining from the seed.

Pull results locally for the monitor: `kaggle kernels output m0k5hg/gambit-m12 -p <dir>`
""")

code("""# ===== SETTINGS =====
RUN_TRAINING     = False   # True only for a training session (Save Version -> Save & Run All)
FIRST_SESSION    = False   # True ONLY for m12's first session: allows starting from the bundle's
                           # m11b seed. Later sessions must find a runs/m12 checkpoint in the
                           # attached previous output, so a forgotten input fails fast instead of
                           # silently retraining from the seed for 11 h.
RUN_TESTS        = False   # full trainer suite (~2 min); passed on Kaggle, re-run after code changes
RUN_BENCHMARK    = False   # rollout steps/sec per num_envs; needs a GPU; skipped when RUN_TRAINING
CONFIG           = "configs/m12.yaml"
RUN_NAME         = "m12"

# Kaggle cadence, deep-merged into CONFIG. m12.yaml's cadences were sized for the
# 5090 at ~6.6k steps/s. The RTX PRO 6000 benchmarked ~150k steps/s rollout-only
# (GPU-bound: the env was only 25-32% of the time, and 4096 -> 16384 envs gained 4%),
# so num_envs stays 4096 and the step-based cadences are stretched to keep their
# wall-clock spacing sane at an estimated ~50k steps/s of full training:
#   milestone 50M -> 200M  the 15-deck matrix (30 eval calls) every ~70 min, not ~15
#   promotion 25M -> 50M   inert at the top rung (ante 8); kept as a cheap heartbeat
#   save 20 -> 100 iters   ~52M steps between checkpoints, ~35 per session (~1.2 GB)
# >>> After the first session, retune these from the real `sps` in the log.
OVERRIDES = {
    "curriculum": {"milestone_every_steps": 200_000_000, "eval_every_steps": 50_000_000},
    "log": {"save_every_iters": 100},
}
REQUIRE_TORCH    = "2.10."  # training refuses other torch builds: 2.10 (pinned image) is
                           # the verified one; the 2.11 "latest" image is untested for m12
RESUME_FROM      = None    # force a checkpoint path; None = newest one found
STOP_AFTER_HOURS = 11.0    # graceful stop before the 12 h session limit
KEEP_CKPTS       = None    # None keeps every checkpoint; an int keeps only the newest N
BENCH_ENVS       = [4096, 8192, 16384]
BENCH_SECONDS    = 45
""")

code("""import os, sys, glob, re, shutil, subprocess, hashlib, time, threading, pathlib

WORK = "/kaggle/working/gambit"
WHEELHOUSE = "/kaggle/working/wheelhouse"

def sh(cmd, cwd=None, env=None, check=True):
    print("$", cmd, flush=True)
    rc = subprocess.run(cmd, shell=True, cwd=cwd, env=env).returncode
    if check and rc:
        raise RuntimeError(f"command failed ({rc}): {cmd}")
    return rc

def inputs(pattern):
    return sorted(glob.glob(f"/kaggle/input/**/{pattern}", recursive=True))

# Bundle source: Kaggle auto-extracts the uploaded tarball to gambit_src/.
src = [os.path.dirname(os.path.dirname(p)) for p in inputs("gambit_src/sim/Cargo.lock")]
assert src, "gambit-bundle not attached (Add Input -> Your Datasets -> gambit-bundle)"
SRC = src[0]
manifest = inputs("MANIFEST.txt")
BUNDLE_ROOT = os.path.dirname(manifest[0]) if manifest else os.path.dirname(SRC)
print("source :", SRC)
if manifest:
    print(open(manifest[0]).read())

print(subprocess.run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                     shell=True, capture_output=True, text=True).stdout.strip() or "GPU: none")
cpu_max = open("/sys/fs/cgroup/cpu.max").read().split() if os.path.exists("/sys/fs/cgroup/cpu.max") else []
print("cpu    :", os.cpu_count(), "logical,",
      f"cgroup quota {int(cpu_max[0]) / int(cpu_max[1]):.0f} cores" if cpu_max and cpu_max[0] != "max" else "no cgroup quota")
""")

code("""# Source tree (input is read-only; tests and training write into it).
if not os.path.exists(f"{WORK}/train/balatro_train"):
    shutil.copytree(SRC, WORK)

def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(pathlib.Path(root).rglob("*")):
        if p.is_file() and "target" not in p.parts and "__pycache__" not in p.parts:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]

SIM_HASH = tree_hash(f"{WORK}/sim")
print("sim source hash:", SIM_HASH)

def binding_ok():
    return subprocess.run([sys.executable, "-c", "import balatro_sim"]).returncode == 0

# Reuse a wheel built from this exact sim source (from a previous output or this session).
cached = inputs(f"wheelhouse/{SIM_HASH}/*.whl") + glob.glob(f"{WHEELHOUSE}/{SIM_HASH}/*.whl")
installed = False
for whl in cached:
    if sh(f"{sys.executable} -m pip install -q --force-reinstall --no-deps {whl}", check=False) == 0 and binding_ok():
        print("using cached binding:", whl)
        installed = True
        break

if not installed:
    import urllib.request
    try:
        urllib.request.urlopen("https://sh.rustup.rs", timeout=8)
    except Exception:
        raise RuntimeError(
            f"No prebuilt balatro_sim wheel matches this sim source (hash {SIM_HASH}) and "
            "Internet is off, so it cannot be rebuilt. Turn Internet on for one session "
            "(the rebuilt wheel lands in this notebook's output under wheelhouse/), or add "
            "a matching wheel to gambit-bundle.")
    cargo_bin = os.path.expanduser("~/.cargo/bin")
    if not os.path.exists(f"{cargo_bin}/cargo"):
        sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable")
    os.environ["PATH"] = cargo_bin + os.pathsep + os.environ["PATH"]
    sh(f"{sys.executable} -m pip install -q 'maturin>=1.5,<2'")
    out = f"{WHEELHOUSE}/{SIM_HASH}"
    os.makedirs(out, exist_ok=True)
    sh(f"maturin build --release -i {sys.executable} -o {out}", cwd=f"{WORK}/sim/py",
       env={**os.environ, "CARGO_TARGET_DIR": "/tmp/cargo-target"})
    whl = sorted(glob.glob(f"{out}/*.whl"))[-1]
    sh(f"{sys.executable} -m pip install -q --force-reinstall --no-deps {whl}")
    assert binding_ok(), "balatro_sim failed to import after build"
    print("built binding:", whl)
""")

code("""# Trainer runs from its source tree: no pip install, so no network, and Kaggle's CUDA
# torch is never touched. Training and pytest run as `python -m` with cwd=train/,
# which puts train/ on their path as well.
TRAIN = f"{WORK}/train"
sys.path.insert(0, TRAIN)
os.environ["PYTHONPATH"] = TRAIN + os.pathsep + os.environ.get("PYTHONPATH", "")
missing = [mod for mod in ("torch", "numpy", "yaml", "tensorboard")
           if subprocess.run([sys.executable, "-c", f"import {mod}"]).returncode]
assert not missing, f"not in this Kaggle image (pip install needs Internet): {missing}"

import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "| bf16", torch.cuda.is_available() and torch.cuda.is_bf16_supported())
""")

code("""import yaml
from balatro_train.config import load_config
from balatro_train.policy import BalatroPolicy

if RUN_TESTS and not RUN_TRAINING:
    sh(f"{sys.executable} -m pytest tests -q -x -p no:cacheprovider", cwd=f"{WORK}/train")

# Effective config = CONFIG with OVERRIDES deep-merged, written next to it.
def merge(base, over):
    for k, v in over.items():
        base[k] = merge(base.get(k, {}), v) if isinstance(v, dict) else v
    return base

with open(f"{WORK}/train/{CONFIG}") as f:
    cfg_dict = merge(yaml.safe_load(f), OVERRIDES)
assert cfg_dict["log"]["run_name"] == RUN_NAME, "RUN_NAME must match the config's log.run_name"
EFFECTIVE = CONFIG.replace(".yaml", "_kaggle.yaml")
with open(f"{WORK}/train/{EFFECTIVE}", "w") as f:
    yaml.safe_dump(cfg_dict, f, sort_keys=False)
cfg = load_config(f"{WORK}/train/{EFFECTIVE}")

# Newest checkpoint: this run's own (any attached output, or this session) or the bundle seed.
def ckpt_step(p):
    m = re.search(r"ckpt_(\\d+)\\.pt$", p)
    return int(m.group(1)) if m else -1

cands = [p for p in inputs("ckpt_*.pt")
         if f"/runs/{RUN_NAME}/" in p or os.path.dirname(p) == BUNDLE_ROOT]
cands += glob.glob(f"{WORK}/train/runs/{RUN_NAME}/ckpt_*.pt")
RESUME = RESUME_FROM or max(cands, key=ckpt_step)
FROM_SEED = RESUME_FROM is None and os.path.dirname(RESUME) == BUNDLE_ROOT

ck = torch.load(RESUME, map_location="cpu", weights_only=False)
BalatroPolicy(cfg.policy).load_state_dict(ck["policy"])   # raises on any shape mismatch
assert ck.get("win_ante") in cfg.curriculum.ladder, "restored win_ante missing from ladder"
batch = cfg.env.num_envs * cfg.ppo.rollout_len
assert batch % cfg.ppo.minibatch_size == 0, f"minibatch_size must divide {batch}"
print(f"resume : {RESUME}  (step {ck['global_step']:,}, win_ante {ck.get('win_ante')})"
      + ("  <- bundle seed: no runs/" + RUN_NAME + " checkpoint attached" if FROM_SEED else ""))
print(f"config : {EFFECTIVE}  num_envs={cfg.env.num_envs}  minibatch={cfg.ppo.minibatch_size}"
      f"  decks={'on' if cfg.deck_curriculum.enabled else 'off'}")
print("SETUP OK")
""")

code("""if RUN_BENCHMARK and not RUN_TRAINING:
    if not torch.cuda.is_available():
        print("benchmark skipped: no GPU attached")
    else:
        from balatro_train.config import TrainConfig, _build
        from balatro_train.env_api import make_vec_env
        from balatro_train.policy import actions_to_numpy, obs_to_torch

        dev = torch.device("cuda")
        bf16 = torch.cuda.is_bf16_supported()
        if not bf16:
            print("note: no bf16 on this GPU -- numbers are NOT representative of training")
        net = BalatroPolicy(_build(TrainConfig, ck["config"], "ckpt").policy).to(dev).eval()
        net.load_state_dict(ck["policy"])

        def bench(n, seconds, warmup=8):
            env = make_vec_env("sim", n)
            env.set_deck_distribution([1.0] * 15)
            obs, masks = env.reset(list(range(n)))
            steps = 0
            t_pol = t_env = 0.0
            t0 = None
            while True:
                a0 = time.time()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                    acts, *_ = net.act(obs_to_torch(obs, dev), obs_to_torch(masks, dev),
                                       need_entropy=False)
                acts = actions_to_numpy(acts)          # device->host copy syncs the GPU
                a1 = time.time()
                obs, masks, *_ = env.step(acts)
                a2 = time.time()
                steps += 1
                if steps == warmup:
                    t0, s0, t_pol, t_env = time.time(), steps, 0.0, 0.0
                elif t0 is not None:
                    t_pol += a1 - a0
                    t_env += a2 - a1
                    if time.time() - t0 >= seconds:
                        break
            return n * (steps - s0) / (time.time() - t0), t_env / (t_env + t_pol)

        print(f"{'num_envs':>9} {'steps/s':>9} {'env share':>10}")
        for n in BENCH_ENVS:
            sps, env_share = bench(n, BENCH_SECONDS)
            print(f"{n:>9} {sps:>9.0f} {env_share:>9.0%}", flush=True)
        print("rollout only (no PPO update). env share = fraction of time in the Rust "
              "env.step; high share means CPU-bound, so more envs help only if cores are idle.")
""")

code("""if RUN_TRAINING:
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), \\
        "m12 needs a bf16 GPU -- select GPU RTX PRO 6000 (T4/P100 cannot run bf16)"
    if REQUIRE_TORCH and not torch.__version__.startswith(REQUIRE_TORCH):
        raise RuntimeError(
            f"torch {torch.__version__} but training is verified on {REQUIRE_TORCH}x. In the "
            "editor: Settings -> Environment -> Pin to original environment (or set "
            "REQUIRE_TORCH = None to accept this build).")
    if FROM_SEED and not FIRST_SESSION:
        raise RuntimeError(
            f"No runs/{RUN_NAME} checkpoint among the inputs, so this would restart from the "
            "bundle's m11b seed. Add Input -> this notebook's previous training output, or set "
            "FIRST_SESSION = True if this really is the first session.")
    run_dir = f"{WORK}/train/runs/{RUN_NAME}"
    os.makedirs(run_dir, exist_ok=True)
    stop_path = os.path.join(run_dir, "STOP")

    def request_stop():
        time.sleep(STOP_AFTER_HOURS * 3600)
        with open(stop_path, "w") as f:
            f.write("kaggle session time limit")
        print(f"\\n[notebook] STOP requested after {STOP_AFTER_HOURS} h", flush=True)

    threading.Thread(target=request_stop, daemon=True).start()
    cmd = [sys.executable, "-u", "-m", "balatro_train.ppo", "--config", EFFECTIVE, "--resume", RESUME]
    print("$", " ".join(cmd), flush=True)
    with open(f"{run_dir}/train.log", "a") as log:
        p = subprocess.Popen(cmd, cwd=f"{WORK}/train", stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
        rc = p.wait()

    ckpts = sorted(glob.glob(f"{run_dir}/ckpt_*.pt"), key=ckpt_step)
    if KEEP_CKPTS:
        for old in ckpts[:-KEEP_CKPTS]:
            os.remove(old)
        ckpts = ckpts[-KEEP_CKPTS:]
    print(f"\\ntrainer exited with {rc}; {len(ckpts)} checkpoints in output, newest:",
          ckpts[-1] if ckpts else "none")
    if rc != 0:
        raise RuntimeError(f"trainer exited with {rc} -- see the [diverged] line or traceback "
                           "above. Checkpoints already written are kept in the output.")
else:
    print("RUN_TRAINING is False -- set up and verified only.")
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(sys.argv[1], "w", encoding="utf-8", newline="\n") as f:
    json.dump(nb, f, indent=1)
print("wrote", sys.argv[1], "with", len(cells), "cells")
