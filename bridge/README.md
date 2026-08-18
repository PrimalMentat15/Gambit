# bridge — real-game bridge (balatrobot)

Connects our tooling to the real Balatro game (Steam + Proton) through the
[coder/balatrobot](https://github.com/coder/balatrobot) mod (MIT), which serves
a **JSON-RPC 2.0 HTTP API** on `http://127.0.0.1:12346`.

Docs: <https://coder.github.io/balatrobot/latest/> (install / CLI / API pages;
full text also in `llms-full.txt` on that site).

## What is installed (P4 setup record, 2026-07-15)

| Component | Version | Location |
| --- | --- | --- |
| balatrobot Lua mod | **v1.5.2** (git tag, commit `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`) | `~/.local/share/Balatro/Mods/balatrobot/` |
| balatrobot Python CLI/client | **1.5.2** (PyPI `balatrobot`, pinned `>=1.5.2,<1.6`) | the `balatro` conda env |
| Steamodded (pre-existing) | 1.0.0-beta-1814a | `~/.local/share/Balatro/Mods/smods-1.0.0-beta-1814a/` |
| Lovely injector (pre-existing) | 0.9.0 (`version.dll`) | `~/.local/share/Steam/steamapps/common/Balatro/version.dll` |

The mod was installed per the official install docs (copy `balatrobot.json`,
`balatrobot.lua`, `src/lua` into the Mods dir), from a clone of the repo at tag
`v1.5.2` — the GitHub release ships no separate mod zip. See
`~/.local/share/Balatro/Mods/balatrobot/VERSION.txt` for provenance. Note the
upstream `balatrobot.json` manifest still says `"version": "1.5.1"` (their
manifest lags the release tag); the code is the v1.5.2 tag.

### Compatibility verdict

* **Steamodded: compatible, no upgrade needed.** balatrobot requires
  Steamodded `v1.0.0-beta-1221a+` (install docs) and declares
  `"Steamodded (>=1.~)"` in its manifest. Installed beta-1814a > beta-1221a.
* **Lovely: compatible.** Requires v0.8.0+, installed 0.9.0.
* **Balatro:** requires v1.0.1+ (Steam copy is current).
* **DebugPlus** (v1.5.1+) is *not* installed. It is only needed for
  `balatrobot serve --debug` and upstream's test-only endpoints — not for
  normal bot play. Install it later only if we need those.

### Mods directory note (Proton)

The balatrobot docs give the Proton mods path as
`~/.local/share/Steam/steamapps/compatdata/2379780/pfx/drive_c/users/steamuser/AppData/Roaming/Balatro/Mods`.
On this machine that path is a symlink chain:

```
compatdata/.../AppData/Roaming/Balatro/Mods -> ~/.config/Balatro/Mods -> ~/.local/share/Balatro/Mods
```

so installing into `~/.local/share/Balatro/Mods/` is exactly the documented
location. (Set up by Balatro Mod Manager; BMM compat mode itself is disabled.)

## Windows (native, no Proton)

The bridge runs on Windows without Steam or Proton. Paths and the link strategy
switch automatically; what differs from the Linux flow:

| | Linux (Proton) | Windows (native) |
|---|---|---|
| Mods dir | `~/.local/share/Balatro/Mods` | `%APPDATA%\Balatro\Mods` |
| Save dir | Steam `compatdata` Wine prefix | `%APPDATA%\Balatro` |
| `LOVELY_MOD_DIR` | `Z:`-mapped Wine path | plain Windows path |
| Profile links | symlinks | symlinks, else directory junctions |
| Stopping `serve` | SIGINT | `terminate()` |

Directory *symlinks* on Windows need Developer Mode or an elevated shell;
*junctions* need neither and work identically here, so the launcher tries a
symlink and falls back rather than demanding elevation.

### Prerequisites

1. **Lovely injector** — `version.dll` beside `Balatro.exe`.
2. **Steamodded** → `%APPDATA%\Balatro\Mods\smods\`
3. **balatrobot v1.5.2** (must match the pinned PyPI client) →
   `%APPDATA%\Balatro\Mods\balatrobot\`

The launcher builds its isolated profile from the last two, so it fails with a
named error if either is missing rather than launching an unmodded game.

**Extracting a release zip is fine either way.** A GitHub release zip expands
to `Mods/smods/smods-1.0.0-beta-1814a/` rather than `Mods/smods/` directly —
one folder level deeper than Lovely (`Mods/<mod>/lovely/*.toml`) and Steamodded
(`Mods/<mod>/manifest.json`) expect. Left alone, both scanners find nothing at
that extra level, load nothing, and Balatro starts completely unmodded with
**no error** — the only symptom is the API health check timing out 30s later,
which looks like a networking problem, not a mod-loading one. The launcher
detects this wrapper layout and links straight to the inner directory, so
extracting a zip as-is or renaming the inner folder both work; nothing to do
here. If you ever hit the same silent-unmodded symptom some other way, the
tell is `bridge/profiles/vanilla/Mods/lovely/log/<latest>.log`: a healthy run
lists loaded patch files, a broken one says `Initialization complete` with
nothing above it.

### Environment

`balatrobot serve` finds Steam, Proton and the game by itself on Linux. Native
Windows has none of that to go on, so point it at the executable:

```bash
set BALATRO_EXE=C:\path\to\Balatro.exe
```

`--platform windows` is added automatically, and `--lovely-path` defaults to
`version.dll` beside that executable.

Note the flag is `--love-path`, not the more obvious `--balatro-path`:
balatrobot's Windows launcher validates and launches `config.love_path`, and
falls back to a hardcoded Steam location when it is unset. Passing only
`--balatro-path` therefore fails with `Balatro executable not found:
C:\Program Files (x86)\Steam\...` no matter how correct your path was. The
launcher here passes the right one.

Overrides, all optional:

| Variable | Purpose |
|---|---|
| `BALATRO_EXE` | Path to `Balatro.exe` |
| `LOVELY_DLL` | Path to `version.dll` (default: beside the exe) |
| `BALATRO_MODS_DIR` | Override the user Mods dir |
| `BALATRO_SAVE_DIR` | Override the love.filesystem save dir |

Then launch as usual — `--gamespeed 4` (the default) is the pace to watch at:

```bash
balatro-bridge-launch
```

## Running a "vanilla + balatrobot only" instance (for cross-validation)

The user Mods dir also contains content mods (Cryptid, Talisman, Blueprint,
Cartomancer, Amulet, HandyBalatro, JokerDisplay, BMM-Compat) that alter game
behavior and would poison sim-vs-real validation. They are left untouched.
We isolate them **per launch, non-destructively** via a separate mods
directory:

### Mechanism (verified against source)

1. **Lovely honors `LOVELY_MOD_DIR`.** `crates/lovely-core/src/lib.rs` in
   [ethangreen-dev/lovely-injector](https://github.com/ethangreen-dev/lovely-injector)
   checks the env var before falling back to the default dir:
   `if let Some(env_path) = env::var_os("LOVELY_MOD_DIR") { PathBuf::from(env_path) }`
   (it also supports `--mod-dir` and `--vanilla` game args, but env vars are
   easiest to pass through Proton).
2. **Steamodded takes its mod dir from lovely**, so the override cascades:
   `smods src/preflight/core.lua` does
   `local lovely_mod_dir = lovely.mod_dir` … `SMODS.MODS_DIR = lovely_mod_dir`
   (verified in the installed smods-1.0.0-beta-1814a).
3. **Env vars reach the game under Proton.** `balatrobot serve` launches
   `proton run Balatro.exe` as a child process with a copy of the caller's
   environment (upstream `src/balatrobot/platforms/linux.py`), and the mod
   already reads its own `BALATROBOT_*` config the same way (`os.getenv` in
   `src/lua/settings.lua`) — same passthrough path.
4. **Path form:** lovely is a Windows DLL under Proton, so the profile path is
   passed as `Z:<unix path>` (Proton maps the host filesystem root to `Z:`).
   `launch.py` does this automatically.

`launch.py` builds `bridge/profiles/vanilla/Mods/` containing **symlinks** to
only `smods-…` and `balatrobot`, then sets `LOVELY_MOD_DIR` to it. Wine follows
host symlinks (the existing Mods chain above already relies on this). Nothing
in `~/.local/share/Balatro/Mods/` is created, deleted, or modified.

**Verify on first launch:** the lovely log
(`bridge/profiles/vanilla/Mods/lovely/log/lovely-*.log`) must say
`Using mod directory at "Z:/home/teq/Desktop/balatroagent/bridge/profiles/vanilla/Mods"`
and list no content mods. (If `LOVELY_MOD_DIR` were somehow not picked up, the
log would show the `C:\users\steamuser\AppData\Roaming\Balatro\Mods` default —
in that case fall back to the alternatives below.)

### Alternatives considered (not used)

* **`Mods/lovely/blacklist.txt`** — lovely 0.9 skips listed mod dirs and smods
  reads the same file (`src/preflight/loader.lua`). It works (this machine's
  log shows `'Cryptid' was found in blacklist, skipping it` — the user already
  blacklists Cryptid/Talisman for normal play), but it is *global mutable
  state* shared with the user's own sessions, not a per-launch profile.
* **`.lovelyignore` file inside each mod folder** — what Steamodded's in-game
  mod toggle writes (`smods src/ui.lua`); smods skips any mod dir containing
  it (`src/preflight/loader.lua`). Non-fatal but it *writes into the content
  mods' folders* and is also global.
* **lovely `--vanilla` game arg** — disables *all* mods including balatrobot;
  useful someday for pure-vanilla checks, not for the bot.

## Python package (`balatro_bridge`)

project; upstream ships a Python package (PyPI `balatrobot`) containing the
launcher CLI and a generic JSON-RPC client (`balatrobot.BalatroClient`), so we
**depend on it** rather than reimplement transport. Our adapter layer adds:

* `balatro_bridge.client.BalatroBridgeClient` — one typed method per API
  endpoint (all 21 methods + `rpc.discover`):
  `health`, `gamestate`, `rpc_discover`, `start`, `menu`, `save`, `load`,
  `select`, `skip`, `play`, `discard`, `rearrange`, `use`, `cash_out`,
  `next_round`, `reroll`, `buy`, `pack`, `sell`, `add`, `set`, `screenshot`;
  plus `wait_until_ready()` and a raw `call()` escape hatch. "Exactly one of"
  parameter groups (`buy`, `sell`, `rearrange`, `pack`) are validated
  client-side. API errors raise `balatro_bridge.APIError`
  (re-exported upstream class: `.name`, `.code`, `.message`).
* `balatro_bridge.types` — `Literal` enums (`Deck`, `Stake`, `State`, …) and
  `TypedDict`s (`GameState`, `Card`, `Area`, …) matching the OpenRPC spec.
* `balatro_bridge.launch` (`balatro-bridge-launch`) — vanilla profile +
  `balatrobot serve` wrapper (see above).
* `balatro_bridge.smoke` (`balatro-bridge-smoke`) — end-to-end smoke test.
* `balatro_bridge.demo` (`balatro-bridge-demo`) — the trained policy plays
  the live game end-to-end (see "Live demo" below).

Setup:

```bash
cd bridge
conda activate balatro
```

## Live demo (trained policy plays the real game)

Terminal 1 — launch the game window with the RPC mod:

```bash
cd bridge
# deps already in the conda env (see repo README "Environment")
balatro-bridge-launch           # leave running
```

Terminal 2 — run the demo:

```bash
cd bridge
balatro-bridge-demo --ckpt ../train/runs/cloud_mirror/m5e2/ckpt_4000317440.pt
```

The policy (argmax) picks every action from the Rust sim's encoded state and
each action is mirrored onto the live game; the full state is diffed sim-vs-
game after every action (P5 machinery), so the demo doubles as deep
cross-validation — divergences are reported but don't stop the run. With no
`--seed` it first pre-screens random seeds in the sim (~1s each) and plays
one the policy wins, so the audience gets a full ante-1→8 victory; the
certified policy wins ~71% of unscreened seeds. Same policy path exists in
the harness proper: `balatro-crossval --policy ckpt --ckpt <pt>`.

Timing note: cash-outs deliberately pause ~2–8s — pressing balatrobot's
`cash_out` while the round-eval rows still animate pays a stale
`G.GAME.current_round.dollars` (previous round's total; found live, seed
PQD3RI3I). `GameDriver` sizes the wait from the sim's expected payout.

### In-game BOT RUN button

`mods/BotRun/` is a tiny Steamodded mod (auto-linked into the vanilla
profile by `balatro-bridge-launch`) that adds a green **BOT RUN** button to
the main menu. It writes a request marker via love.filesystem; the autoplay
daemon answers it:

```bash
balatro-bridge-autoplay --ckpt ../train/runs/cloud_mirror/m5e2/ckpt_4000317440.pt
```

Leave the daemon running and click BOT RUN whenever — each click starts a
fresh **random-seed** Red Deck / White Stake run played by the policy
(~71% of them are wins). Clicks with no daemon running are harmless no-ops;
clicks while a run is playing are ignored.

The panel also has **4X / 8X / 16X** speed buttons — the vanilla settings UI
caps game speed at 4x, but the engine takes any multiplier, so click 8X/16X
before BOT RUN to watch runs fast. Session-only (a game restart returns to
the boot default); for a persistent default launch with e.g.
`balatro-bridge-launch -- --gamespeed 16`.

Intervening in a bot-driven run (clicking cards / quitting mid-action) can
hard-crash the modded game — balatrobot mutates the hand concurrently with
user input (observed live: log ends mid-`discard`, no Lua error). Both sides
recover on their own: the daemon survives the aborted run and goes back to
waiting, and `balatro-bridge-launch` supervises the game via API health
polls and relaunches it (~30 s). Wait for the menu to come back, then click
BOT RUN again. Quitting on purpose (QUIT button / closing the window) is NOT
treated as a crash — the mod marks it from `love.quit` and the supervisor
shuts down with the game.

### Watching a run at a readable pace

The point of a live session here is to *watch* the trained policy play, so speed
is not the goal — legibility is. `--gamespeed 4` (the launcher default) matches
the fastest setting the vanilla options menu offers and is the pace to use.

```bash
balatro-bridge-launch
```

The BotRun panel's **8X / 16X** buttons exist for unattended runs and are worth
avoiding while watching: past 4x the animations blur together, and the engine
caps the real speed-up anyway. Two engine paths ignore the multiplier entirely —
`EventManager:update` runs at most one blocking event per 1/60 s of **real**
time regardless of `GAMESPEED`, and `Moveable:move` is never scaled at all — so
16x buys far less than the number suggests while making the run unwatchable.

## First live smoke test (run by a human / later phase — not automated here)

Terminal 1 — launch the game (needs a display server; Steam itself need not be
running, `balatrobot serve` drives Proton directly):

```bash
cd ~/Desktop/balatroagent/bridge
conda activate balatro                 # once
balatro-bridge-launch           # windowed first time, so you can watch
# later, for training: balatro-bridge-launch -- --headless --fast
```

Terminal 2 — once the game window is up:

```bash
cd ~/Desktop/balatroagent/bridge
balatro-bridge-smoke --wait 120
# checks health, starts seeded run (RED/WHITE, seed BRIDGE1),
# selects Small Blind, plays one hand, prints the gamestate summary.
# add --full to dump the entire gamestate JSON
```

Then confirm mod isolation (one-time):

```bash
grep "Using mod directory" profiles/vanilla/Mods/lovely/log/lovely-*.log
```

Quick manual poke without the bridge (equivalent):

```bash
curl -X POST http://127.0.0.1:12346 -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "health", "id": 1}'
```

Useful `serve` flags (CLI reference): `--fast` (10x speed), `--headless`,
`--render-on-api` (mutually exclusive with `--headless`), `--gamespeed N`,
`--port N` / `BALATROBOT_PORT`, `--no-shaders`. Audio is off by default.

## P5 cross-validation harness (`balatro-crossval`)

Drives the SAME seeded run in the Rust sim (`balatro_sim.CrossvalRun`,
direct Run-level interface added in P5) and the live game, applies a
deterministic scripted policy chosen from the SIM's legal-action set,
mirrors each action onto balatrobot, and diffs the full normalized state
after every action. Divergences dump repro bundles to
`tools/crossval/reports/<seed>_<step>.json`; progress checkpoints to
`tools/crossval/reports/progress.json` (reruns resume; `--redo` overrides).

```bash
cd bridge
pip install ./sim/py                  # builds sim/py via maturin
# sanity (no game needed): two sim instances must diff clean
balatro-crossval --dry-run --seeds 3 --policy bot --max-antes 8
# live (game running under balatro-bridge-launch):
balatro-crossval --seeds 5 --policy bot   --max-antes 3
balatro-crossval --seeds 5 --policy mixed --max-antes 3   # bot + 30% random
balatro-crossval --seeds 5 --policy random --max-antes 2
# flags: --seed-list A,B  --max-steps N  --continue-on-divergence --redo
```

Schema mapping, normalization rules and the known SMODS-beta scoring quirk
are documented in `src/balatro_bridge/crossval.py` (module docstring) and
`sim/py/src/export.rs`.
