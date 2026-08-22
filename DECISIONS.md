# Engineering Decisions Log

A chronological record of the non-obvious calls made on this project: what was
tried, what was measured, what was rejected and why. The goal is that six months
from now, nobody — including whoever is reading this — has to re-derive a
decision that was already made and already had evidence behind it.

**Format:** each entry states the decision, the reasoning, and — where one
exists — the alternative that was considered and rejected. Numbers are real
measurements from this repo's own runs unless marked otherwise. Entries are
grouped by phase and dated where a date is known; early Phase 0/1 entries were
squashed into a single initial commit during a repo rebase (see D009), so their
internal ordering is approximate.

---

## Phase 0 — Environment parity

Goal: reproduce the original [angelvalentin80/balatro-rl](https://github.com/angelvalentin80/balatro-rl)
project's environment and model architecture on Windows, since the original was
built and tested on Linux.

### D001 — Named pipes → localhost TCP socket

**Decision:** replace Unix named pipes (`os.mkfifo`, `/tmp/balatro_request`,
`/tmp/balatro_response`) with a single localhost TCP connection, line-delimited
JSON, `TCP_NODELAY` on both ends. Python is the server (binds and `accept()`s);
the Lua mod is the client.

**Why:** `os.mkfifo` doesn't exist on Windows, so the original transport couldn't
run here at all. Two other options were considered:

- **Windows Named Pipes** — the native equivalent, but needs `pywin32` plus
  hand-rolled message framing, and doesn't port to Linux/macOS without a second
  code path.
- **Shared file + polling** — no new dependency, but polling adds latency to
  every single request/response, which matters at the step rates this project
  cares about.

Sockets won because they need **zero new dependencies on either side**: Python's
stdlib `socket`, and LuaSocket is already compiled into Balatro's `love.dll`
(confirmed by finding `luaopen_socket_core` inside the shipped DLL) — so
`require("socket")` just works with no mod-side dependency added. Sockets also
keep the code path identical across Windows, Linux and macOS, unlike named pipes.

**Behaviour carried over:** the mod connects lazily (only on the first action
request, i.e. after the training key is pressed or autostart fires), and retries
at most once per second if the trainer isn't listening yet, so a missing trainer
never stalls the game.

### D002 — CUDA setup, with expectations set correctly up front

**Decision:** wire explicit CUDA device selection through to `MaskablePPO`
(`get_device()` in `ai/train_balatro.py`), pinned to the cu126 wheel index.

**Why cu126 over cu130:** both ship a matching wheel for this Python/OS
combination; cu126 was chosen as the more battle-tested build for this GPU
generation.

**Important caveat documented at the time, still true:** the GPU was not expected
to meaningfully speed up training, and it hasn't needed to — the bottleneck has
always been the live game's round-trip time, not the policy network's forward
pass on a small `MlpPolicy` over a 216-dim observation. SB3 itself warns that PPO
with `MlpPolicy` is often faster on CPU. The GPU only starts to matter once the
network is scaled up or multiple environments run in parallel — neither of which
has happened yet.

**Verified:** `torch.cuda.is_available()` True, device reports as the expected
GPU, and a `4096×4096` matmul was run on-device as a non-trivial sanity check
(not just an import check).

---

## Phase 1 — Telemetry, speed, and the monitoring application

### D003 — Telemetry: stdlib-only, non-blocking, drop-on-full

**Decision:** a custom event stream (`ai/telemetry/`) rather than Python's
`logging` module or a metrics library — newline-delimited JSON, one file per run,
written by a background thread reading from a bounded queue.

**Why:** the training loop spends nearly all its time blocked on the game (see
D005), so the one rule that could not be compromised was that instrumentation
must never add latency or ever be able to block training. `emit()` does one dict
construction and a `queue.put_nowait()`; if the queue is full, the event is
dropped and counted, never blocked on. Measured cost: **~1.3 µs per `emit()`**,
and total instrumentation overhead in an end-to-end test came to **0.25–0.47 ms
per step against a 480 ms simulated step** (0.05–0.1%).

**Why stdlib-only:** so instrumenting the trainer adds no install burden to a
machine that only needs to train, not visualize.

### D004 — Profile before optimizing, and question the obvious explanation

**Decision (methodological, not code):** when the game turned out to run at
0.483 s/step despite `GAMESPEED = 100` and `DISABLE_ANIMATIONS = true` already
being set, the response was to build a latency-attribution tool
(`ai/tools/latency_report.py`) before touching any game logic — because the
obvious explanation ("the settings aren't being applied") turned out to be
**wrong**, and would have led to fixing the wrong thing.

**What the report found, precisely:** `t_wait` (blocked on the game) was 99.9%
of every step, with a **bimodal distribution** the mean alone hid completely
(mean 283.8 ms, median 13.8 ms, p95 1105.5 ms). Splitting by which action was
sent isolated it exactly:

| Action | n | mean wait | % of all wall time |
|---|---|---|---|
| `PLAY_HAND` | 412 | 995 ms | 73.4% |
| `DISCARD_HAND` | 410 | 340 ms | 24.9% |
| `SELECT_HAND` | 1083 | 8.7 ms | 1.7% |

This is the entry that matters most methodologically: a plausible but wrong
hypothesis ("the seed-transition animation is the remaining cost") was floated
and **rejected by the data** before code was written — episode-boundary steps
turned out to be only 14.8% of wall time, not the dominant cost.

### D005 — Root cause: `GAMESPEED` doesn't scale the event queue's own clock

**Decision:** patch `RLBridge/ai.lua` to force-drain the event queue within a
single frame, rather than continuing to raise `GAMESPEED` or disable more
animations.

**Why, with the actual mechanism:** reading Balatro's own Lua source
(`balatro-source-code/`, extracted from the shipped exe and gitignored as
reference material) found that `EventManager:update` in `engine/event.lua` is
driven by `real_dt` — wall-clock time — and rate-limits itself to completing at
most one blocking event per `1/60`s of *real* time, capping the game at ~60
blocking events per real second **no matter how high `GAMESPEED` is set**.
`GAMESPEED` does scale `G.TIMERS.TOTAL`, so delays *inside* individual events
already expire instantly — but the queue's own processing clock was never
touched by it. A ~60-event scoring sequence (`PLAY_HAND`) costs roughly one real
second for exactly this reason.

**The fix:** `drain_event_queue()` in `ai.lua` calls `G.E_MANAGER:update(dt,
true)` in a forced loop, advancing `G.TIMERS.TOTAL` in lockstep so chained
`after`-triggered delays still expire correctly. Same events, same order, same
functions — only the wall-clock spacing between them changes. It bails the
moment a pass makes no progress (meaning remaining events are waiting on
something other than time, e.g. card position — see D006), so worst case it
degrades to stock pacing rather than hanging.

**Measured result — a 4.6× speedup:**

| | ms/step | steps/sec | `PLAY_HAND` mean |
|---|---|---|---|
| Stock pacing | 296 | 3.4 | 995 ms |
| Forced event drain | **65** | **15.5** | **193 ms** |

Default on, `BALATRO_RL_DRAIN` env var to disable.

### D006 — Movement pump: tried, measured, rejected

**Decision:** implement card-movement pumping (`BALATRO_RL_PUMP`) as an
experiment, measure it, and **turn it off by default** based on the result.

**The hypothesis:** `Moveable:move` (card movement) is *also* driven by
`real_dt`, uncapped by any speed setting — the same class of bug as D005, one
layer deeper. Pumping it alongside the event drain should let position-gated
events finish instead of stalling.

**What actually happened when measured:** pumping card movement stopped the
drain loop's "no progress" bail-out from ever triggering, since the moved cards
kept `count_blocking()` decreasing turn after turn. It ran hundreds of forced
passes per step, each walking every moveable object.

| | ms/step | steps/sec | worst single step |
|---|---|---|---|
| Event drain alone | 65 | 15.5 | 298 ms |
| Event drain + movement pump | **114** | **8.8** | **13,146 ms** |

A **1.75× regression**, with one step spiking to 13 seconds. Left implemented
(`BALATRO_RL_PUMP=1` to opt in for further experiments) but off by default. This
entry is the clearest example on this project of "measure before shipping the
theoretically-motivated fix" — the mechanism reasoning was sound, the outcome
was still wrong.

### D007 — Monitor UI stack: Python/PySide6/pyqtgraph over C++/Qt or a web dashboard

**Decision:** build the monitoring application in Python with PySide6 (Qt
bindings) and pyqtgraph for plotting, not C++/native Qt, not a browser-based
dashboard as the primary interface.

**Why, with the numbers that drove it:** the measured step rate (D005) is ~2–15
steps/sec. A monitoring UI processing a few events per second is >99% idle no
matter what language it's written in — so C++ would optimize a resource budget
that was never the bottleneck, at direct cost to iteration speed (the explicitly
stated top priority: a compile/MOC/CMake cycle per tweak, and every new metric
crossing a Python↔C++ boundary). Python keeps a new panel a one-file addition
(see `monitor/panels/`), not a cross-language change.

**Why not Qt Charts/QtGraphs:** they're GPL-3.0-or-commercial, incompatible with
this project's MIT license. pyqtgraph is MIT and purpose-built for real-time
data (handles 100k+ points at interactive frame rates).

**Why not a web dashboard as the primary UI:** a browser process costs ~300MB
RAM and adds a JS toolchain on top of the Python stack, for a benefit (remote
access) that's better served by a small dedicated read-only web view alongside
the desktop app rather than replacing it (see D008's sibling — the remote view
shipped separately, stdlib-only, in Stage 5).

**Resource discipline that followed from this choice:** one shared repaint timer
at 10 Hz (not 60 — already ~5× oversampled relative to the event rate), panels
skip repainting entirely while their tab isn't visible, and every plotted series
is a bounded ring buffer decimated before drawing — needed because a 33-hour run
at the measured rate is on the order of hundreds of thousands to low millions of
points.

### D008 — Palette: validated, not eyeballed

**Decision:** every color used in the monitor (panel series, status colors,
action-identity colors) comes from a palette run through an automated
colorblind-safety validator before use, not chosen by eye.

**Why:** "these look different enough" is not a check. The palette in
`monitor/theme.py` passes the lightness band, chroma floor, CVD separation
(worst adjacent pair ΔE **9.4**, deutan simulation), normal-vision floor
(**20.9**), and ≥3:1 contrast against the dark surface — in both the adjacent
pairlist (bars, lines) and the all-pairs pairlist (scatter plots), where a
categorical palette's safety margins are much harder to hold.

**A specific, deliberate rule that followed from this:** action-identity colors
are keyed to the action's numeric id (`RLBridge/actions.lua`), not assigned by
current rank/order. `PLAY_HAND` is the same hue in every panel it appears in,
permanently — color follows the entity, never its position in a sorted list.

### D009 — Repo rebase: single root commit, renamed to Gambit

**Decision:** at the point this became the user's own independent project
(rather than a fork being actively synced with upstream), the git history was
rebased to a single clean root commit rather than carrying forward
`angelvalentin80/balatro-rl`'s ~26-commit history.

**License handling:** the original repository's MIT copyright line was
**retained** in `LICENSE` (a line was added below it, not in place of it) — MIT's
terms require the original copyright notice survive in redistributions; this is
an obligation, not a style choice. Attribution is additionally given in a
dedicated `README.md` Credits section per the project owner's explicit
instruction. The pre-rebase history was tagged (`upstream-history`) before the
rewrite so it wasn't destroyed, only made non-default.

**Why this is worth logging:** it's a case where an instruction ("just credit
them in the README") had a legal constraint underneath it (retain the copyright
notice) that wasn't optional, and both were satisfied without conflict — worth
recording so a future contributor doesn't wonder why the LICENSE file has two
names in it.

### D010 — Windows console encoding: reconfigure stdout/stderr to UTF-8 at the very top

**Date:** 2026-08-13 (discovered when the Control tab's "Start run" button first
failed).

**Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character
'\U0001f4c2'` — the trainer died the instant it tried to print an emoji, but
**only** when launched via the monitor's supervisor, never when run directly in
a terminal.

**Root cause:** Windows Python picks the system codepage (`cp1252` on this
machine) for `sys.stdout`/`sys.stderr` whenever they are **not a real console** —
which is exactly what happens when the supervisor redirects a child process's
stdout to a file. A real terminal bypasses this via the Windows console API,
which is why it never showed up in interactive testing.

**Fix:** `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (and
`stderr`) as the literal first thing that executes in `ai/train_balatro.py` —
placed before even the SB3/gymnasium imports, since those can print their own
warnings on import. Also explicitly passed `encoding="utf-8"` to the
`logging.FileHandler` for `training.log`, which had the identical latent bug.

**Verified with a minimal repro** matching the supervisor's exact subprocess
pattern (stdout redirected to an opened file handle) before and after the fix,
rather than trusting that "it works now" from a single successful run.

### D011 — Stop vs Kill: two mechanisms, not one, because cooperative stop cannot always work

**Decision:** the monitor's kill switch is **two independent controls** — a
cooperative "Stop" and a terminate-based "Kill" — not one button that tries to
be graceful and falls back to force.

**Why cooperative alone is insufficient, precisely:** `StopRequestCallback`
implements Stop via a `STOP` file the trainer checks between SB3 steps (so
`model.learn()` exits normally and still checkpoints). But the trainer spends
nearly all of its time **blocked inside a socket read** waiting on the game
(D005's whole premise). Python on Windows will not interrupt a thread blocked in
a socket read, so if Balatro hangs or deadlocks, the trainer never reaches a
step boundary and never reads the stop flag. A purely cooperative design would
have no way to end a run in exactly the scenario where ending it matters most.

**Kill is deliberately unconditional:** `taskkill /T /F` on the whole process
tree, and it **always** also closes Balatro — a half-dead trainer/game pair
holding the listening port is precisely what would block the next run from
starting.

**Both work without the GUI** (`python -m ai.tools.killrun --stop|--kill`),
sharing the exact same process-control code (`ai/tools/procs.py`) as the
toolbar buttons, specifically so the two paths cannot silently drift apart.

**Verified against a process that ignores `SIGINT`** (not just a normal one) —
the case Kill exists for — confirming Stop has no effect on it and Kill
terminates it regardless.

### D012 — Card-selection clamping: the 15.3%-wasted-steps fix

**Date:** 2026-08-13.

**Finding:** in a completed run (`2026-08-12_203900_run`), **4,619 of 30,227
steps (15.3%) had `retry_count > 0`**, and every single one was `SELECT_HAND`
with 0 or 6–8 cards selected — never 1–5.

**Root cause:** the action space is 8 *independent* binary bits (one per hand
slot); Balatro's `highlighted_limit` caps a legal selection at 5, and
`RLBridge/input.lua` rejects anything outside 1–5 outright. A `MultiDiscrete`
action mask is per-dimension and structurally **cannot express** a cardinality
constraint like "at most 5 of 8" — so the policy could and did emit illegal
selections freely, each one burning a full environment step for nothing, and
`-0.1 × retry_count` fed the noise straight into the reward signal.

**Decision:** clamp rather than redesign the action space immediately. Selections
over 5 are truncated to the first 5 (`cards_dropped` counter); a genuinely empty
selection falls back to a single card (`empty_selection` counter, later found to
need its own fix — see D013). This is **action projection** — the executed
action differs from the one the policy actually sampled, which the code
comments describe honestly as "slightly muddying credit assignment." It was
chosen as the pragmatic immediate fix over a full action-space redesign because
the latter changes the observation/action contract and invalidates checkpoints;
see the "Open questions" section below for where that redesign stands.

**Measured impact**, comparing the pre-fix run to a post-fix live run:

| | Before | After |
|---|---|---|
| Steps retried | 15.3% | **0.0%** |
| Win rate | 2.0% | **14.6–15.0%** |

A ~7.3× win-rate improvement, though see D013 for why part of the credit
attributed to "clamping" initially included telemetry noise that made the
picture look worse than it was, and D017's Open Questions for the caveat that
this outcome is dominated by *removing wasted steps*, not by the policy having
learned card selection well — 58.5% of `SELECT_HAND` steps still require
clamping as of the last completed run.

### D013 — `empty_selection` counter bug: PLAY/DISCARD steps were false positives

**Date:** 2026-08-13, found while verifying D012's results before writing them
up — the counter itself was being trusted before it was checked.

**Symptom:** the `empty_selection` counter read **71,753** on a run of ~143k
steps — roughly half of everything — which does not look like genuine policy
behavior and turned out not to be.

**Root cause:** card params were being extracted for *every* action type, but
only `SELECT_HAND` uses them. `RLBridge/input.lua`'s `play_hand()` and
`discard_hand()` take **no arguments** — they act on whatever is already
highlighted — and the action mask already forces all 8 card bits to 0 on those
steps by design, since card selection is irrelevant there. The clamp logic
didn't know the difference: it read "nothing selected," logged an empty
selection, and fabricated a card index the mod then silently discarded.

Splitting the same run's telemetry by action type made the scale of the false
positives unambiguous:

| Action | Steps | Counted "empty" | Genuinely empty |
|---|---|---|---|
| SELECT | 71,816 | 7 | **7** |
| PLAY | 37,055 | 37,055 | 0 |
| DISCARD | 34,761 | 34,761 | 0 |

**Fix:** `BalatroActionMapper.process_action` now only calls
`_extract_select_hand_params` when the resolved action is `SELECT_HAND`;
PLAY/DISCARD send `[]`. Zero behavioral change to the game — it already ignored
these values — but `empty_selection` went from meaningless to a genuine 7-case
signal, and roughly half of all steps skip needless work.

**A second, smaller bug fixed in the same pass:**
`response_validator.validate_response()` was being called twice in
`process_action` — once unguarded, then again inside a `try/except` — which
meant the `except` block could never actually catch anything, since the
unguarded first call would already have raised. Fixed by removing the redundant
unguarded call.

### D014 — Reconnect resilience: bounded wait instead of hard failure

**Date:** 2026-08-13, prompted by an actual run dying this way
(`2026-08-13_030459_run`'s predecessor closed Balatro mid-session and the
trainer exited with `RuntimeError: Failed to receive initial request from
Balatro`).

**Decision:** `BalatroEnv.reset()` now retries `wait_for_request()` in a bounded
loop (`BALATRO_RL_RECONNECT_TIMEOUT`, default 300s) instead of raising on the
first missed request. `BalatroSocketIO.accept_connection()` gained a settable
`timeout` parameter to make this interruptible rather than blocking forever on a
single `accept()` call.

**Why bounded rather than infinite:** a game that is genuinely never coming back
should still surface as an error rather than hang the trainer silently forever —
so the wait is generous (5 minutes, covering a manual relaunch) but not
unconditional.

**Verified** with a fake client that disconnects and reconnects after a delay
(confirming the resume path actually works, not just that it doesn't crash), and
a separate case with no client at all (confirming the timeout is honored rather
than blocking indefinitely).

### D015 — Web remote view: one shared reader, explicitly locked

**Date:** 2026-08-13, found during the same verification pass as D013 — by
inspecting the code for what *would* break under concurrent access, not because
a failure was observed.

**Bug:** `monitor/web/server.py` used `ThreadingHTTPServer`, which runs each
connected client on its own thread, but held the `TailReader` (which tracks a
byte offset and a partial-line buffer) as a single object shared across all of
them with no synchronization. Two people checking the same run from their phones
at once would race on that offset and could corrupt the read.

**Fix:** `State` gained a `threading.Lock` and a 0.5s snapshot cache
(`snapshot_cached()`), so concurrent viewers share one recent read instead of
each triggering (and racing on) their own.

**Verified** with 8 threads doing 20 reads each concurrently against a real
event file, asserting the reported step counts are monotonic (a corrupted shared
reader would show counts going backwards) and that no thread raises.

### D016 — Analysis tab: move the summary read off the GUI thread

**Date:** 2026-08-13, found by extrapolation before it was ever felt as
sluggish in practice — `summarize_run` measured 0.34s for a 30k-step run, and a
33-hour run is on the order of 1.8M steps, which extrapolates to roughly 20
seconds of a frozen window if run synchronously.

**Fix:** the summary read moved to a `QRunnable` on a `QThreadPool`
(`SummaryWorker` in `monitor/tabs/analysis.py`), with the existing per-run cache
(`self.summaries`) still applied so a previously-viewed run doesn't re-read.

### D017 — Observation size: only correct at hand size 0 or 8

**Date:** 2026-08-14. This is the most significant correctness fix in the
project's history so far — a **live bug**, not a future-proofing exercise,
despite initially being framed as one.

**How it was found:** raised as a hypothetical during a conversation about a
completely different feature (the game's built-in "Sort Hand" UI helper) —
the question "could hand size ever meaningfully exceed 8?" led to checking the
game's own Lua source for hand-size modifiers and consumable-insertion
mechanics, rather than assuming the answer.

**What was actually wrong:** `_extract_hand_features` appended 21 features per
card *actually present in the hand*, and only padded to a fixed size when the
hand was **completely empty**. So the declared `Box(216,)` observation was only
the correct length at hand size exactly 0 or exactly 8:

| Hand size | Observation length (before fix) |
|---|---|
| 3 | 111 |
| 5 | 153 |
| 8 | 216 ✓ |
| 10 | 258 |
| 100 | 1,728 |

This had never surfaced because ante-1 with a full 52-card deck refills to
exactly 8 every time. It was not a dormant edge case waiting for "Phase 2 scope
expansion" — it was already reachable in the current game via consumables. The
initial claim that hand size was bounded around 21 by joker/voucher modifiers
alone (`h_size` deltas of `+5, +3, +2, +2, +1, -1` on a base of 8) turned out to
be **incomplete**: `cardarea.lua`'s `card_limit` auto-expand only applies to
`G.deck`, never to hand, and the consumable card Cryptid ("Create N copies of 1
selected card in your hand") inserts directly into the hand list, uncapped. A
stockpile of Cryptid copies — the exact mechanic behind a known
"naneinf-in-one-turn" community strategy — can push hand size into the hundreds.

**Fix:** `max_cards` became a constructor parameter on `BalatroStateMapper`,
sourced from `BalatroEnv.MAX_CARDS` at construction time rather than existing as
a second hardcoded literal that had to be kept in sync by hand. The card block
now always pads to exactly `max_cards × 21` regardless of actual hand size,
cards beyond `max_cards` are truncated with a visible `hand_truncated` counter,
and a final length check in `process_game_state` logs an error and forcibly
corrects the array if it's ever the wrong length — a safety net that should
never fire if the padding logic is right, and testing across hand sizes 0
through 100 confirms it doesn't.

**Verified byte-identical to the pre-fix output at hand size 8**
(`np.array_equal`), specifically so that comparison runs recorded before and
after this fix remain valid against each other.

**Why this belongs in the log even though it's "just" a bugfix:** it's a
concrete instance of a general failure mode worth naming — *"a value that is
constant in the current narrow scope gets hardcoded, and the assumption survives
long after the scope that justified it has been outgrown."* `max_cards = 8` and
`MAX_ACTIONS = 3` in `BalatroEnv` are exactly this pattern. The fix here
addressed the one that was already live; `MAX_ACTIONS` is the same shape of risk
the moment the shop/blind-select/joker-sell actions are added (see Open
Questions).

### D018 — Replay seeds: already captured, only needed surfacing

**Date:** 2026-08-14.

**Finding:** `RLBridge/output.lua` reads `G.GAME.pseudorandom.seed` and it was
already flowing through `BalatroEnv.step()` into
`ReplaySystem.try_save_replay()` and into `replays.json` on every save — this
had been true since the initial commit. The Replays tab simply never rendered
the column.

**Change:** added a monospace `Seed` column to the Replays tab table; double-
clicking a cell in it copies the seed to the clipboard (via
`QApplication.clipboard()`) for pasting into Balatro to replay that exact game.
No data-capture change was needed — this is included in the log mainly as a
reminder to check what's already flowing through the system before assuming new
plumbing is required.

### D019 — Autoregressive card selection: illegal actions made unrepresentable

**Date:** 2026-08-14. Supersedes the clamping mechanism from D012.

**The problem D012 left behind:** clamping removed the 15.3% wasted-step cost,
but it worked by *projection* — the action the policy sampled and the action the
game executed were different whenever the selection was illegal. That was firing
on **58.5% of `SELECT_HAND` steps**, so it was the common case, not an edge case.
PPO computes its update ratio against the *sampled* action while the *projected*
action is what actually earned the reward, so more than half of all selection
decisions were being credited imprecisely.

**Decision:** replace the 8-independent-binary-bits card space with a sequential
one. Action space is now `MultiDiscrete([3, 9, 9, 9, 9, 9])`: one action-type
choice, then up to 5 card picks each holding a hand position (0–7) or STOP (8).
`AutoregressiveCardPolicy` in `ai/policies/autoregressive.py` samples the picks
one at a time, masking each against the cards already taken and gating STOP on
`min_picks`. Every representable action is legal, so nothing is ever clamped.

**Why not enumerate the legal subsets instead:** considered and rejected. At hand
size 8 there are only 218 legal selections, which is tractable — but the count is
combinatorial in hand size (`C(N,1)+…+C(N,5)`), reaching ~28k at N=21, ~760k at
N=40 and ~79M at N=100. Since D017 established that hand size is genuinely
unbounded in real play (consumables insert directly into hand, uncapped),
enumeration is not merely expensive but structurally unusable at sizes the game
can actually produce. The autoregressive form is always *at most 5 decisions*
regardless of hand size — only the per-decision option count grows, linearly.

**Two properties deliberately preserved, both tested:**

- **One environment step per decision.** All 5 sub-decisions occur inside a
  single forward pass; the env still performs exactly one socket round-trip per
  `step()`. Given the game round-trip is ~99% of wall-clock time (D005), a
  five-env-step formulation would have been a serious throughput regression —
  this was raised as an objection during design and drove the implementation
  toward an in-policy loop rather than a multi-step env.
- **`log_prob` reproducibility.** Sampling and `evaluate_actions` scoring share
  one code path (`_rollout`, teacher-forcing when actions are supplied),
  measured identical to 0.0. This is the failure mode with no symptom: if the
  stored and recomputed log-probabilities diverged, PPO's ratio would be wrong
  and training silently corrupted, with nothing raising an error.

**Also handled:** `PLAY_HAND`/`DISCARD_HAND` take no card parameters, so their
card slots are forced to STOP and contribute *zero* log-probability and entropy
— the policy is neither asked to make, nor graded on, a decision the game
discards. Verified explicitly (`test_forced_actions_contribute_no_log_prob`).

**The clamping code from D012 is retained**, but only as a backstop with its
counters wired to telemetry. `clamped_count` or `empty_count` reading non-zero
now means the policy's masking has a bug — it is an assertion, not a mechanism.

**Verified:** every sampled action legal across 512 samples × 5 hand sizes;
action-type masking respected; gradients reaching both heads; a full
`MaskablePPO.learn()` run of 1,024 steps against a fake environment that asserts
on every action, with save/load/predict round-tripping. **This invalidates
existing checkpoints** — the action space shape changed.

### D020 — Auto-resume must verify checkpoint compatibility

**Date:** 2026-08-15. Found the hard way, on the first real run after D019.

**What happened:** the first training session started after the autoregressive
switch died 90 seconds in:

```
Action spaces do not match:
MultiDiscrete([3 2 2 2 2 2 2 2 2]) != MultiDiscrete([3 9 9 9 9 9])
```

Auto-resume picked the newest checkpoint on disk — trained the previous day under
the old 8-binary-bit action space — and `MaskablePPO.load()` rejected it. Balatro
had already launched and connected; the run was lost.

**Why this was avoidable:** D019 explicitly recorded that the change "invalidates
existing checkpoints." That warning went into prose and was never enforced in
code. `find_latest_checkpoint` sorted by modification time and returned the
newest with no regard for whether it could actually be loaded, so a documented
known-incompatible state was left reachable by the default code path. Writing the
caveat down is not the same as handling it.

**Decision:** check spaces before attempting a load, and distinguish the two
cases by intent:

- **Auto-discovered** (`--resume` not given): incompatible checkpoints are
  skipped, oldest-to-newest, and the run starts fresh. A warning names the
  offending checkpoint and both action spaces, because "why did it not resume"
  is otherwise invisible.
- **Explicitly requested** (`--resume <path>`): raises. Silently training
  something other than what was asked for is worse than failing.

Spaces are read with `load_from_zip_file` without constructing the model, so the
check is cheap — scanning all 45 checkpoints on disk takes 0.30s.

Checkpoint discovery also moved from `__main__` into `train_agent`, because
compatibility can only be judged once the environment exists.

**Verified** against the actual checkpoint that caused the failure: rejected for
the new action space, still accepted for an environment matching its own — the
second assertion matters, since a check that refuses everything would also pass
the first.

**The general lesson**, worth more than the fix: a caveat recorded in
documentation is not a safeguard. If a known-bad state is reachable through the
default path, it will be reached.

---

## Phase 2 — Replacing the game with a simulator

**Decision:** retire the socket stack entirely. The `RLBridge/` Lua mod, its
JSON-over-TCP protocol, the Gymnasium wrapper, the SB3 trainer and the
threshold-based reward in `ai/` are gone. A seed-faithful Rust reimplementation
of Balatro (`sim/`) is now the environment, driven by a PPO trainer in `train/`;
`bridge/` remains only to run a trained checkpoint against the real game.

**Why:** every measurement in Phase 1 pointed at the same wall. Training
throughput was bound by the game's own event pacing — `EventManager:update` is
driven by real time, not `GAMESPEED`, so the engine completes at most ~60
blocking events per real second no matter how the multiplier is set. Forcing the
queue to drain in-frame bought 4.6x (296 → 65 ms/step) and that was the ceiling.
The simulator does ~10,000 steps/sec, four orders of magnitude past what the
mod could reach, and it does it deterministically per seed.

**What survived the move**, ported rather than rewritten:

- **Telemetry** → `train/balatro_train/telemetry/`. Unchanged in design: `emit()`
  does a dict construction and a non-blocking queue put, and drops events rather
  than adding latency. The principle held up under a 1500x throughput increase.
- **Graceful stop** → a stop file checked once per training iteration. The
  original justification was that Windows will not interrupt a thread blocked in
  a socket read; that specific reason is gone, but a file still beats a signal
  because it works regardless of who started the trainer or whether they share a
  process tree.
- **The monitor** → repointed at the new event schema.

**What did not survive, and why it is worth naming:** the per-step event. At two
steps per second, logging every step was free and the panels read from it
directly. At ten thousand steps per second the same design would write gigabytes
an hour and slow the thing it was measuring. Action distribution is now
accumulated in the rollout loop and emitted once per iteration.

**The general lesson:** an instrumentation design is only correct relative to a
throughput. None of the telemetry code was wrong; the assumption underneath one
of its consumers was.

---

## Phase 2a — One environment

**Decision:** a single conda environment, `balatro`, for the trainer, the
simulator bindings, the bridge and the monitor. The three virtualenvs it
replaces are deleted.

**Why:** they had silently diverged. The monitor's environment carried
CUDA-enabled torch; the trainer's, created from the same `pyproject.toml` on the
same machine, carried a CPU-only build — because the default PyPI torch wheel is
CUDA-enabled on Linux and CPU-only on Windows, and nothing in the resolution
surfaces that. A trainer that runs at a fraction of the expected speed with no
error is exactly the failure that a per-directory environment invites: the
mistake is invisible from the directory you are standing in.

Torch is now installed explicitly from the CUDA index as a separate first step,
before anything that depends on it, so the choice is stated rather than
inherited.

---

## Phase 2b — Pruning the event drain

**Decision:** the in-frame event drain was ported to a Steamodded mod for the
new bridge, then deleted before it was ever used.

**Why:** it was built to make the *real game* fast enough to train against. Once
the simulator took over training, the only remaining use for the live game was
watching a trained checkpoint play — and for that, speed is not the goal.
`--gamespeed 4` is the pace at which a run is legible to a human.

The drain also carried real risk for no remaining benefit: running queued
animations earlier than the engine intended can fire an animation whose target
UI object has already been destroyed, a crash class balatrobot already ships
patches for.

**The general lesson:** an optimisation outlives its justification more quietly
than it outlives its cost. This one was correct, measured, and worth 4.6x — and
still the right call was to delete it, because the workload it was written for
no longer exists.

---

## Phase 2c — auto_minibatch optimises for fitting, not for speed

**Decision:** `configs/m3_fast.yaml` ships `perf.auto_minibatch: false` with a
hand-tuned `ppo.minibatch_size: 4096`, and `perf.buffer_device: cpu` rather than
`auto`.

**Measured**, same run, same 6GB RTX 4050, everything else identical:

| `minibatch_size` | steps/sec |
|---|---|
| 4096 | ~5,250 |
| 16384 (chosen by the probe) | ~640 |

**Why it happens:** the probe halves `minibatch_size` until a synthetic
forward/backward/Adam step *fits* in VRAM. On this card 16384 fits — it measured
a 7,702 MiB peak and accepted it. But "fits" and "fast" are different questions.
At 16384 the card sits at ~5.8 of 6.1 GB, and CUDA responds to that pressure by
thrashing rather than cleanly OOMing, which the probe cannot observe because it
only ever asks whether one allocation succeeds.

**How it was found**, which is the part worth remembering: throughput collapsed
8x across a resume and the obvious suspect was the thing that had visibly
changed — `torch.compile` had crashed on missing Triton, so it had been turned
off. Installing `triton-windows` and re-enabling it recovered ~5% (610 → 639
steps/s), not 8x. Only then did comparing the config *baked into the old
checkpoint* against the running one show the real difference: the fast era ran
`minibatch_size: 4096, auto_minibatch: false` and, notably, `compile: false`
too. The 5,029 steps/s era never used torch.compile at all.

**The general lesson:** the thing that changed most visibly is not necessarily
the thing that changed causally. Checkpoints carry their own config; when
throughput moves, diff the recorded config against the live one before theorising
about the component that happened to throw the loudest error.

A second, narrower lesson: an autotuner is only as good as the quantity it
measures. This one measures a boolean (does it fit) and was trusted for a
continuous one (how fast is it).

---

## Phase 2d — bridge gains native Windows support

**Decision:** `bridge/` no longer assumes Linux + Steam + Proton. Paths, the
profile-link strategy, and process signalling all branch on platform, and
`balatrobot serve` is handed explicit paths on Windows since it cannot locate
Steam or a game to infer them from.

**What changed**, concretely: `paths.py` resolves the love.filesystem save
directory per platform (`%APPDATA%\Balatro` on Windows, the Proton compatdata
prefix on Linux, both overridable via `BALATRO_SAVE_DIR`); `launch.py` resolves
the user Mods directory the same way (`BALATRO_MODS_DIR` override); directory
links fall back from symlinks to junctions when Developer Mode / elevation
isn't available; `unix_to_wine_path` is a no-op on native Windows instead of
prepending `Z:`; stopping `serve` uses `terminate()` instead of `send_signal
(SIGINT)`, since POSIX signal delivery to another process doesn't exist on
Windows the way `_stop()` assumed.

**Two failures found only by actually running it**, both silent in a way worth
recording:

1. **Wrong CLI flag.** balatrobot's Windows launcher (`platforms/windows.py`)
   validates and launches `config.love_path`, not `config.balatro_path`, and
   silently substitutes a hardcoded Steam path when `love_path` is `None`.
   Passing `--balatro-path` therefore "succeeds" at the argument-parsing level
   and fails 30 seconds later with `Balatro executable not found:
   C:\Program Files (x86)\Steam\...` — a path the user never supplied, pointing
   suspicion at Steam/the install rather than the flag name.

2. **Release-zip nesting.** Extracting a Steamodded or balatrobot release zip
   produces `Mods/smods/smods-1.0.0-beta-1814a/`, one directory level below
   where Lovely (`Mods/<mod>/lovely/*.toml`) and Steamodded
   (`Mods/<mod>/manifest.json`) scan. Neither loader treats the extra level as
   an error — they scan, find nothing, and Balatro starts completely unmodded.
   The only symptom is the API health check timing out 30s later, which reads
   as a networking problem. `launch.py` now detects a single-child wrapper
   directory holding none of the mod markers and links through it
   (`_resolve_mod_root`).

**The common thread**: both failures were silent by construction — a wrong
path that resolves to *something* (a different Balatro, an empty mod
directory) produces no error at the point of the mistake, only a timeout
several steps downstream that implicates the wrong layer. The fix in both
cases was the same kind of move as Phase 2c: read what the log or the loader
actually saw (`Initialization complete in 2ms`, no patches listed) rather than
theorising from the exception's own stack trace, which pointed at the health
check and not at mod loading.

**Not fixed, deliberately left to the user**: the game install itself. A
Steam-emulator-patched executable (`steam_emu.ini`, `steam_api64.rne`,
`steam_appid.txt` beside `Balatro.exe`) is outside what this project sets up
regardless of platform; the Windows path support above works with any
legitimately-owned copy.

---

## Phase 7 — Decks, stakes, and the observation contract

Goal: lift the stack off its single supported configuration (Red Deck / White
Stake) onto the full 15 decks × 8 stakes grid, plus a separate endless
objective. Stage 0 is the observation-contract half, done first and in
isolation so the simulator work lands against a contract that is already
frozen and already proven not to cost anything.

---

## Phase 7 Stage 0a — Migrating a contract without retraining

**Decision:** widen the per-token feature vectors (`F_JOKER_FEAT` 11 → 15,
`F_SHOP_FEAT` 13 → 16) by **appending** fields, and migrate existing
checkpoints by filling the new input columns of `joker_proj` / `shop_proj`
with **zeros**. Deck identity and stake go into `global[48:64)`, the block
already reserved-and-asserted-zero, which they consume exactly — so
`F_GLOBAL` stays 64 and the global MLP's weight shape never changes.

**Why:** `Linear(x) = Wx + b`, so a zero column contributes `0 * x_new = 0`
for any input. The migrated policy is therefore mathematically identical on
every pre-P7 state, which means the ~2B steps already trained are not a
prototype to be thrown away — they are the starting point. Measured: tokens,
attention summary and value head all agree to **max delta exactly 0**.

**The alternative rejected** was widening `F_GLOBAL` and re-running from
scratch. It is simpler to implement (no migration script) and costs a
multi-day run plus a re-climb of the whole ante-ladder curriculum. The
reserved block existed precisely so this choice would not have to be made.

**Consequence worth recording:** `global` now has **no headroom left**. The
next global feature must widen `F_GLOBAL`, and that *will* change the global
MLP's shape. `encoding.py` says so at the point of use.

---

## Phase 7 Stage 0b — The lossy layer was the encoder, not the simulator

**Decision:** raise the observation slot caps from 10/6/3 to
`HAND_MAX = 12`, `JOKER_SLOTS = 16`, `CONSUMABLE_SLOTS = 8`.

**Why:** the simulator models Negative editions correctly — every Negative
joker adds a capacity slot, and Perkeo mints Negative consumable copies — so
a run can legitimately hold far more than 6 jokers and 3 consumables. The
*encoder* then truncated: `.take(JOKER_SLOTS)` dropped the rest from the
observation, and because the same constants size the **target masks**, those
jokers were also unsellable and those consumables unusable.

The jokers past the cap still *scored*. So the policy was being rewarded for
builds it could neither see nor manage, and the strategies most affected —
Perkeo / Blueprint / Cryptid negative-stacking — are among the strongest in
the game. This was live on `m4_branch` for its entire ~2B steps.

**Measured cost of the fix** (exact, from `OBS_SPEC`; buffer is
`rollout_len 128 × num_envs 4096`):

| caps (hand/joker/cons) | obs B/step | tokens | rollout buffer |
|---|---|---|---|
| 10 / 6 / 3 (pre-P7) | 3,351 | 27 | 1.64 GB |
| 12 / 8 / 5 | 3,813 | 33 | 1.86 GB |
| 12 / 12 / 6 | 4,098 | 38 | 2.00 GB |
| **12 / 16 / 8** | **4,392** | **44** | **2.14 GB** |

VRAM is the binding constraint on a 6 GB card, not FLOPs: +0.50 GB of rollout
buffer. Every cap is one constant in `encoding.py` plus its `consts.rs`
mirror, so falling back to 12/8 is a one-line change.

**`slot_embed` needed a structured remap, not a zero-pad.** Slot embeddings
are positional, so raising a cap shifts every later group — the joker group
moves from token index 10 to 12. Rows are moved; genuinely new rows are seeded
with the mean of their own group, so a new joker slot starts out looking like
a typical joker slot rather than like nothing. `type_embed` is indexed by
type, not slot, and needed nothing.

**This step is mathematically identical on pre-P7 states too**, which was not
obvious and is worth the evidence. Slots beyond the old caps are empty, hence
attention-padding, hence never read — so the surviving tokens must match
despite sitting at shifted indices:

| dtype | max delta over all token groups + summary + value |
|---|---|
| float32 | `5.96e-07`  (1 ULP, 2⁻²⁴) |
| float64 | `1.33e-15`  (1 ULP, 2⁻⁵⁰) |

Not *bit*-identical, for a benign reason: attention now reduces over 44 keys
instead of 27, and although the added keys carry exactly zero softmax weight,
the matmul's tiling and summation order change. The float64 run is the proof —
the residual **tracks machine epsilon** rather than staying fixed, collapsing
by 4.5×10⁸. That is the signature of rounding, not of a structural difference.
`test_slot_bump_difference_is_rounding_not_structure` pins it so it cannot
quietly become a real divergence later.

These figures come from a freshly-initialised policy; the residual scales with
activation magnitude, and the trained 3.5B checkpoint sits two orders higher at
`4.79e-05` while collapsing to `4.32e-14` in float64 exactly as above. The
*collapse* is the invariant, not the number — see Stage 0e.

**The general lesson:** the simulator was right and the tests were green; the
loss happened at a boundary whose job is to *summarize*. A cap sized for the
common case is invisible until the uncommon case is the one that wins.

---

## Phase 7 Stage 0c — A read-only monitor that crashed training

**Decision:** cap `TailReader` reads at 4 MB, and skip replaying
`episode_end` from deep history using a byte-level type peek that runs
*before* `json.loads`.

**Why:** attaching the monitor to a long run could OOM the GPU and kill
training — from a process that contains no CUDA code at all.

`TailReader.read()` read `size - self._offset` in a single call. On attach
that is the entire file, and the raw chunk, the `split()` list and the parsed
dicts are all live at once. Measured against the real `m4_branch` stream:

| | |
|---|---|
| file | 1.01 GB / 5,960,921 lines |
| parsed cost per event | 955 B |
| **peak RAM to attach** | **~7.5 GB** |
| machine / available with trainer running | 15.7 GB / **5.4 GB** |

**The OOM mechanism, which is the part worth writing down:** the training
config sets `pin_memory: true`, so the trainer holds **page-locked** host
memory for host-to-device staging. Pinned pages cannot be swapped out. The
monitor demands 7.5 GB against 5.4 GB available, physical RAM is exhausted,
the pinned allocation fails, and CUDA reports it as an OOM. The monitor never
touched the GPU — it starved the host buffers the GPU transfer depends on.

**The work was also ~100% waste.** `episode_end` is **99.95%** of a mature
run's file, and every consumer is a bounded `RingSeries(deque(maxlen=20000))`.
Nine of the ten panels do not even subscribe to it — and `panels/base.py`
filters by `EVENT_TYPES` *after* the full parse. Millions of events were
constructed at 955 B each and immediately discarded.

Result:

| | before | after |
|---|---|---|
| peak RAM | ~7,500 MB | **55 MB** |
| attach time | minutes, or a crash | **19.2 s** |
| events parsed | 5,960,921 | 118,560 |

Rare events survive in full — 2,189 rollouts, 573 promotion evals, 45
milestones — so curriculum history is complete rather than truncated.

**The alternative rejected** was seeking straight to the tail, which makes
attach instant. It also silently loses the early promotion history the
curriculum panel accumulates in an unbounded list. Scanning the whole file
while parsing only the 0.05% that matters keeps that history and still fits
in 55 MB; the 19 s is the price, and it is paid once per attach.

Two supporting fixes, both root causes of the incident rather than the crash
itself:

- **"Follow latest" now means the latest *active* run**, not the newest
  mtime. A short-lived debug run sorts newest and would yank the view off a
  live multi-day run — and switching back paid the full re-read above.
- **`test_perf.py` was writing into the repo's real `train/runs/`**, because
  it loaded `configs/debug.yaml` without redirecting `log_dir`. That is what
  created the newer-sorting directory in the first place. It now uses a
  scratch dir, and an autouse fixture in `train/tests/conftest.py` **fails
  the suite** if any test leaves a run directory behind.

**The general lesson:** "read-only" bounds what a component can *corrupt*, not
what it can *consume*. The monitor's isolation guarantee was written in terms
of the event stream — it never writes, so it cannot disturb a run — and that
reasoning is correct and irrelevant, because the shared resource that actually
coupled the two processes was host RAM. The guarantee needed a second half:
read-only *and* bounded. This is the same shape as Phase 2b's lesson about an
instrumentation design being correct only relative to a throughput; here it
was correct only relative to a file size.

---

## Phase 7 Stage 0d — One contract in two files, now actually enforced

**Decision:** `test_rust_mirror_matches_encoding` parses `sim/py/src/consts.rs`
and asserts every shared constant and both frozen enums (`Deck`, `Stake`) match
`balatro_train/encoding.py`.

**Why:** the invariant was already stated — `consts.rs` line 1 says the Python
file is the source of truth and this one must match it exactly — but nothing
checked. It was a comment, and comments do not fail builds.

It got tested for real during Stage 0. With the P7 contract landed but the
PyO3 binding not yet rebuilt (a live run held the `.pyd` locked on Windows),
the Python constants were temporarily hand-edited back to the pre-P7 values so
training could be restarted against the old binary. Reasonable, and marked
`TEMPORARY` at every site. Two things went wrong anyway:

1. **The revert was partial.** The feature *widths* went back
   (`F_JOKER_FEAT` 15 → 11, `F_SHOP_FEAT` 16 → 13) but the *offsets* into them
   did not, leaving `JOKER_ETERNAL_OFF = 11` addressing index 11 of an 11-wide
   vector, and `SHOP_ETERNAL_OFF = 13` of a 13-wide one. Latent only because
   the sticker encoders are Stage 2 work and nothing writes those slots yet —
   a loaded gun for whoever writes them.
2. **The two halves of the contract silently disagreed.** `consts.rs` still
   held the P7 values. Nothing anywhere reported this; the only mechanism that
   would have was the runtime boundary validator in `sim_env.py`, and it cannot
   fire until the binding is rebuilt, which was exactly the blocked step.

**The trade that was actually made** is the interesting part: reverting turned
17 loud, precisely-worded failures (`obs[joker_feats]: shape (4,6,11) !=
(4,6,15)`) into 6 confusing ones plus a source-level divergence that no test
covered. The loud failures were *correct* — they were the system reporting
"your binding is stale" in the clearest terms available.

**The general lesson:** a boundary validator that runs at boundary-crossing
time cannot protect you while the boundary is what you are unable to cross.
The check has to live somewhere that is always reachable — parsing the other
side's source text needs no toolchain, no rebuild, and no running process, so
it works precisely in the degraded state where the real check is unavailable.
Verified the guard fires rather than passing vacuously: perturbing one Rust
constant produces `consts.rs has drifted from encoding.py: JOKER_SLOTS:
rust=6 != encoding.py=16`.

---

## Phase 7 Stage 0e — The half of the migration that was not in the state dict

**Decision:** `migrate_contract_p7.py` migrates `ck["optimizer"]` alongside
`ck["policy"]`, and `--verify` gates on the float32-vs-float64 epsilon
collapse rather than on an absolute float32 tolerance.

**Why the optimizer:** Stage 0a and 0b established that the *weights* can be
widened losslessly, and the verification proved it on tokens, summary and
value. All of that was true and none of it was sufficient. Adam carries
`exp_avg` and `exp_avg_sq` per parameter, both shaped like the parameter, and
resuming loads them straight back:

    RuntimeError: params, grads, exp_avgs, and exp_avg_sqs
                  must have same dtype, device, and layout

The crash is the lucky case. `slot_embed` is the unlucky one: its moments have
the same *shape* under any cap that sums to 44, so a stale tensor of the right
size would have loaded cleanly with every row attached to the wrong slot —
joker momentum applied to hand positions, silently, for the rest of the run.
Optimizer state is keyed by parameter POSITION, not by name, so nothing in
that path can notice. The moments therefore get the same treatment as the
weights: appended columns zero-filled, `slot_embed` rows *remapped*.

New rows and columns get **zero** moments, not their group's mean. The weights
seed a new slot with the group mean so it starts out looking like a typical
slot; a moment is a record of gradients that a never-trained parameter does
not have, and averaging its neighbours' would invent one. Adam's first update
from `m = v = 0` is exactly zero, so nothing jumps.

**Why the tolerance:** `verify()` asserted `max |delta| < 1e-5` in float32.
That passed on a freshly-initialised policy — Stage 0b's table records
`5.96e-07` — and failed on the real 3.5B checkpoint at `4.79e-05`, with
nothing structural changed. The residual scales with the activation magnitudes
of the weights being migrated, so any absolute threshold is really a threshold
on how *trained* the checkpoint is, which is exactly backwards for a gate
meant to protect trained checkpoints. The dtype comparison Stage 0b used as
*evidence* is the correct *gate*: float32 `4.79e-05` → float64 `4.32e-14`, a
collapse of ~1e9 that tracks machine epsilon. Extra keys that genuinely
influenced the result would persist at float64.

**How it was found:** by running the migration end to end — migrate, load,
`collect()`, `update()` — instead of stopping at "the script printed
`wrote`". Both defects live strictly after the last thing the script checks.

**The general lesson**, and it rhymes with the `auto_minibatch` entry above: a
migration is defined by what the *next process* reads, not by what the
migration script writes. A checkpoint is not a model, it is everything needed
to continue — weights, optimizer moments, normalizer statistics, step
counters. Verifying the model and calling the checkpoint migrated is checking
the half that was easy to think about. The test that would have caught it is
not another assertion about `joker_proj`; it is one `optimizer.step()`.

---

## Phase 7 Stage 1 — Decks and stakes in the core

**Decision:** `Run::new(seed)` stays, and becomes a shim for
`Run::with_config(seed, RunConfig::default())`.

**Why:** there are 106 `Run::new` call sites and 100 of them are oracle-vector
tests. Changing the signature would have churned every one of them for no gain,
and worse, it would have made "did this refactor move a frozen vector?" hard to
answer — the diff would be enormous and mostly noise. With the shim the answer
is visible: the vectors compile unchanged, so if they still pass, Red/White is
still Red/White. `config.rs` carries an executable restatement of the old
hardcoded constants (`red_white_reproduces_the_previous_hardcoded_constants`)
so the claim does not rest on the vectors alone.

**Deck construction consumes RNG for exactly one deck.** Erratic draws 52 times
on the `'erratic'` key; every other deck is a pure function. Since Balatro keys
each pseudorandom stream separately (`G.GAME.pseudorandom[key]`), order *across*
keys is unobservable — so no deck can shift a stream that a frozen vector
depends on. `only_erratic_consumes_rng` asserts this for all 15 rather than
leaving it as a comment.

**Ordering that is observable, and was nearly wrong:** `Back:apply_to_run` runs
at game.lua:2043, and the run-start `'Voucher1'` roll at :2178. The Zodiac Deck
pre-redeems three vouchers, and redemption marks them used — which removes them
from the pool that roll draws from. Applying the deck's vouchers *after* the
roll would let the shop offer a voucher the deck already owns. Pinned by
`pre_redeemed_vouchers_leave_the_shop_pool`.

**A fidelity bug the deck system exposed.** The Needle's hand reduction was
`STARTING_HANDS - 1`, a constant. blind.lua:184 reads
`G.GAME.round_resets.hands - 1`. On Red/White those are the same number, so the
bug was invisible and every vector passed. On the Blue Deck (+1 hand) or the
Black Deck (−1 hand) they diverge. Now reads `self.round_resets_hands - 1`.

This is the recurring shape of this phase: a constant is indistinguishable from
a correct expression until something makes the inputs vary. The deck system did
not introduce the bug, it made it *reachable* — which is an argument for landing
the configuration axis before the content that depends on it, not after.

**Plasma is two independent mechanics, not one.** `ante_scaling = 2` multiplies
the blind requirement (blind.lua:107) and the `final_scoring_step` hook averages
chips and mult (back.lua:125-171). They are separate fields —
`StartingParams::ante_scaling` and `Modifiers::plasma_balance` — because the
stake's `scaling` curve is a *third* independent input: Plasma on Gold Stake
gets curve 3 and the ×2. `ante_scaling_is_independent_of_the_stake_curve` pins
the three-way combination.

The balance step floors both halves independently (`math.floor(tot/2)` twice),
so an odd total silently loses one chip and one mult. Ported verbatim.

**A test that was wrong before the code was.** The three blind curves share an
identical tail formula past ante 8, so it is tempting to assert they stay
exactly proportional. They do not: each is truncated to two significant figures
*independently* (`amount - amount % 10^floor(log10(amount)-1)`), so at ante 9
the ratio is 230000/110000 = 2.0909, not 2. The truncation is the real
invariant — and even that can only be asserted below ~2^53, because past there
an f64 is no longer an exact integer and `%` against a power of ten returns
noise. The implementation is faithful up there regardless: it uses `lua_fmod`
exactly as Lua does, and Lua numbers are the same f64.
