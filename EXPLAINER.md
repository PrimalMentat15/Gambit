# Explainer: the six things worth understanding before deciding

Companion to `COMPARISON.md`. Written to be read in order — parts 3 and 6 lean on part 1.

---

## 1. Is JSON-RPC over HTTP actually slower than your raw socket?

**Short answer: not meaningfully. Your intuition about "less work over a simpler protocol" is correct about the protocol and wrong about where the time goes.**

Your own README already contains the decisive number:

> "Training throughput is bound entirely by the game, not by Python — measurements put Python at ~0.2 ms of a step."

Your step costs 65 ms with the drain patch on. Of that, 0.2 ms is Python — including your socket I/O, your observation encoding, and your action mapping. That's **0.3% of a step**. The other 99.7% is Balatro doing Balatro things.

Now price the difference between the two transports:

| | Gambit (raw TCP, line-delimited JSON) | balatrobot (JSON-RPC 2.0 over HTTP) |
|---|---|---|
| Connection | one persistent socket | HTTP keep-alive via `httpx` |
| Per-call overhead | ~0.05–0.2 ms | ~0.2–0.5 ms |
| Extra bytes | none | HTTP headers (~150–300 B), JSON-RPC envelope (~60 B) |

Worst case you're paying an extra **~0.3 ms** on a **65 ms** step. That is half a percent. It will never be the thing that limits you.

### So why is Balatron at ~1.4 steps/sec and you at 15.5?

Three reasons, none of them the transport:

1. **Their steps contain more game.** Your step is `SELECT_HAND` or `PLAY_HAND` or `DISCARD` inside a single small blind. Theirs includes shop transitions, pack openings, round-eval cash-out screens, blind selection — the animation-heaviest parts of Balatro. A cash-out screen alone takes seconds of real time.
2. **They don't have your drain patch.** They set `BALATROBOT_GAMESPEED=8` and stop there — which, as your own profiling proved, does not speed up the blocking-event path at all. See part 4.
3. **Different definition of a step.** Not strictly comparable.

### The part you should actually plan for

**Your 15.5 steps/sec will not survive full-game scope.** That number is measured on ante-1 hand play. Once your episodes include shop, packs, and cash-outs, expect it to fall — probably to somewhere in the 8–10 range per instance even with the drain patch ported, because those transitions are animation-bound in ways hand-play isn't.

This is worth internalizing before you migrate, so the drop doesn't read as "balatrobot made it slower." It didn't; the scope did.

### Two throughput levers balatrobot gives you that RLBridge cannot

- **`--headless`** — the game runs with no window. You stop paying for rendering entirely. RLBridge can't do this; you're always rendering a full Balatro.
- **`--no-shaders`** — cheaper rendering when you do want a window.
- **`--port N`** — multiple instances as a supported flag rather than a hack. Balatron measured 196 → 309 → 433 steps/min for 1 → 2 → 3 instances (sublinear, but real).

These may well net out to *more* throughput than you have now, not less.

### Verdict

**Your conclusion is right and the reasoning is right.** You'd have to substantially rewrite RLBridge to expose jokers, shop, consumables, deck state, vouchers, tags, packs and boss effects — and you'd be reimplementing balatrobot, worse, alone, with no ecosystem. Migrate.

---

## 2. Fingerprints vs. ID embeddings — a worked example

Take five real Balatro jokers that share one mechanism:

| Joker | Effect |
|---|---|
| The Duo | ×2 mult if played hand contains a **Pair** |
| The Trio | ×3 mult if hand contains **Three of a Kind** |
| The Family | ×4 mult if hand contains **Four of a Kind** |
| The Order | ×3 mult if hand contains a **Straight** |
| The Tribe | ×2 mult if hand contains a **Flush** |

Mechanically these are one joker with five parameter settings: *"if the hand contains hand-type X, multiply mult by Y."*

### How balatroagent sees them (learned ID embedding)

Each joker is an integer index into a table of 160 learned vectors:

```
The Duo    → id 47  → embedding[47]  = [0.13, −0.88, 0.41, ...]   (128 floats)
The Tribe  → id 51  → embedding[51]  = [−0.62, 0.09, 0.77, ...]   (128 floats)
```

At initialization those two vectors are **random and unrelated**. Nothing tells the network that The Duo and The Tribe are cousins. The network discovers what The Duo does only from episodes in which it actually owned The Duo, and that discovery transfers to The Tribe **not at all**.

Do the counting. Roughly 150 jokers, 3 shop slots per shop, a handful of shops per run. Any specific joker shows up in maybe 1–2% of runs. So:

- **10,000 episodes** → you've owned The Duo maybe 100–200 times, across wildly different builds. That's noise. The embedding stays near-random.
- **Tens of millions of episodes** (what 4B steps buys) → hundreds of thousands of Duo-owning episodes. Now the embedding is sharp — and it can encode things a hand-written schema could never express: how The Duo interacts with *this specific* Blueprint position, how its value shifts by ante, that it's a trap pick when you're already committed to Flush.

### How Balatron sees them (property fingerprint)

Balatron hand-wrote a schema per joker in `data/jokers.py`. Conceptually:

```python
"The Duo":   make_joker(xmult=True, xmult_value=2.0,
                        triggers=["specific_hand_type"], hand="Pair",
                        scoring_timing="after_cards", tier_weight=...)
"The Tribe": make_joker(xmult=True, xmult_value=2.0,
                        triggers=["specific_hand_type"], hand="Flush",
                        scoring_timing="after_cards", tier_weight=...)
```

Those become 54-float vectors that are **nearly identical** — same effect type, same magnitude, same trigger class, same timing. Differing in one field.

The consequence is the whole point: **a gradient update from an episode with The Duo automatically improves the network's handling of The Tribe**, because they occupy almost the same point in input space. You have effectively multiplied your data by the number of jokers sharing each mechanism. Balatron's README states the intent directly:

> "The network generalizes across jokers with similar effects — it doesn't need to memorize 150 individual joker behaviors, it learns that 'x2 mult on face cards' is valuable regardless of which joker provides it."

### The trade, stated plainly

| | Fingerprints | ID embeddings |
|---|---|---|
| Data needed | low | very high |
| Generalizes to unseen/modded jokers | yes, free | no, needs retraining |
| Can represent things you didn't think of | **no** | yes |
| Can be *wrong* | **yes — it's hand-written** | no (nothing to get wrong) |
| Authoring cost | 66 KB of schemas | zero |

The two failure modes are opposites:

- **Fingerprints fail by omission and error.** The network can only see what you encoded. Balatron's `dec-100` audit found exactly this: *"two estimators disagreeing 4× on the same jokers, Blue Joker valued at zero in every shop decision."* Their agent had been making shop decisions with a joker priced at nothing for an unknown length of time. And structurally: *"A depth-1 marginal `build_value` structurally cannot value a synergy piece worthless on its own."* A fingerprint describes a joker in isolation; synergy lives between jokers, so a fingerprint scheme is blind to it by construction.
- **Embeddings fail by starvation.** They're not wrong, they're just untrained, and an untrained embedding is a random vector feeding your policy noise.

### The analogy that makes it stick

Fingerprints are teaching chess with piece values: pawn 1, knight 3, rook 5. Enormously useful when you've played ten games. A grandmaster does not think in piece values — after millions of games they've learned positional judgment the table cannot express, and by then the table is a *ceiling*, not a scaffold.

That's why I called it a data-budget decision rather than a correctness one. **Both are correct. Which is better depends only on how many episodes you will ever have.** Live game → fingerprints. Sim → embeddings.

---

## 3. Which reward design is "better"? Which hyperparameters?

### Reward: the single principle

**Reward shaping trades bias for sample efficiency.**

Every shaping term encodes something you *believe* about the game. If your belief is right, learning speeds up. If it's subtly wrong, the agent optimizes your belief rather than the actual objective — and RL agents are extremely good at finding the gap.

So the question isn't "which reward is better," it's "how much bias can I afford to trade for speed?" And that's set by your sample budget.

### Worked example: your thresholds vs. balatroagent's continuous term

Blind requires 300 chips. Agent plays a hand.

| Chips scored | % of blind | Gambit reward | balatroagent reward (β=1) |
|---|---|---|---|
| 224 | 74.7% | **+4.0** | +0.747 |
| 225 | 75.0% | **+10.0** | +0.750 |
| 300 | 100% | **+20.0** | +1.000 |
| 900 | 300% | **+20.0** | **+1.000** |

Three things fall out of that table.

**(a) The cliff.** One extra chip triples the reward. PPO estimates advantages from sampled returns; a discontinuity means two nearly identical actions produce wildly different returns. That inflates advantage variance for no informational reason — the agent must burn samples learning where a boundary is that *you invented*. It isn't in the game. And note you've already tuned these thresholds once ("lowered from 80%", "lowered from 50%"): needing to tune a threshold is the symptom. A continuous function has no thresholds to tune.

**(b) The cap.** Look at the last two rows. balatroagent's `min(chips/req, 1)` says scoring 900 is worth **exactly the same** as scoring 300 — because in Balatro it is. You cleared the blind; excess chips don't carry over, don't buy anything, don't help. Your `+20 MONSTER HAND` bonus says overkill is worth double a clean clear, which *actively teaches the agent to farm chips*.

Balatron hit this exact wall one scope level up and wrote it down:

> "The single-hand high-water bonus is **Phase-2 only** (it farms chips against depth in Phase 1)."

And more broadly, after a plateau audit: they cut dense score shaping from 0.1 to 0.02 because *"the policy was optimizing comfortable mid-game scoring instead of winning."*

**(c) Scale.** A winning ante-1 episode nets you roughly +75 to +95 total. balatroagent's nets ~+16, with return normalization on top. Large unnormalized returns make the value loss dominate the policy loss on a shared trunk — which is *precisely* the bug Balatron spent weeks on:

> "at VL~20 the value term (0.5×20=10) swamped the policy+entropy gradient (~370× the entropy term, ~2000× policy_loss) — which is why entropy_coef 0.04→0.10 only moved entropy ~0.45→0.48 and ante stayed flat."

They were tuning entropy to fix a problem that was actually reward scale.

### One reward mistake that's easy to make by accident, and the fix for it

Balatron found this bug and wrote it up as: "pay for the change, not for having it."

Here's the mistake, plainly. Say you want to encourage the agent to hold a joker that gives xmult (a multiplier), so you give it +0.5 reward on every single step it owns one. Sounds reasonable. But over a 200-step run, that's +100 total — way more than the +15 you give for actually winning. So the agent's best strategy becomes "own this joker and drag the run out as long as possible," not "win." You paid it more for existing than for winning, without meaning to. Their own numbers on this: *"they accrued +20–40 over a run vs +10 for winning."*

The fix is simple: only pay when something **changes**, not for every step it stays true. +1 the moment you acquire the joker category, −1 if you lose it, nothing while you just hold it. Now holding it for 200 steps nets you the same +1 as holding it for 2 steps — the reward reflects the one-time event (getting it), not how long you happened to have it.

This has a name — potential-based reward shaping — and there's a real math result behind it (Ng, Harada & Russell, 1999) saying that if you shape rewards this way (pay only on the change between two states, discounted), it's mathematically guaranteed you haven't changed what the "best" strategy actually is — you've just made it easier to learn faster. Reward anything else — like a flat per-step bonus — and you have no such guarantee, and you can accidentally teach the agent to do something other than win, like stall.

You're mostly fine right now: your reward terms fire on actual chip-gain events, and you have a guard (`blind_already_defeated`) that stops you from double-paying. But the moment you add something like "reward for owning a scaling joker" or "reward for money held," you're at risk of hitting exactly this trap — so pay for the change (getting it / losing it), not for every step it's true.

### The verdict on reward

**balatroagent's is better in an absolute sense** — unbiased, correctly capped, and it *anneals to near-nothing* (β: 1.0 → 0.1 over steps 500M–900M), meaning the training wheels come off and the agent ends up optimizing the true objective.

**But it is only trainable because they have 4B steps.** With a live-game budget, a reward that says almost nothing will teach almost nothing. On live you need something closer to Balatron's — informed, denser, and accepting the bias.

Practical rule if you stay live: shape densely, but make every term **continuous**, **capped where the game caps it**, **potential-based if it's about state**, and **annealed toward zero** so you can find out later whether it was helping or lying.

### Hyperparameters: the γ / λ example, because it's the deepest one

`gamma` (γ) is how much a reward N steps away is worth now: `γ^N`.

| Episode length | γ = 0.99 | γ = 0.995 | γ = 0.999 |
|---|---|---|---|
| 10 steps (your ante-1) | 0.90 | 0.95 | 0.99 |
| 250 steps (full run) | **0.081** | 0.29 | **0.78** |

So at full-game scope, γ=0.99 means the win reward is worth 8% by the time it reaches your first shop decision. Your 0.99 is **correct today and wrong the moment you expand scope.**

Now the subtle part. Balatron raised γ to 0.995 for exactly this reason — and it didn't help, because GAE discounts by **γ·λ**, not γ:

> A step-40 shop decision, win at step 179 = 139 steps back.
>
> | | credit reaching that decision |
> |---|---|
> | γ alone: 0.995^139 | **0.50** ← what the fix intended |
> | γ·λ: 0.9452^139 | **0.0004** ← what actually arrived |

They then raised λ to 0.99, watched explained variance fall 0.620 → 0.387 with value-loss spikes in 18% of updates, and reverted on a **pre-registered** variance trigger.

**So why does balatroagent run γ=0.999 with λ=0.95 and win?**

Because λ < 1 doesn't destroy long-horizon credit — it *routes it through the value function*. With λ=0.95 the advantage estimate leans on the critic: a shop decision is credited not by the raw reward 250 steps later, but by `r + γV(s') − V(s)` — the critic's opinion of whether that purchase improved the position. That works beautifully **if the critic is accurate**, and a critic becomes accurate with data.

- balatroagent: 4B steps, 4096 parallel envs, return normalization → excellent critic → λ=0.95 is fine, and γ=0.999 does real work.
- Balatron: ~18M live steps, EV hovering 0.39–0.62 → mediocre critic → credit genuinely doesn't flow, and raising λ (leaning on realized returns instead) blows up variance because the returns themselves are high-variance at a 1% win rate.

**This is the general shape of the answer to "which hyperparameters are better":** they aren't better or worse in isolation. Nearly every difference between the two projects is downstream of sample budget.

| Setting | Balatron | balatroagent | Why they differ |
|---|---|---|---|
| `num_envs` | 1–3 | 4096 | An env instance costs a whole game process vs. a struct |
| `epochs` | 8 | 4 (A/B'd; 2 beat 4) | Balatron's rollout costs ~25 min of live game and the update costs seconds — squeeze every rollout. balatroagent can just collect more. |
| `minibatch` | 4 minibatches | 32768 | Batch size follows from 4096 envs × 128 rollout |
| `gamma` | 0.995 | 0.999 | Both want long horizon; only one has the critic to support it |
| `ent_coef` | 0.03 (after 0.01→0.10 and back) | 0.01 | See below |
| `target_kl` | 0.03 | none | Early-stopping epochs matters when rollouts are precious |

### The most transferable hyperparameter lesson

Balatron escalated `entropy_coef` 0.01 → 0.025 → 0.04 → 0.06 → 0.10 chasing what looked like exploration collapse. Entropy moved from 0.45 to 0.48. Ten-fold increase, nothing happened.

The actual causes, found later: (1) the action mask was injecting a heuristic bias ~57× stronger than the network's own logits, **structurally flooring entropy at ~0.24**; and (2) the value gradient was swamping the policy gradient on a shared trunk. Their own note: *"the escalation was a wrong-diagnosis lever (the limiter was the frozen policy gradient, not entropy-starvation)."*

**The lesson: when a hyperparameter doesn't move the metric it's supposed to move, that is evidence of a bug — not a reason to push it harder.** That will save you weeks at some point.

---

## 4. Your event-drain patch, in plain language

### What Balatro is doing

Balatro is a LÖVE (Love2D) game. It runs a frame loop ~60 times a second: update everything, draw everything, repeat.

Animations — a card flying to the played area, chips ticking up, a joker flashing when it triggers — are not inline code. They're **events pushed onto a queue**. Some are marked *blocking*: the game refuses to move on until that event completes.

### The two bugs you found

**(a) The event queue is rate-limited by real time, not game time.**

`EventManager:update` is driven by `real_dt` — actual wall-clock seconds — rather than the `GAMESPEED`-scaled dt. It completes at most one blocking event per pass, and it only runs a pass once per 1/60 s of real time.

That caps the game at **~60 blocking events per real second, no matter what you set GAMESPEED to.**

This is the important bit: `GAMESPEED=8` speeds up the *visual interpolation* but not the event chain. A scored hand queues dozens of blocking events — one per scoring card, one per joker trigger, one for the chip count-up — so it costs about a real second regardless. Your comment says it exactly:

> "That caps the game at ~60 blocking events per second no matter how high GAMESPEED is, which is why a scored hand costs about a second."

**(b) Card movement is never sped up at all.** `Moveable:move` is driven by `real_dt` capped at 1/20 s and never has `SPEEDFACTOR` applied. Cards travel at real-time speed forever.

### What your patch does

When a frame detects pending blocking events, instead of waiting one frame per event you loop **inside that single frame**:

```lua
while pending > 0 and passes < MAX_DRAIN_PASSES do
    G.TIMERS.TOTAL = G.TIMERS.TOTAL + dt * speed   -- advance the clock so
                                                    -- timer-gated events expire
    G.E_MANAGER:update(dt, true)                    -- run one more pass
    passes = passes + 1
    local remaining = count_blocking()
    if remaining >= pending then break end          -- bail if no progress
    pending = remaining
end
```

Two details that make it correct rather than a hack:

- You manually advance `G.TIMERS.TOTAL`, because inside your loop the game clock is frozen — so chained `after(delay)` events would never become eligible and the queue would stall.
- You bail the moment a pass makes no progress, which means the remaining events are gated on card *movement* (which only the real frame loop advances), not on time.

**The same events run, in the same order, with the same results. Only the wall-clock spacing between them changes.** That's why it's legitimate and not cheating — which matters, given your `CLAUDE.md` says *"we don't want to modify the game in any way to give an unfair advantage."*

Measured: **296 → 65 ms/step**, `PLAY_HAND` **995 → 193 ms**. A 4.6× speedup.

### The thing you tried that didn't work, and why that's good

`BALATRO_RL_PUMP` also advances card movement inside the drain loop. It *does* let position-gated events complete — but it removes the early-bail condition, so the loop runs hundreds of passes that each walk every moveable object. Measured **114 ms/step with one step hitting 13 seconds**. You measured it, found it worse, and left it off behind a flag.

That's disciplined work, and it's the same instinct Balatron's `DECISIONS.md` is built on.

### How this contrasts with the other two

**Balatron** sets `BALATROBOT_GAMESPEED=8` and stops — which is precisely the lever your profiling proved doesn't touch the blocking path. They tried 16× and 100× and got stalls and desyncs, and settled on 8×. So a meaningful chunk of their ~714 ms/step is **the exact bottleneck you diagnosed and fixed, undiagnosed**.

**This means your patch is directly applicable to a balatrobot stack and is potentially a large win** — plausibly the single highest-leverage thing you own. Two caveats:

- balatrobot already needs three local crash patches at high speed, because deferred animation events fire after a fast programmatic transition destroyed the UI object they reference. Draining aggressively pushes on exactly that class of bug. Expect to find some.
- Those patches live in the mod directory and get **silently wiped by a mod update**. Script their re-application.

**balatroagent** doesn't care at all. Their sim has no frames, no animations, no event queue — a step is a pure function call over a struct. That is where 48,000 steps/sec comes from.

Which leads to the honest framing: **your best piece of engineering is a well-executed mitigation for a strategic choice.** A 4.6× speedup on 3.4 steps/sec is excellent work. It is also 0.03% of what removing the game engine buys.

---

## 5. Path B — what would you actually improve over balatroagent's sim?

You asked the right question: *would building my own be a major improvement in performance and accuracy, enough to justify the time?* Let me answer it with their actual gaps rather than in principle.

### Real gaps, from their own documentation

Under **"Not expressible in the v1 action space"** (`train/README.md`):

1. **Joker reordering.** `MOVE_JOKER` is a *reserved* action type — declared, never legal. *"inventory order is kept as-is"* and *"the sim auto-orders jokers."*

   **This is the big one, and it's a Phase-2 problem specifically.** Joker order determines the order effects apply (chips → mult → xmult, left to right), and Blueprint/Brainstorm copy *positionally adjacent* jokers. Naneinf builds are substantially about stacking xmult in the correct order with Blueprint chains. Balatron implements joker-order optimization heuristically and treats it as load-bearing. balatroagent's agent structurally cannot reorder jokers.

2. **Buy-and-use from the shop** — buying a planet and using it immediately is a real line the action space can't express.
3. **Boss reroll** (Director's Cut / Retcon voucher) — and *"boss blind manipulation"* is explicitly on your Phase-2 list.
4. **To Do List's target hand** absent from the observation; **flipped-joker id hiding** not modeled (Amber Acorn ids stay visible). Minor fidelity issues.

### Scope gaps

5. **Red Deck / White Stake only.** Fine for Phase 1 by definition. Phase-2 naneinf builds often want a specific deck.
6. **Base-game 150 jokers only.** No mod content.
7. **⚠️ Endless mode — verify this first.** Their episode *terminates* at the ante-8 boss (`win_ante`, and a win *"reports the post-boss ante (9)"*). Phase 2 — pushing toward naneinf — happens in **endless**, antes 9 through 20+, where blinds keep scaling and scaling jokers compound. I could not confirm from their docs that the sim supports playing past ante 8 at all. **If it doesn't, that's the one gap that genuinely bears on your Phase-2 goal**, and it's the first thing to check.
8. Models a **fully unlocked + discovered profile**, so the live game must be pinned to match.

### But here's the reframe that matters

**None of those justify writing a simulator from scratch. They justify forking theirs.**

Compare the two jobs honestly:

| | Fork + extend | Build from scratch |
|---|---|---|
| Add `MOVE_JOKER` + reordering | ~a week incl. learning enough Rust | — |
| Add endless mode | days–weeks | — |
| RNG oracle bit-exact to the game | **done, frozen** | weeks (they needed a pinned LuaJIT build because the system one's `math.random` *diverged from the game*) |
| Item registry from `game.lua` | **done, generated** | weeks |
| 150 joker ports w/ cited Lua line numbers | **done** | months |
| Cross-validation harness vs. live game | **done** | weeks |
| Test suite | **294 tests, done** | months |
| Validation status | live-verified ante 1→8, **zero divergences** | zero |

And the accuracy argument runs *against* building your own, not for it. Their sim has been diffed against the real game after every action across 10 seeds × antes 1–4, plus a full live ante-1→8 run with zero divergences. Yours would begin at zero validation — and the failure mode of an unvalidated sim is the worst one in RL: **your agent learns to exploit a bug you don't know you wrote**, and you don't find out until it plays the real game and collapses.

They also found real bugs *through* that harness that would have silently corrupted training: *"Red Deck +1 discard (back.lua:264, discards/round = 4 not 3 — affects training!)"*. That's a one-line difference that changes every episode. You would have made several of those and caught none.

### The direct answer to your question

**No — building your own would not be a major improvement in performance or accuracy for Phase 1.** You would spend months to arrive at approximately the same 71%, minus the validation. The only honest justifications for from-scratch are:

- You want sim-building to *be* the project (legitimate, but it's a different project than "train an agent to hit naneinf");
- Or you need something so architecturally different that extending theirs is harder than starting over — and nothing on the gap list is close to that bar.

### So the real option is Path A′

**Fork balatroagent's sim and extend it.** Sequence:

1. **Verify endless-mode support.** This is the gate — it's the only gap that touches Phase 2 fundamentally. If the sim can't play past ante 8, find out how hard that is to add *before* committing.
2. Build it, benchmark on your hardware, confirm you get the throughput.
3. Reproduce their result to validate your setup end-to-end.
4. Add `MOVE_JOKER` + joker reordering — Phase-2 critical, and a well-scoped first Rust task.
5. Then extend toward endless / naneinf.

It's MIT licensed (Copyright 2026 Jahan Kazimi), so this is all permitted. Attribute it.

---

## 6. If you cherry-pick both onto the live game — what should you expect?

Let's do the arithmetic rather than assert.

### Step 1 — realistic live throughput

- Port the drain patch to balatrobot: optimistically it holds up. But full-game steps include shop, packs, round-eval and blind-select — more animation-bound than hand play. Call it **8–10 steps/s per instance**, generous.
- `--headless --no-shaders` helps some.
- 3 parallel instances, scaling sublinearly as Balatron measured (~2.2× for 3): **~20 steps/s aggregate.**

### Step 2 — steps per calendar time

| Duration (continuous, 24/7) | Steps |
|---|---|
| 1 week | ~12M |
| 1 month | ~52M |
| 3 months | ~155M |

Minus downtime. Balatron's supervisor exists because crashes, desyncs, orphan processes, mod patches getting wiped, and *Steam leaking 13 GB of RAM* are constant, real taxes on uptime. Assume 70–85% realized.

### Step 3 — calibrate against known points

| Project | Steps | Result |
|---|---|---|
| Balatron | ~18.4M | 1.0% win, mean ante 4.28 |
| **You, 3 months live** | **~155M** | **?** |
| balatroagent (m4 certified) | ~1,478M | ~45% at the ≥25% gate |
| balatroagent (m5) | 2,180M | 64.5–68% |
| balatroagent (m5e2 certified) | 4,000M | 71.0% |

You'd land at **~8× Balatron's total** and **~1/26th of balatroagent's**. On balatroagent's own curriculum ladder (1→2→3→5→8, promoting at 70% held-out win rate), 155M steps is early — plausibly around the `win_ante` 3→5 region.

### Step 4 — the estimate

**Mean ante 5–6, win rate somewhere in the 3–10% range, after a multi-month campaign.**

Better than Balatron, for reasons that are mechanistic rather than hopeful:

- The autoregressive action space removes action projection entirely — every sampled action is legal, so PPO's ratio is computed on the action that actually ran.
- No heuristic overrides means the policy's choices actually influence outcomes, so the gradient isn't near-zero (Balatron spent months discovering their gradient was honestly ~0 and building BC/prior-KL machinery to work around it).
- The ante curriculum gives dense reward early instead of a <1% terminal signal.
- Snapshot start-state mixing puts the agent in deep-ante states it can't yet reach on its own.
- Balatron's own diagnosis is that its remaining wall may be *architectural* — a depth-1 marginal build evaluator cannot value synergy — and the cherry-picked design doesn't have that limitation.

**An order of magnitude short of 71%. And Phase 2 (naneinf) is not reachable on this path** — naneinf requires deep endless-mode play with compounding scaling engines, which is far more sample-hungry than clearing ante 8.

### How confident am I in that number?

**Moderately, and the interval is wide.** It's an extrapolation across a 26× gap in sample count, not a measurement.

It could be **better** than I've said if the curriculum plus snapshot mixing are more sample-efficient than I'm crediting — they are specifically engineered to be, and balatroagent didn't need them at 4B steps, so their true value at 100M is genuinely untested.

It could be **worse** if live-game instability eats your uptime, or if throughput at full scope lands nearer 5 steps/s than 10.

### Two things worth saying about this path

**First: it isn't wasted work even if you later move to a sim.** Everything in the appendix — action space, encoding contract, reward design, evaluation rig, telemetry — is environment-agnostic. You would port all of it. So "cherry-pick onto live now, sim later" is a coherent sequence, not a detour.

**Second: there's a real argument for it that isn't about the number.** You own every part of the stack. You'd learn far more about RL debugging, reward design and credit assignment than you would about Rust. Balatron's `DECISIONS.md` is 164KB of hard-won lessons precisely *because* that path forces you to confront every one of them. If part of the point of this project is the journey, that's legitimate.

Just go in with the ceiling written down in advance, so that hitting ante 5 at month three reads as *expected* rather than as failure.
