# Tests

No game required — each test stands up a fake Balatro client over the real socket.

Run from the repo root:

```bash
venv/Scripts/python.exe tests/test_telemetry.py
```

| Test | Covers |
|------|--------|
| `test_telemetry.py` | Event schema and sequencing, drop-on-full under flood, emit cost, writer surviving unserializable payloads, run-directory creation and discovery |
| `test_comm.py` | Socket round trip, disconnect detection, automatic re-accept on reconnect |
| `test_e2e.py` | Real `BalatroEnv` driven by a fake game with a simulated 480 ms delay; asserts the latency breakdown is captured and that Python overhead stays under 5 ms/step |
| `test_monitor.py` | Tail reader (incremental reads, partial lines, split UTF-8, restart detection), panel discovery, and an offscreen render of the real window to a PNG |
| `test_supervisor.py` | Kill switch against a process that ignores SIGINT, re-attach from `meta.json`, concurrent web-view readers, and analysis over a real run |
| `test_reconnect.py` | Card-slot decoding to a legal selection, observation size stable across hand sizes 0-100, reconnect after the game is closed and relaunched, and a bounded accept that times out rather than hanging |
| `test_autoregressive.py` | Every sampled action legal (1-5 cards, no duplicates, in-hand only), action-type masking, `log_prob` reproducibility between sampling and PPO's recompute, forced slots contributing zero log-prob, gradient flow, the mapper's clamp backstop staying unused, and incompatible checkpoints being skipped rather than crashing the run |

`test_comm.py` and `test_e2e.py` bind ports 12399 and 12455, so stop any running
trainer first.

`test_monitor.py` needs `requirements-monitor.txt` installed. It runs headless via
`QT_QPA_PLATFORM=offscreen` — no display needed — and writes a screenshot so the
dashboard can actually be looked at rather than merely asserted on:

```bash
venv/Scripts/python.exe tests/test_monitor.py runs/<run_id> -o preview.png
```

Pass a run directory to render real data, or omit it to use generated data.

Then inspect what the e2e run recorded:

```bash
venv/Scripts/python.exe -m ai.tools.latency_report <run dir printed by the test>
```
