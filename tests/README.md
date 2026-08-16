# Tests

This directory holds the monitor's tests. The trainer, the telemetry stream and
the simulator have their own suites:

```bash
cd sim && cargo test
```

```bash
cd train && pytest
```

| Test | Covers |
|------|--------|
| `test_monitor.py` | Tail reader (incremental reads, partial lines, split UTF-8, restart detection), panel discovery, and an offscreen render of the real window to a PNG |

`test_monitor.py` needs the `balatro` conda env active (see the repo README). It renders without
putting a window on screen and writes a screenshot per tab, so the dashboard can
actually be looked at rather than merely asserted on:

```bash
python tests/test_monitor.py train/runs/<run_id> -o preview.png
```

Pass a run directory to render real data, or omit it to use generated data.
Set `BALATRO_RL_HEADLESS=1` on a machine with no display at all — the offscreen
Qt platform loads no system fonts on Windows, so every label renders as a tofu
box, which is why it is not the default.
