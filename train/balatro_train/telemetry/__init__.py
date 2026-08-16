"""
Balatro RL telemetry

Durable, drop-tolerant event stream shared by the trainer, the monitor UI and the
analysis tools. Stdlib only, by design: instrumenting the trainer must not add
install burden.

Typical trainer usage:

    from balatro_train.telemetry import (
        EventType, RunSession, TelemetryEmitter, emit, set_emitter,
    )

    session = RunSession.for_run(name="m3_fast", runs_dir="runs", config={...})
    set_emitter(TelemetryEmitter(session.events_path))
    emit(EventType.SESSION_START, run_id=session.run_id)
"""

from .events import ALL_TYPES, SCHEMA_VERSION, EventType, make_event
from .emitter import (
    NullEmitter,
    TelemetryEmitter,
    close_emitter,
    emit,
    get_emitter,
    set_emitter,
)
from .session import (
    DEFAULT_RUNS_DIR,
    RunSession,
    find_latest_run,
    list_runs,
)
from .timing import Stopwatch, phase

__all__ = [
    "ALL_TYPES",
    "SCHEMA_VERSION",
    "EventType",
    "make_event",
    "NullEmitter",
    "TelemetryEmitter",
    "close_emitter",
    "emit",
    "get_emitter",
    "set_emitter",
    "DEFAULT_RUNS_DIR",
    "RunSession",
    "find_latest_run",
    "list_runs",
    "Stopwatch",
    "phase",
]
