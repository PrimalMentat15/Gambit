"""
Telemetry event schema

Every event is a single JSON object on its own line:

    {"v": 1, "seq": 1234, "t": 1786523417.123, "type": "step", "data": {...}}

The schema is the contract between the trainer and every consumer (monitor UI,
analysis tools, web view). Consumers must ignore unknown event types and unknown
fields inside ``data`` so that adding a metric never breaks an older reader.
"""

SCHEMA_VERSION = 1


class EventType:
    """Event type constants"""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    STEP = "step"
    EPISODE_END = "episode_end"
    ROLLOUT = "rollout"
    COMM = "comm"
    LOG = "log"
    GAME = "game"


ALL_TYPES = (
    EventType.SESSION_START,
    EventType.SESSION_END,
    EventType.STEP,
    EventType.EPISODE_END,
    EventType.ROLLOUT,
    EventType.COMM,
    EventType.LOG,
    EventType.GAME,
)


def make_event(seq: int, timestamp: float, event_type: str, data: dict) -> dict:
    """
    Build a schema-conformant event envelope

    Args:
        seq: Monotonic sequence number, unique within a session
        timestamp: Unix timestamp (seconds, float)
        event_type: One of EventType
        data: Event payload

    Returns:
        Event dictionary ready to serialize
    """
    return {
        "v": SCHEMA_VERSION,
        "seq": seq,
        "t": timestamp,
        "type": event_type,
        "data": data,
    }
