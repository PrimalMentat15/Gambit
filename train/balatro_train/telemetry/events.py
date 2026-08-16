"""
Telemetry event schema

Every event is a single JSON object on its own line:

    {"v": 1, "seq": 1234, "t": 1786523417.123, "type": "rollout", "data": {...}}

The schema is the contract between the trainer and every consumer (monitor UI,
analysis tools, web view). Consumers must ignore unknown event types and unknown
fields inside ``data`` so that adding a metric never breaks an older reader.
"""

SCHEMA_VERSION = 1


class EventType:
    """Event type constants"""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    EPISODE_END = "episode_end"
    ROLLOUT = "rollout"
    CURRICULUM_PROMOTION = "curriculum_promotion"
    PROMOTION_EVAL = "promotion_eval"
    MILESTONE_EVAL = "milestone_eval"
    CHECKPOINT_SAVED = "checkpoint_saved"
    LOG = "log"


ALL_TYPES = (
    EventType.SESSION_START,
    EventType.SESSION_END,
    EventType.EPISODE_END,
    EventType.ROLLOUT,
    EventType.CURRICULUM_PROMOTION,
    EventType.PROMOTION_EVAL,
    EventType.MILESTONE_EVAL,
    EventType.CHECKPOINT_SAVED,
    EventType.LOG,
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
