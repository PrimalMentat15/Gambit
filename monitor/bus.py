"""
Event bus

Tails a run's events.jsonl and delivers batches of events to panels as Qt signals.

The trainer appends to the file and never blocks on us, so tailing is the whole
integration: the monitor can start late, restart, or die entirely without
affecting training. Reading is incremental via a byte offset, and a file that
shrinks is treated as a new run.
"""

import json
import os
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QThread, Signal


_TYPE_KEY = b'"type"'


def _peek_type(line: bytes):
    """Extract the envelope's ``type`` without parsing the whole line.

    Parsing is the expensive part -- ~955 bytes of Python objects per event --
    so the bulk filter has to decide BEFORE calling json.loads. Scans for the
    envelope key and reads the quoted value after it, which tolerates both
    ``"type": "x"`` (json.dumps default) and compact ``"type":"x"``.

    Returns the type as bytes, or None if the line does not look like an
    envelope (in which case it is parsed normally rather than dropped).
    """
    i = line.find(_TYPE_KEY)
    if i < 0:
        return None
    start = line.find(b'"', i + len(_TYPE_KEY))
    if start < 0:
        return None
    end = line.find(b'"', start + 1)
    if end < 0:
        return None
    return line[start + 1:end]


class TailReader:
    """
    Incremental newline-delimited JSON reader

    Pure Python with no Qt dependency so it can be tested directly. Reads in
    binary and only decodes complete lines, which avoids splitting a multi-byte
    character across two reads.
    """

    #: Never hold more than this much of the file in memory at once. Without a
    #: cap, attaching to a long run reads the whole file in ONE call -- a 1.1 GB
    #: events.jsonl costs ~7.5 GB peak (raw chunk + split list + parsed dicts),
    #: which on a 16 GB box starves the trainer's page-locked H2D buffers and
    #: surfaces as a CUDA OOM in a process the monitor never touches.
    MAX_CHUNK = 4 << 20

    #: How much of the tail to parse in FULL. Everything earlier goes through
    #: the bulk filter below.
    BACKFILL_BYTES = 16 << 20

    #: Types too numerous to replay, that nothing accumulates run-wide.
    #: `episode_end` is 99.95% of a mature run's file, and its only consumer
    #: (panels/reward.py) holds a bounded RingSeries -- so replaying millions
    #: of them just fills a ring that discards all but the last few hundred.
    #: The tail window still delivers those.
    BULK_TYPES = frozenset({b"episode_end"})

    def __init__(self, path: str, max_chunk: int = 0, backfill_bytes: int = 0):
        self.path = path
        self.max_chunk = max_chunk or self.MAX_CHUNK
        self.backfill_bytes = backfill_bytes or self.BACKFILL_BYTES
        self._offset = 0
        self._buffer = b""
        self._full_from = 0
        self.malformed = 0
        self.skipped = 0
        self.pending = False

    def rewind(self) -> None:
        """Start reading from the beginning again"""
        self._offset = 0
        self._buffer = b""
        self._full_from = 0

    def read(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Read whatever has been appended since the last call

        Returns:
            (events, restarted) where restarted is True if the file shrank or
            vanished, meaning consumers should clear their state
        """
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return [], False

        restarted = False
        if size < self._offset:
            # Truncated or replaced by a new run
            self.rewind()
            restarted = True

        # On first contact with an existing file, everything before this offset
        # is "history" and gets the bulk filter; the tail is parsed in full.
        if self._offset == 0 and size > self.backfill_bytes:
            self._full_from = size - self.backfill_bytes

        if size == self._offset:
            self.pending = False
            return [], restarted

        want = min(size - self._offset, self.max_chunk)
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read(want)
        except OSError:
            self.pending = False
            return [], restarted

        self._offset += len(chunk)
        self.pending = self._offset < size

        # A chunk is contiguous, so one comparison classifies all of it. The
        # chunk straddling the boundary counts as full, erring toward more data.
        history = self._offset <= self._full_from
        data = self._buffer + chunk
        lines = data.split(b"\n")

        # The final element is whatever follows the last newline: either empty
        # or a line the writer has not finished yet
        self._buffer = lines.pop()

        events = []
        for line in lines:
            # Classify BEFORE strip(): stripping allocates a fresh bytes
            # object per line, and on a mature run 98% of them are about to be
            # dropped. _peek_type does not care about surrounding whitespace.
            if history and _peek_type(line) in self.BULK_TYPES:
                self.skipped += 1
                continue
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.malformed += 1
        return events, restarted


class TailThread(QThread):
    """Polls a TailReader off the GUI thread"""

    batch = Signal(list)
    restarted = Signal()

    def __init__(self, path: str, poll_ms: int = 100, parent=None):
        super().__init__(parent)
        self.path = path
        self.poll_ms = poll_ms
        self._stop = threading.Event()

    def run(self) -> None:
        reader = TailReader(self.path)
        while not self._stop.is_set():
            events, restarted = reader.read()
            if restarted:
                self.restarted.emit()
            if events:
                self.batch.emit(events)
            if reader.pending:
                # Still catching up on an existing file. Keep draining without
                # the poll delay -- 1 GB at MAX_CHUNK is a few hundred reads,
                # and each one is bounded, so this stays responsive instead of
                # stalling for minutes on a single enormous read.
                continue
            self._stop.wait(self.poll_ms / 1000.0)

    def stop(self, timeout_ms: int = 2000) -> None:
        """Signal the loop to exit and wait for it"""
        self._stop.set()
        self.wait(timeout_ms)


class EventBus(QObject):
    """
    Owns the tail thread and republishes events to the application

    Switching runs tears down the old thread and starts a new one, emitting
    ``restarted`` so panels clear stale series.
    """

    batch = Signal(list)
    restarted = Signal()
    source_changed = Signal(str)

    def __init__(self, poll_ms: int = 100, parent=None):
        super().__init__(parent)
        self.poll_ms = poll_ms
        self.path: Optional[str] = None
        self._thread: Optional[TailThread] = None

        self.total_events = 0
        # Bounded: a catch-up batch can carry many thousands of events, and an
        # unbounded list here would grow as fast as the thing it measures.
        self._recent: Deque[float] = deque(maxlen=20000)

    def set_source(self, path: Optional[str]) -> None:
        """
        Point the bus at a different events.jsonl

        Passing the current path is a no-op so periodic run rescans do not
        constantly restart the tail.
        """
        if path == self.path:
            return

        self.stop()
        self.path = path
        self.total_events = 0
        self._recent.clear()
        self.restarted.emit()

        if not path:
            self.source_changed.emit("")
            return

        self._thread = TailThread(path, self.poll_ms, parent=self)
        self._thread.batch.connect(self._on_batch)
        self._thread.restarted.connect(self.restarted)
        self._thread.start()
        self.source_changed.emit(path)

    def _on_batch(self, events: List[Dict[str, Any]]) -> None:
        """Track throughput, then hand the batch on"""
        now = time.monotonic()
        self.total_events += len(events)
        self._recent.extend([now] * min(len(events), self._recent.maxlen))

        cutoff = now - 5.0
        while self._recent and self._recent[0] < cutoff:
            self._recent.popleft()

        self.batch.emit(events)

    @property
    def events_per_sec(self) -> float:
        """Event rate over the last few seconds"""
        if len(self._recent) < 2:
            return 0.0
        span = self._recent[-1] - self._recent[0]
        if span <= 0:
            return 0.0
        return len(self._recent) / span

    def stop(self) -> None:
        """Tear down the tail thread"""
        if self._thread is not None:
            self._thread.stop()
            self._thread = None
