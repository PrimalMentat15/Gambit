"""
Read-only remote view

Serves one self-contained page that streams live run stats, so a multi-hour
session can be checked from a phone on the same network.

Stdlib only, and Server-Sent Events rather than WebSockets: the stream is
one-way, SSE needs no handshake library and no extra dependency, and it
reconnects on its own. Nothing here can modify a run -- there is no route that
writes, and the kill switch is deliberately not exposed over the network.

    python -m monitor.web              # bind localhost
    python -m monitor.web --host 0.0.0.0 --port 8770
"""

import argparse
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.telemetry import find_latest_run  # noqa: E402
from monitor.bus import TailReader  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


class State:
    """
    Rolling summary of the active run, rebuilt as events arrive

    One instance is shared by every connected viewer, since ThreadingHTTPServer
    runs each connection on its own thread. Reads are serialised and the result
    cached briefly, so N viewers cost one file read rather than N -- and, more
    importantly, two threads cannot interleave inside the same TailReader and
    corrupt its offset and partial-line buffer.
    """

    # How long a snapshot is reused before the file is polled again
    CACHE_TTL = 0.5

    def __init__(self, runs_dir: str):
        self.runs_dir = runs_dir
        self.run_id: Optional[str] = None
        self.reader: Optional[TailReader] = None
        self._lock = threading.Lock()
        self._cached: Optional[dict] = None
        self._cached_at = 0.0
        self.reset()

    def snapshot_cached(self) -> dict:
        """Poll and summarise, reusing a recent result across concurrent viewers"""
        with self._lock:
            now = time.monotonic()
            if self._cached is None or now - self._cached_at >= self.CACHE_TTL:
                self.poll()
                self._cached = self.snapshot()
                self._cached_at = now
            return self._cached

    def reset(self) -> None:
        self.steps = 0
        self.episodes = 0
        self.wins = 0
        self.rewards = []
        self.rate_samples = []
        self.action_wait = {}
        self.last_step = {}
        self.target = 0

    def poll(self) -> None:
        """Follow the newest run and absorb anything new"""
        session = find_latest_run(self.runs_dir, active_only=True) or find_latest_run(self.runs_dir)
        if session is None:
            return

        if session.run_id != self.run_id:
            self.run_id = session.run_id
            self.reader = TailReader(session.events_path)
            self.reset()

        if self.reader is None:
            return

        events, restarted = self.reader.read()
        if restarted:
            self.reset()

        for event in events:
            self._absorb(event)

    def _absorb(self, event: dict) -> None:
        kind = event.get("type")
        data = event.get("data", {})

        if kind == "session_start":
            self.target = data.get("total_timesteps", 0) or 0

        elif kind == "step":
            self.steps += 1
            self.last_step = data
            stamp = event.get("t")
            if stamp:
                self.rate_samples.append(stamp)
                if len(self.rate_samples) > 60:
                    self.rate_samples = self.rate_samples[-60:]

            action = data.get("action")
            wait = (data.get("timings") or {}).get("t_wait")
            if action is not None and wait is not None:
                bucket = self.action_wait.setdefault(action, [0, 0.0])
                bucket[0] += 1
                bucket[1] += wait

        elif kind == "episode_end":
            self.episodes += 1
            if data.get("outcome") == "win":
                self.wins += 1
            if data.get("reward") is not None:
                self.rewards.append(data["reward"])
                if len(self.rewards) > 400:
                    self.rewards = self.rewards[-400:]

    def snapshot(self) -> dict:
        """Everything the page needs, as plain JSON"""
        rate = 0.0
        if len(self.rate_samples) >= 2:
            span = self.rate_samples[-1] - self.rate_samples[0]
            if span > 0:
                rate = (len(self.rate_samples) - 1) / span

        remaining = max(0, self.target - self.steps)
        eta = remaining / rate if rate > 0 and remaining else 0

        names = {1: "SELECT_HAND", 2: "PLAY_HAND", 3: "DISCARD_HAND"}
        actions = [
            {"name": names.get(a, str(a)), "mean_ms": total / count * 1000, "count": count}
            for a, (count, total) in sorted(self.action_wait.items())
        ]

        recent = self.rewards[-120:]
        return {
            "run_id": self.run_id or "-",
            "steps": self.steps,
            "target": self.target,
            "episodes": self.episodes,
            "wins": self.wins,
            "win_rate": (self.wins / self.episodes * 100) if self.episodes else 0.0,
            "rate": rate,
            "eta_seconds": eta,
            "mean_reward": (sum(recent) / len(recent)) if recent else 0.0,
            "rewards": recent,
            "actions": actions,
            "chips": self.last_step.get("chips", 0),
            "blind_chips": self.last_step.get("blind_chips", 0),
            "hands_left": self.last_step.get("hands_left", 0),
            "discards_left": self.last_step.get("discards_left", 0),
        }


class Handler(BaseHTTPRequestHandler):
    """Two routes: the page, and the event stream"""

    state: State = None
    protocol_version = "HTTP/1.1"

    def do_GET(self):  # noqa: N802 - required name
        if self.path.startswith("/events"):
            self._serve_stream()
        elif self.path in ("/", "/index.html"):
            self._serve_page()
        elif self.path in ("/icon.png", "/favicon.ico"):
            self._serve_icon()
        else:
            self.send_error(404)

    def _serve_icon(self) -> None:
        from monitor.resources import ICON_PNG

        try:
            with open(ICON_PNG, "rb") as handle:
                body = handle.read()
        except OSError:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def _serve_page(self) -> None:
        try:
            with open(os.path.join(HERE, "index.html"), "rb") as handle:
                body = handle.read()
        except OSError:
            self.send_error(500, "page missing")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            while True:
                payload = json.dumps(self.state.snapshot_cached())
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            # Client navigated away; SSE reconnects on its own
            return

    def log_message(self, fmt, *args):
        """Quieter than the default one-line-per-request logging"""
        return


def lan_address() -> str:
    """Best guess at this machine's LAN address, for the printed URL"""
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        address = probe.getsockname()[0]
        probe.close()
        return address
    except Exception:
        return "127.0.0.1"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Read-only remote view for Gambit runs")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address. Use 0.0.0.0 to allow other devices on your LAN.")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args(argv)

    Handler.state = State(args.runs_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)

    shown = lan_address() if args.host == "0.0.0.0" else args.host
    print(f"Remote view on http://{shown}:{args.port}")
    if args.host != "0.0.0.0":
        print("Bound to localhost only. Pass --host 0.0.0.0 to reach it from your phone.")
    print("Read-only: this server cannot start, stop or kill a run.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
