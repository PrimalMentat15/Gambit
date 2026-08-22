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

import monitor  # noqa: F401  (puts train/ on sys.path)
from balatro_train.telemetry import find_latest_run  # noqa: E402
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
        self.returns = []
        self.rate = 0.0
        self.latest = {}
        self.milestone = {}
        self.promotions = 0
        self.target = 0
        self.status = None

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

        # Reads are capped (see TailReader.MAX_CHUNK), so catching up on an
        # existing file takes several passes. Drain them here rather than
        # advancing one chunk per HTTP poll, which would leave the page
        # minutes behind a long run.
        while True:
            events, restarted = self.reader.read()
            if restarted:
                self.reset()
            for event in events:
                self._absorb(event)
            if not self.reader.pending:
                break

    def _absorb(self, event: dict) -> None:
        kind = event.get("type")
        data = event.get("data", {})

        if kind == "session_start":
            self.target = data.get("total_timesteps", 0) or 0

        elif kind == "rollout":
            self.latest = data
            self.steps = data.get("global_step", self.steps)
            # sps_inst (rate over the last iteration) when present; sps
            # (session average) for older streams. See ppo.py's train loop.
            self.rate = float(data.get("sps_inst") or data.get("sps") or 0)
            if data.get("ep_return_mean") is not None:
                self.returns.append(data["ep_return_mean"])
                if len(self.returns) > 400:
                    self.returns = self.returns[-400:]

        elif kind == "episode_end":
            self.episodes += 1
            if data.get("won"):
                self.wins += 1

        elif kind == "milestone_eval":
            self.milestone = data

        elif kind == "curriculum_promotion":
            self.promotions += 1

        elif kind == "session_end":
            self.status = data.get("status")

    def snapshot(self) -> dict:
        """Everything the page needs, as plain JSON"""
        remaining = max(0, self.target - self.steps)
        eta = remaining / self.rate if self.rate > 0 and remaining else 0

        recent = self.returns[-120:]
        return {
            "run_id": self.run_id or "-",
            "status": self.status or ("running" if self.steps else "-"),
            "steps": self.steps,
            "target": self.target,
            "iteration": self.latest.get("iteration", 0),
            "episodes": self.episodes,
            "wins": self.wins,
            # Training win rate is against the current curriculum goal; the
            # milestone below is the one comparable across the whole run.
            "win_rate": (self.latest.get("ep_win_rate") or 0) * 100,
            "win_ante": self.latest.get("win_ante"),
            "promotions": self.promotions,
            "milestone_win_rate": (self.milestone.get("win_rate") or 0) * 100,
            "milestone_step": self.milestone.get("step", 0),
            "ante_mean": self.latest.get("ep_ante_mean", 0),
            "rate": self.rate,
            "eta_seconds": eta,
            "mean_return": (sum(recent) / len(recent)) if recent else 0.0,
            "returns": recent,
            "entropy": self.latest.get("entropy", 0),
            "approx_kl": self.latest.get("approx_kl", 0),
            "shaping_beta": self.latest.get("shaping_beta"),
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
    parser.add_argument("--runs-dir", default=os.path.join("train", "runs"))
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
