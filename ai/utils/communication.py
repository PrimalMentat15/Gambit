"""
Balatro Socket Communication Abstraction

Pure line-delimited JSON I/O layer over a localhost TCP socket for communicating
with the Balatro mod. Handles low-level transport without any game logic or AI logic.

Python acts as the server (it owns the training lifecycle); the Balatro mod connects
as a client and reconnects on game relaunch.

This abstraction can be used by:
- balatro_env.py (for SB3 training)
- file_watcher.py (for testing)
- Any other component that needs to talk to Balatro
"""

import json
import logging
import os
import socket
from typing import Dict, Any, Optional

from ..telemetry import EventType, Stopwatch, emit

DEFAULT_HOST = os.environ.get("BALATRO_RL_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("BALATRO_RL_PORT", "12345"))


class BalatroSocketIO:
    """
    Clean abstraction for socket communication with the Balatro mod

    Responsibilities:
    - Accept a connection from the Balatro mod
    - Read JSON requests from the mod
    - Write JSON responses back to the mod
    - Keep the connection open persistently across an episode
    - Re-accept transparently when Balatro restarts
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)

        self.server: Optional[socket.socket] = None
        self.client: Optional[socket.socket] = None
        self.reader = None
        self.writer = None

        # Latency instrumentation. t_wait is time blocked on the game and is
        # expected to dominate the step budget; t_send is our own write cost.
        self.last_send_s = 0.0
        self.last_wait_s = 0.0
        self.round_trips = 0
        self._wait_watch = Stopwatch()

        self.start_server()
        self.accept_connection()

    def start_server(self) -> None:
        """
        Bind and listen on the configured host/port

        Safe to call once per instance; raises if the port is already taken.
        """
        try:
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind((self.host, self.port))
            self.server.listen(1)
            self.logger.info(f"Listening on {self.host}:{self.port}")
        except Exception as e:
            self.logger.error(f"Failed to bind {self.host}:{self.port}: {e}")
            self.cleanup()
            raise RuntimeError(f"Could not start communication server: {e}")

    def accept_connection(self, timeout: Optional[float] = None) -> None:
        """
        Wait for the Balatro mod to connect

        Replaces any previous connection. Disables Nagle so the
        request/response ping-pong is not delayed.

        Args:
            timeout: Seconds to wait, or None to block indefinitely. A bounded
                wait is what lets a mid-run disconnect be retried rather than
                hanging forever on a game that is never coming back.

        Raises:
            TimeoutError: If no connection arrived within the timeout
            RuntimeError: If accepting failed for any other reason
        """
        self.close_client()

        try:
            self.logger.info("🔧 Waiting for Balatro to connect...")
            self.logger.info("   Press 'R' in Balatro now to activate RL training!")

            self.server.settimeout(timeout)
            try:
                self.client, addr = self.server.accept()
            finally:
                # Leave the listening socket blocking for any other caller
                self.server.settimeout(None)
            self.client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.reader = self.client.makefile("r", encoding="utf-8", newline="\n")
            self.writer = self.client.makefile("w", encoding="utf-8", newline="\n")

            self.logger.info(f"Balatro connected from {addr[0]}:{addr[1]}")
            emit(EventType.COMM, event="connected", peer=f"{addr[0]}:{addr[1]}")
        except socket.timeout:
            # Distinct from a real failure: the caller may want to retry
            self.close_client()
            raise TimeoutError(f"No Balatro connection within {timeout}s")
        except Exception as e:
            self.logger.error(f"Failed to accept connection: {e}")
            self.close_client()
            raise RuntimeError(f"Could not accept Balatro connection: {e}")

    def ensure_connected(self, timeout: Optional[float] = None) -> bool:
        """
        Re-accept a connection if the previous one was dropped

        Args:
            timeout: Seconds to wait for a new connection, None to block

        Returns:
            True if a live connection is available
        """
        if self.client:
            return True

        try:
            self.accept_connection(timeout=timeout)
            return True
        except Exception:
            return False

    def wait_for_request(self, accept_timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Wait for a new request from the Balatro mod

        Blocks until Balatro writes a request to the socket.

        Args:
            accept_timeout: If a reconnection is needed, how long to wait for it

        Returns:
            Parsed JSON request data, or None on disconnect/error
        """
        if not self.ensure_connected(timeout=accept_timeout):
            return None

        try:
            # Everything between our response and the game's next request is
            # game-side time: animations, event queue, frames where the state
            # hash did not change. This is the number that matters.
            self._wait_watch.start()
            request_line = self.reader.readline().strip()
            self.last_wait_s = self._wait_watch.stop()

            if not request_line:
                # Empty read means the peer closed the connection
                self.logger.warning("Balatro disconnected")
                emit(EventType.COMM, event="disconnected")
                self.close_client()
                return None

            request_data = json.loads(request_line)
            self.round_trips += 1
            self.logger.debug(f"📥 RECEIVED REQUEST: {request_line}")
            return request_data

        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in request: {e}")
            emit(EventType.COMM, event="bad_json", error=str(e))
            return None
        except Exception as e:
            self.logger.error(f"Error reading from socket: {e}")
            emit(EventType.COMM, event="read_error", error=str(e))
            self.close_client()
            return None

    def send_response(self, response_data: Dict[str, Any]) -> bool:
        """
        Send a response back to the Balatro mod

        Args:
            response_data: Response dictionary to send

        Returns:
            True if successful, False if error
        """
        if not self.ensure_connected():
            self.logger.error("No connection to Balatro")
            return False

        watch = Stopwatch().start()
        try:
            json.dump(response_data, self.writer)
            self.writer.write("\n")  # Important: newline delimits the message
            self.writer.flush()  # Force write to socket immediately
            self.last_send_s = watch.stop()

            self.logger.debug(f"📤 SENT RESPONSE: {json.dumps(response_data)}\n\n")
            return True

        except Exception as e:
            self.last_send_s = watch.stop()
            self.logger.error(f"❌ ERROR sending response: {e}")
            import traceback
            self.logger.error(f"❌ TRACEBACK: {traceback.format_exc()}")
            emit(EventType.COMM, event="send_error", error=str(e))
            self.close_client()
            return False

    def timings(self) -> Dict[str, float]:
        """
        Latency of the most recent round trip

        Returns:
            Dict with t_send (our write cost) and t_wait (blocked on the game)
        """
        return {"t_send": self.last_send_s, "t_wait": self.last_wait_s}

    def close_client(self):
        """
        Close the current client connection and its stream wrappers
        """
        for stream in (self.reader, self.writer):
            if stream:
                try:
                    stream.close()
                except Exception as e:
                    self.logger.debug(f"Failed to close stream: {e}")
        self.reader = None
        self.writer = None

        if self.client:
            try:
                self.client.close()
                self.logger.debug("Closed client socket")
            except Exception as e:
                self.logger.warning(f"Failed to close client socket: {e}")
            self.client = None

    def cleanup(self):
        """
        Close the client connection and release the listening port
        """
        self.close_client()

        if self.server:
            try:
                self.server.close()
                self.logger.debug("Closed server socket")
            except Exception as e:
                self.logger.warning(f"Failed to close server socket: {e}")
            self.server = None

        self.logger.debug("Socket communication cleanup complete")


# Backwards-compatible alias for the previous named-pipe implementation
BalatroPipeIO = BalatroSocketIO
