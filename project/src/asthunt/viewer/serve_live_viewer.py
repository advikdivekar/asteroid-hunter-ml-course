"""A small web server for the live viewer: the page, the run's state as JSON, the pictures, and start / next / stop buttons."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from asthunt.viewer.live_state import LiveState, RunCancelled
from asthunt.viewer.run_traced_night import STAGE_PLAN

PAGE = Path(__file__).with_name("live_viewer.html")


def make_server(host, port, units, run_unit):
    """Return (server, state). run_unit(unit, state, n_fakes) does the actual work; it is started in a thread by the Start button."""
    state = LiveState()
    worker = {"thread": None}

    def start(request):
        if worker["thread"] is not None and worker["thread"].is_alive():
            return 409, {"error": "a run is already going; stop it first"}
        unit = next((u for u in units if u["tag"] == request.get("unit")), None)
        if unit is None:
            return 400, {"error": "unknown unit"}
        n_fakes = max(0, min(int(request.get("fakes", 0)), 40))
        state.reset(STAGE_PLAN, unit, bool(request.get("step_mode", True)))

        def work():
            try:
                run_unit(unit, state, n_fakes)
            except RunCancelled:
                state.mark_cancelled()
            except Exception as error:                      # the stage that failed already wrote its own note
                state.finish(f"stopped: {type(error).__name__}: {error}")

        worker["thread"] = threading.Thread(target=work, daemon=True)
        worker["thread"].start()
        return 200, {"ok": True}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, content_type="application/json"):
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/state":
                self._send(200, state.snapshot())
            elif self.path == "/api/units":
                self._send(200, units)
            elif self.path.startswith("/img/"):
                png = state.get_image(self.path[len("/img/"):])
                if png:
                    self._send(200, png, "image/png")
                else:
                    self._send(404, {"error": "no such picture"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/start":
                self._send(*start(request))
            elif self.path == "/api/next":
                state.release()
                self._send(200, {"ok": True})
            elif self.path == "/api/mode":
                state.set_step_mode(bool(request.get("step_mode")))
                self._send(200, {"ok": True})
            elif self.path == "/api/stop":
                state.cancel()
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, format, *args):               # keep the terminal for the run's own messages
            pass

    return ThreadingHTTPServer((host, port), Handler), state
