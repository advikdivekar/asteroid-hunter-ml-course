import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np

from asthunt.viewer import draw_stage_figures as draw
from asthunt.viewer.serve_live_viewer import make_server

UNITS = [{"tag": "u1", "label": "unit one", "field": 1, "ccd": 1, "quadrant": 1, "night": "2020-01-01", "frames": 5}]


def fake_run(unit, state, n_fakes):
    with state.stage("pick") as stage:
        stage.number("fakes asked for", n_fakes)
        stage.image("a picture", draw.draw_score_histogram(np.linspace(0, 1, 50), 0.5))
    state.finish("done")


def call(base, path, body=None):
    request = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request) as reply:
        return reply.read(), reply.headers["Content-Type"]


def wait_until(check, seconds=5):
    end = time.time() + seconds
    while time.time() < end:
        if check():
            return True
        time.sleep(0.05)
    return False


def test_start_then_read_state_and_picture():
    server, state = make_server("127.0.0.1", 0, UNITS, fake_run)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page, kind = call(base, "/")
        assert b"Asteroid hunter" in page and kind.startswith("text/html")
        assert json.loads(call(base, "/api/units")[0])[0]["tag"] == "u1"

        call(base, "/api/start", {"unit": "u1", "fakes": 7, "step_mode": False})
        assert wait_until(lambda: state.snapshot()["status"] == "finished")
        snapshot = json.loads(call(base, "/api/state")[0])
        stage = snapshot["stages"][0]
        assert stage["status"] == "done" and stage["numbers"] == [["fakes asked for", "7"]]
        png, kind = call(base, stage["images"][0]["url"])
        assert png.startswith(b"\x89PNG") and kind == "image/png"
    finally:
        server.shutdown()


def test_unknown_unit_and_missing_picture_are_refused():
    server, _ = make_server("127.0.0.1", 0, UNITS, fake_run)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        for path, body in [("/api/start", {"unit": "nope"}), ("/img/999.png", None)]:
            try:
                call(base, path, body)
                assert False, "should have been refused"
            except urllib.error.HTTPError as error:
                assert error.code in (400, 404)
    finally:
        server.shutdown()
