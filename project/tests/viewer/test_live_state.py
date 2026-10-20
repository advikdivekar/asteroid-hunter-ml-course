import threading
import time

import pytest

from asthunt.viewer.live_state import LiveState, RunCancelled

PLAN = [("a", "First", "does a"), ("b", "Second", "does b")]


def test_a_stage_records_its_numbers_and_ends_done():
    state = LiveState()
    state.reset(PLAN, {"label": "x"}, step_mode=False)
    with state.stage("a") as stage:
        stage.number("blobs", 12)
        stage.note("frame 1")
        stage.table(["k"], [[1]])
    first = state.snapshot()["stages"][0]
    assert first["status"] == "done" and first["numbers"] == [["blobs", "12"]] and first["table"]["rows"] == [["1"]]
    assert state.snapshot()["stages"][1]["status"] == "waiting"          # the whole route is visible from the start


def test_a_failing_stage_is_marked_and_the_run_is_in_error():
    state = LiveState()
    state.reset(PLAN, {"label": "x"}, step_mode=False)
    with pytest.raises(ValueError):
        with state.stage("a"):
            raise ValueError("boom")
    snapshot = state.snapshot()
    assert snapshot["stages"][0]["status"] == "error" and snapshot["status"] == "error" and "boom" in snapshot["message"]


def test_a_skipped_stage_says_why():
    state = LiveState()
    state.reset(PLAN, {"label": "x"}, step_mode=False)
    with state.stage("a") as stage:
        stage.skip("nothing to do")
    assert state.snapshot()["stages"][0]["status"] == "skipped"


def test_step_mode_waits_for_next_and_stop_ends_the_wait():
    state = LiveState()
    state.reset(PLAN, {"label": "x"}, step_mode=True)
    reached = []

    def worker():
        with state.stage("a"):
            pass
        reached.append("after a")
        try:
            with state.stage("b"):
                pass
        except RunCancelled:
            reached.append("cancelled")

    thread = threading.Thread(target=worker)
    thread.start()
    time.sleep(0.5)
    assert state.snapshot()["status"] == "paused" and reached == []     # held after stage a
    state.release()
    time.sleep(0.6)
    assert reached == ["after a"] and state.snapshot()["status"] == "paused"   # now held after stage b
    state.cancel()
    thread.join(timeout=3)
    assert reached == ["after a", "cancelled"]


def test_switching_to_automatic_lets_a_waiting_run_go_on():
    state = LiveState()
    state.reset(PLAN, {"label": "x"}, step_mode=True)
    done = threading.Event()

    def worker():
        with state.stage("a"):
            pass
        with state.stage("b"):
            pass
        done.set()

    threading.Thread(target=worker).start()
    time.sleep(0.4)
    state.set_step_mode(False)
    assert done.wait(3)


def test_old_pictures_are_dropped_beyond_the_limit(monkeypatch):
    monkeypatch.setattr("asthunt.viewer.live_state.MAX_IMAGES", 3)
    state = LiveState()
    urls = [state.store_image(bytes([i])) for i in range(5)]
    assert state.get_image(urls[0].split("/")[-1]) is None and state.get_image(urls[-1].split("/")[-1]) == bytes([4])
