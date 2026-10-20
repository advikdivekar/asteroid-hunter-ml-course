"""What the live viewer's run is doing right now: stages, their pictures and numbers, shared between the worker and the web page."""

import threading
import time
from contextlib import contextmanager

MAX_IMAGES = 400                 # old pictures are dropped beyond this, so a long session cannot fill the memory


class RunCancelled(Exception):
    """Raised inside the worker when the person pressed Stop."""


class StageHandle:
    """What a stage fills in while it runs: numbers, pictures, a table and notes."""

    def __init__(self, state, stage):
        self._state, self._stage = state, stage

    def number(self, label, value):
        with self._state.lock:
            self._stage["numbers"].append([label, str(value)])

    def note(self, text):
        with self._state.lock:
            self._stage["notes"].append(text)

    def image(self, label, png_bytes):
        url = self._state.store_image(png_bytes)
        with self._state.lock:
            self._stage["images"].append({"label": label, "url": url})

    def table(self, columns, rows):
        with self._state.lock:
            self._stage["table"] = {"columns": [str(c) for c in columns], "rows": [[str(v) for v in row] for row in rows]}

    def skip(self, why):
        with self._state.lock:
            self._stage["skipped"] = why


class LiveState:
    def __init__(self):
        self.lock = threading.RLock()
        self._gate = threading.Event()
        self._images, self._image_counter = {}, 0
        self.reset([], None, False)

    def reset(self, stage_plan, unit, step_mode):
        """Start a new run. stage_plan is a list of (id, title, narration): the whole route is shown from the start."""
        with self.lock:
            self.status = "idle" if unit is None else "running"
            self.message = ""
            self.unit = unit
            self.step_mode = step_mode
            self.cancelled = False
            self.started = time.time()
            self.finished_at = None
            self.stages = [{"id": i, "title": t, "narration": n, "status": "waiting", "seconds": None, "numbers": [], "notes": [],
                            "images": [], "table": None, "skipped": None} for i, t, n in stage_plan]
            self._images.clear()
            self._gate.clear()

    def store_image(self, png_bytes):
        with self.lock:
            self._image_counter += 1
            name = f"{self._image_counter}.png"
            self._images[name] = png_bytes
            for old in sorted(self._images, key=lambda n: int(n.split(".")[0]))[:-MAX_IMAGES]:
                del self._images[old]
            return f"/img/{name}"

    def get_image(self, name):
        with self.lock:
            return self._images.get(name)

    @contextmanager
    def stage(self, stage_id):
        """Mark a stage running, hand back a StageHandle, mark it done (or failed); in step mode wait for the person afterwards."""
        with self.lock:
            stage = next(s for s in self.stages if s["id"] == stage_id)
            stage["status"] = "running"
            began = time.time()
        try:
            yield StageHandle(self, stage)
        except RunCancelled:
            raise
        except Exception as error:
            with self.lock:
                stage["status"] = "error"
                stage["notes"].append(f"{type(error).__name__}: {error}")
                self.status, self.message = "error", f"stage '{stage['title']}' failed: {error}"
            raise
        with self.lock:
            stage["seconds"] = round(time.time() - began, 1)
            stage["status"] = "skipped" if stage["skipped"] else "done"
        self._wait_for_person()

    def _wait_for_person(self):
        if not self.step_mode:
            return
        with self.lock:
            self.status = "paused"
        self._gate.clear()
        while not self._gate.wait(0.2):
            if self.cancelled:
                raise RunCancelled()
        with self.lock:
            if self.cancelled:
                raise RunCancelled()
            self.status = "running"

    def release(self):
        """The person pressed Next (or switched to automatic): let the worker go on."""
        self._gate.set()

    def set_step_mode(self, step_mode):
        with self.lock:
            self.step_mode = step_mode
        if not step_mode:
            self._gate.set()

    def cancel(self):
        with self.lock:
            self.cancelled = True
        self._gate.set()

    def finish(self, message=""):
        with self.lock:
            if self.status not in ("error", "cancelled"):
                self.status = "finished"
            self.message = message or self.message
            self.finished_at = time.time()

    def mark_cancelled(self):
        with self.lock:
            self.status, self.message, self.finished_at = "cancelled", "stopped by you", time.time()

    def snapshot(self):
        with self.lock:
            now = self.finished_at or time.time()
            return {"status": self.status, "message": self.message, "unit": self.unit, "step_mode": self.step_mode,
                    "elapsed": round(now - self.started, 1) if self.unit else 0,
                    "stages": [dict(s, numbers=list(s["numbers"]), notes=list(s["notes"]), images=list(s["images"])) for s in self.stages]}
