"""Single-threaded executor for all XPU model work.

The Intel level-zero XPU runtime fails with ``UR_RESULT_ERROR_OUT_OF_RESOURCES``
when a model is loaded in one OS thread and used in another (e.g. Gradio and
FastAPI dispatch each request to a different worker thread). To avoid this, all
model loading and inference must happen on one dedicated thread.

Usage::

    from chatterbox.worker import run

    model = run(lambda: ChatterboxTTS.from_pretrained("xpu:1"))
    wav = run(lambda: model.generate(text, audio_prompt_path=ref))
"""

import itertools
import queue
import threading

import torch


class _Worker:
    def __init__(self, device: str):
        self.device = str(device)
        self._queue: queue.Queue = queue.Queue()
        self._results: dict = {}
        self._thread = threading.Thread(target=self._run, daemon=True, name="xpu-worker")
        self._thread.start()

    def _run(self):
        base = self.device.split(":")[0]
        if base == "xpu":
            idx = int(self.device.split(":")[1]) if ":" in self.device else 0
            torch.xpu.set_device(idx)
        elif base == "cuda":
            torch.cuda.set_device(0)
        while True:
            job_id, fn, event = self._queue.get()
            try:
                self._results[job_id] = ("ok", fn())
            except BaseException as e:  # noqa: BLE001 - propagate to the caller
                self._results[job_id] = ("err", e)
            finally:
                event.set()

    def submit(self, fn):
        job_id = next(_JOB_IDS)
        event = threading.Event()
        self._queue.put((job_id, fn, event))
        event.wait()
        status, result = self._results.pop(job_id)
        if status == "err":
            raise result
        return result


_JOB_IDS = itertools.count(1)
_worker: _Worker | None = None
_worker_lock = threading.Lock()


def _get_worker(device: str) -> _Worker:
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = _Worker(device)
        return _worker


def run(fn) -> object:
    """Run ``fn`` on the process-wide single worker thread and return its result.

    The worker thread is created lazily on first use (using the best available
    device), so the first call should be the model load to pin the device.
    """
    from chatterbox.devices import get_best_device

    return _get_worker(get_best_device()).submit(fn)