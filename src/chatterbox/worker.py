"""Per-device worker threads for XPU model work.

The Intel level-zero XPU runtime fails with ``UR_RESULT_ERROR_OUT_OF_RESOURCES``
when a model is loaded in one OS thread and used in another (e.g. Gradio and
FastAPI dispatch each request to a different worker thread). To avoid this, all
model loading and inference for a given device must happen on one dedicated
thread.

Each device gets its own worker thread (and therefore its own model replicas),
so work on different devices runs truly in parallel.

Usage::

    from chatterbox.worker import run, run_on

    model = run(lambda: ChatterboxTTS.from_pretrained(device))
    wav = run(lambda: model.generate(text, audio_prompt_path=ref))

    # Or target a specific device explicitly:
    wav = run_on("xpu:1", lambda: model.generate(text))
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
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"xpu-worker-{self.device}")
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

    def depth(self) -> int:
        """Number of jobs queued or in flight (running job included)."""
        return self._queue.qsize()


_JOB_IDS = itertools.count(1)
_workers: dict[str, _Worker] = {}
_workers_lock = threading.Lock()


def get_worker(device: str) -> _Worker:
    """Return the dedicated worker thread for ``device`` (creating it on first use)."""
    device = str(device)
    with _workers_lock:
        worker = _workers.get(device)
        if worker is None:
            worker = _Worker(device)
            _workers[device] = worker
        return worker


def worker_devices() -> list[str]:
    """Devices that currently have a live worker thread."""
    with _workers_lock:
        return sorted(_workers)


def depth(device: str) -> int:
    """Queue depth (queued + running jobs) for ``device``; 0 if its worker has not started."""
    with _workers_lock:
        worker = _workers.get(str(device))
    return worker.depth() if worker is not None else 0


def run_on(device: str, fn) -> object:
    """Run ``fn`` on the dedicated worker thread for ``device`` and return its result."""
    return get_worker(device).submit(fn)


def run(fn) -> object:
    """Run ``fn`` on the worker thread for the best available device and return its result.

    The worker thread is created lazily on first use, so the first call should
    be the model load to pin the device.
    """
    from chatterbox.devices import get_best_device

    return run_on(get_best_device(), fn)
