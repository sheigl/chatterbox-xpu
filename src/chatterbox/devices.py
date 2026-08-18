import torch


def is_device_available(device: str) -> bool:
    """Return True if the given device string ("cuda", "xpu", "mps", "cpu") is usable.

    Supports device index suffixes, e.g. "xpu:1".
    """
    device = str(device).lower()
    base = device.split(":")[0]
    if base == "cpu":
        return True
    if base == "cuda":
        return torch.cuda.is_available()
    if base == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            return False
        if ":" in device:
            try:
                idx = int(device.split(":")[1])
                return 0 <= idx < xpu.device_count()
            except ValueError:
                return False
        return True
    if base == "mps":
        return torch.backends.mps.is_available()
    return False


def _xpu_with_most_memory() -> str:
    """Return the XPU device string with the largest total memory."""
    xpu = getattr(torch, "xpu", None)
    best, best_mem = 0, -1
    for i in range(xpu.device_count()):
        props = xpu.get_device_properties(i)
        mem = getattr(props, "total_memory", 0) or 0
        if mem > best_mem:
            best, best_mem = i, mem
    return f"xpu:{best}"


def get_best_device() -> str:
    """Pick the best available accelerator: cuda > xpu > mps > cpu.

    For XPU, the device with the most total memory is selected (e.g. "xpu:1")
    so that larger models like ChatterboxTurbo/Nano can be loaded.
    """
    if is_device_available("cuda"):
        return "cuda"
    if is_device_available("xpu"):
        return _xpu_with_most_memory()
    if is_device_available("mps"):
        return "mps"
    return "cpu"
