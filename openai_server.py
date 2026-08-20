"""OpenAI-compatible audio API for Chatterbox with a drop-in voice library.

Drop voice samples into ``./voices`` (e.g. ``am_welch.wav`` or ``am_welch.mp3``),
then pass ``"am_welch"`` as the OpenAI ``voice`` parameter. The server resolves
the name to the file and uses it as the cloned-voice reference.

Start (device pool via CLI flags)::

    .venv/bin/python openai_server.py --host 0.0.0.0 --port 8040 --devices xpu:0,xpu:2

Start (uvicorn CLI; device pool from env vars)::

    CHATTERBOX_DEVICES=xpu:0,xpu:2 .venv/bin/uvicorn openai_server:app --host 0.0.0.0 --port 8040

Examples::

    curl -X POST http://localhost:8040/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"model":"chatterbox","input":"Hello from the cloned voice.","voice":"am_welch","response_format":"wav"}' \\
      -o out.wav

Quality notes:
- Defaults to the full (non-turbo) English model and lossless 24 kHz WAV output.
- MP3 output is encoded at 320 kbps via ffmpeg (libmp3lame) for the highest
  quality lossy option.
- Voice references prefer lossless files (.wav/.flac) when multiple files share
  a stem.

Parallelism:
The Intel level-zero XPU runtime exhausts resources when the *same model
instance* is used across threads (UR_RESULT_ERROR_OUT_OF_RESOURCES). Each
device therefore gets its own dedicated worker thread and its own model
replicas, and requests are scheduled to the least-busy device — so N devices
give N-way parallel generation.

Device pool (pin which XPUs the server may use):
- CHATTERBOX_DEVICES="xpu:0,xpu:1,xpu:2"  -> exactly this pool (pinned).
- Otherwise CHATTERBOX_DEVICE="xpu:2"     -> single device (legacy behavior).
- Neither set                             -> the single best available device.

Per-model device pinning (control where heavy models load):
- CHATTERBOX_MODEL_DEVICES="chatterbox-multilingual=xpu:1,chatterbox-turbo=xpu:1,xpu:2"
  Pinned models only load on the listed devices; unpinned models spread over
  the whole pool. If a pinned model has no device in the pool, requests for it
  fail with HTTP 503.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torchaudio as ta
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from chatterbox import ChatterboxTTS
from chatterbox.devices import get_best_device, is_device_available
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.vc import ChatterboxVC
from chatterbox.voices import list_voices, resolve_voice, voices_dir
import chatterbox.worker as worker
from chatterbox.worker import run_on


# ---------------------------------------------------------------------------
# CLI flags (applied before the device pool is computed below).
#
# ``--devices`` / ``--model-devices`` set the corresponding env vars so the
# same logic paths are used under any launcher. Precedence: CLI > CHATTERBOX_DEVICES
# > CHATTERBOX_DEVICE > best available device. Under ``uvicorn openai_server:app``
# these flags are absent and the env vars rule.
# ---------------------------------------------------------------------------
def _apply_cli_device_args() -> None:
    import argparse

    if {"-h", "--help"} & set(sys.argv[1:]):
        return  # let main() show full help
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--model-devices", default=None)
    known, _ = parser.parse_known_args()
    if known.devices:
        os.environ["CHATTERBOX_DEVICES"] = known.devices
    if known.model_devices:
        os.environ["CHATTERBOX_MODEL_DEVICES"] = known.model_devices


_apply_cli_device_args()


# ---------------------------------------------------------------------------
# Device pool and model pinning
# ---------------------------------------------------------------------------
def _device_pool() -> list[str]:
    raw = os.getenv("CHATTERBOX_DEVICES") or os.getenv("CHATTERBOX_DEVICE")
    if not raw:
        return [get_best_device()]
    pool: list[str] = []
    for part in str(raw).split(","):
        d = part.strip()
        if d and d not in pool:
            pool.append(d)
    available = [d for d in pool if is_device_available(d)]
    skipped = [d for d in pool if d not in available]
    if skipped:
        print(f"[openai_server] skipping unavailable devices: {skipped}")
    if not available:
        best = get_best_device()
        print(f"[openai_server] no pool devices available, falling back to {best}")
        return [best]
    return available


def _model_device_pins() -> dict[str, set[str]]:
    """model name -> set of devices it is allowed to load on."""
    pins: dict[str, set[str]] = {}
    for part in str(os.getenv("CHATTERBOX_MODEL_DEVICES", "")).split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            print(f"[openai_server] ignoring malformed CHATTERBOX_MODEL_DEVICES entry: {part!r}")
            continue
        name, devs = part.split("=", 1)
        name = name.strip()
        devs = {d.strip() for d in devs.split(",") if d.strip()}
        if name in MODELS and devs:
            pins[name] = devs
    return pins


DEVICES = _device_pool()
print(f"[openai_server] device pool: {DEVICES}")


# Model name -> (kind, loader(device), description)
MODELS = {
    "chatterbox": ("tts", lambda dev: ChatterboxTTS.from_pretrained(dev), "Full English TTS (highest quality)"),
    "chatterbox-tts": ("tts", lambda dev: ChatterboxTTS.from_pretrained(dev), "Full English TTS (highest quality)"),
    "chatterbox-turbo": ("turbo", lambda dev: ChatterboxTurboTTS.from_pretrained(dev), "Turbo English TTS"),
    "chatterbox-nano": ("turbo", lambda dev: ChatterboxTurboTTS.from_pretrained(dev, nano=True), "Nano English TTS"),
    "chatterbox-multilingual": ("mtl", lambda dev: ChatterboxMultilingualTTS.from_pretrained(dev), "Multilingual TTS (23 languages)"),
    "chatterbox-vc": ("vc", lambda dev: ChatterboxVC.from_pretrained(dev), "Voice conversion"),
}
MODEL_DEVICE_PINS = _model_device_pins()

CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}
VALID_FORMATS = set(CONTENT_TYPES)

MAX_INPUT_CHARS = int(os.getenv("CHATTERBOX_MAX_INPUT_CHARS", "1000"))


# ---------------------------------------------------------------------------
# Model cache: (device, model name) -> model. Accessed only from worker threads.
# ---------------------------------------------------------------------------
_models: dict[tuple[str, str], object] = {}


def _pick_device(model_name: str) -> str:
    model_name = model_name or "chatterbox"
    pool = DEVICES
    if model_name in MODEL_DEVICE_PINS:
        allowed = MODEL_DEVICE_PINS[model_name]
        pool = [d for d in DEVICES if d in allowed]
        if not pool:
            raise HTTPException(
                503,
                f"model '{model_name}' is pinned to {sorted(allowed)} but none of those "
                f"devices are in the pool {DEVICES}",
            )
    return min(pool, key=lambda d: worker.depth(d))


def _get_model(name: str, device: str):
    """Load (on the device's worker thread) and return (kind, model)."""
    name = name or "chatterbox"
    if name not in MODELS:
        raise HTTPException(404, f"Unknown model '{name}'. Available: {sorted(MODELS)}")

    kind, loader, _ = MODELS[name]
    key = (device, name)

    def load():
        if key not in _models:
            model = loader(device)
            # Voice-encoder LSTM is flaky on level-zero after repeated runs on B580
            # (UR_RESULT_ERROR_OUT_OF_RESOURCES on the 2nd generation). It's tiny, so
            # keep it on CPU and leave all GPU memory for the main model.
            if str(device).split(":")[0] == "xpu" and hasattr(model, "ve"):
                model.ve.to("cpu")
            _models[key] = model
        return kind, _models[key]

    return run_on(device, load)


def _run_generation(device: str, model_name: str, fn):
    def job():
        try:
            return fn(_models[(device, model_name)])
        finally:
            _release_gpu_memory(device)

    return run_on(device, job)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_voice_or_404(voice: str | None) -> str | None:
    if not voice:
        return None
    path = resolve_voice(voice)
    if path is None:
        raise HTTPException(
            404,
            f"Voice '{voice}' not found in {voices_dir()}. "
            f"Available voices: {list_voices() or '(none — drop a .wav/.mp3 into the folder)'}",
        )
    return path


def _tensor_to_bytes(wav: torch.Tensor, sr: int, response_format: str) -> bytes:
    """Encode a waveform to the requested output format (quality-first)."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dtype != torch.float32:
        wav = wav.float()

    if response_format == "pcm":
        pcm = (torch.clamp(wav, -1.0, 1.0) * 32767).round().to(torch.int16)
        return pcm.cpu().numpy().tobytes()

    # MP3: upsample to 48 kHz first (24 kHz is MPEG-2, capped at 160 kbps), then
    # encode at 320 kbps CBR for the highest-quality lossy output.
    mp3_sr = sr
    if response_format == "mp3" and sr < 48000:
        wav = ta.functional.resample(wav, sr, 48000)
        mp3_sr = 48000

    with tempfile.NamedTemporaryFile(suffix=f".{response_format}", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        ta.save(tmp_path, wav, mp3_sr)
        if response_format == "mp3":
            mp3_path = tmp_path + ".320.mp3"
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is not None:
                proc = subprocess.run(
                    [
                        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-i", tmp_path, "-codec:a", "libmp3lame",
                        "-b:a", "320k", "-ar", str(mp3_sr), mp3_path,
                    ],
                    capture_output=True,
                )
                if proc.returncode == 0 and os.path.exists(mp3_path):
                    os.unlink(tmp_path)
                    tmp_path = mp3_path
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _apply_speed(wav: torch.Tensor, sr: int, speed: float) -> torch.Tensor:
    if speed <= 0 or speed > 4.0:
        raise HTTPException(422, "speed must be in (0, 4]")
    if abs(speed - 1.0) < 1e-6:
        return wav

    # High-quality time-stretch via ffmpeg's rubberband filter (WSOLA-based).
    # librosa.effects.time_stretch uses a phase vocoder, which smears speech
    # ("distant/echoey" artifacts) even at mild speeds like 0.9.
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise HTTPException(500, "ffmpeg is required for speed adjustment")

    y = wav.detach().cpu().float()
    if y.dim() == 1:
        y = y.unsqueeze(0)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        in_path = tmp.name
    out_path = in_path + ".spd.wav"
    try:
        ta.save(in_path, y, sr)
        proc = subprocess.run(
            [
                ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                "-i", in_path,
                "-af", f"rubberband=tempo={speed:.6f}",
                "-ar", str(sr),
                out_path,
            ],
            capture_output=True,
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            raise HTTPException(
                500,
                f"ffmpeg rubberband failed: {proc.stderr.decode(errors='replace')}",
            )
        out_wav, out_sr = ta.load(out_path)
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.unlink(p)

    return out_wav.squeeze(0) if out_wav.dim() > 1 else out_wav


def _release_gpu_memory(device: str):
    """Return cached freed blocks to the driver to avoid VRAM ratcheting.

    The PyTorch level-zero caching allocator retains freed device (and host)
    memory across requests, so VRAM creeps upward under sustained synthesis
    until the vocoder fails to allocate (``could not create a primitive`` ->
    HTTP 500). Calling ``empty_cache()`` after each request returns those
    blocks to the driver. Must be called from that device's worker thread,
    where it is the current device.
    """
    base = str(device).split(":")[0]
    if base == "xpu":
        torch.xpu.empty_cache()
    elif base == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Chatterbox OpenAI-compatible API")


class SpeechRequest(BaseModel):
    model: str = "chatterbox"
    input: str = "You need to add some text for me to talk."
    voice: str | None = None
    response_format: str = "wav"
    speed: float = 1.0
    language_id: str | None = None  # extra field for the multilingual model


@app.get("/")
def root():
    return {
        "service": "Chatterbox OpenAI-compatible API",
        "docs": "/docs",
        "models": "/v1/models",
        "voices": "/v1/voices",
        "speech": "POST /v1/audio/speech",
        "voice_conversion": "POST /v1/audio/voice-conversion",
        "devices": DEVICES,
    }


@app.get("/health")
def health():
    # Snapshot under a retry: workers may be inserting model replicas mid-request.
    try:
        loaded = list(_models)
    except RuntimeError:
        loaded = []
    return {
        "status": "ok",
        "devices": {
            d: {
                "busy_jobs": worker.depth(d),
                "loaded_models": [n for (dev, n) in loaded if dev == d],
            }
            for d in DEVICES
        },
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "resemble-ai", "description": desc}
            for name, (_, _, desc) in MODELS.items()
        ],
    }


@app.get("/v1/voices")
def list_voices_endpoint():
    return {"object": "list", "data": [{"id": v, "object": "voice"} for v in list_voices()]}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest):
    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(413, f"input too long: {len(req.input)} chars > {MAX_INPUT_CHARS} max")
    if req.response_format not in VALID_FORMATS:
        raise HTTPException(422, f"response_format must be one of {sorted(VALID_FORMATS)}")

    ref_path = _resolve_voice_or_404(req.voice)
    device = _pick_device(req.model)
    kind, model = _get_model(req.model, device)

    def generate(m):
        if kind == "mtl":
            language = req.language_id or "en"
            return m.generate(req.input, language, audio_prompt_path=ref_path)
        return m.generate(req.input, audio_prompt_path=ref_path)

    wav = _run_generation(device, req.model or "chatterbox", generate)
    sr = model.sr

    wav = _apply_speed(wav, sr, req.speed)
    data = _tensor_to_bytes(wav, sr, req.response_format)

    return Response(
        content=data,
        media_type=CONTENT_TYPES[req.response_format],
        headers={
            "X-Chatterbox-Model": req.model,
            "X-Chatterbox-Voice": req.voice or "default",
            "X-Chatterbox-Device": device,
        },
    )


@app.post("/v1/audio/voice-conversion")
async def voice_conversion(
    file: UploadFile = File(...),
    target_voice: str = Form(...),
    response_format: str = Form("wav"),
):
    if response_format not in VALID_FORMATS:
        raise HTTPException(422, f"response_format must be one of {sorted(VALID_FORMATS)}")
    ref_path = _resolve_voice_or_404(target_voice)

    suffix = Path(file.filename or "in.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        device = _pick_device("chatterbox-vc")
        kind, model = _get_model("chatterbox-vc", device)

        def convert(m):
            return m.generate(tmp_path, target_voice_path=ref_path)

        wav = _run_generation(device, "chatterbox-vc", convert)
        sr = model.sr
        data = _tensor_to_bytes(wav, sr, response_format)
        return Response(
            content=data,
            media_type=CONTENT_TYPES[response_format],
            headers={
                "X-Chatterbox-Model": "chatterbox-vc",
                "X-Chatterbox-Voice": target_voice,
                "X-Chatterbox-Device": device,
            },
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        description=(
            "Chatterbox OpenAI-compatible API server. Device pool precedence: "
            "CLI --devices > CHATTERBOX_DEVICES > CHATTERBOX_DEVICE > best device."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument(
        "--devices",
        default=None,
        help='comma-separated XPU pool, e.g. "xpu:0,xpu:2" (pins devices; overrides CHATTERBOX_DEVICES)',
    )
    parser.add_argument(
        "--model-devices",
        default=None,
        help='pin models to devices, e.g. "chatterbox-multilingual=xpu:1" (overrides CHATTERBOX_MODEL_DEVICES)',
    )
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
