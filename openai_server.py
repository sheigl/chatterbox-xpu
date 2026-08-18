"""OpenAI-compatible audio API for Chatterbox with a drop-in voice library.

Drop voice samples into ``./voices`` (e.g. ``am_welch.wav`` or ``am_welch.mp3``),
then pass ``"am_welch"`` as the OpenAI ``voice`` parameter. The server resolves
the name to the file and uses it as the cloned-voice reference.

Start::

    .venv/bin/uvicorn openai_server:app --host 0.0.0.0 --port 8040

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

All model work runs on a single dedicated worker thread: the Intel level-zero
XPU runtime exhausts resources when the same model is used across threads, so
every load and generation is serialized through one thread.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio as ta
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from chatterbox import ChatterboxTTS
from chatterbox.devices import get_best_device
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.vc import ChatterboxVC
from chatterbox.voices import list_voices, resolve_voice, voices_dir
from chatterbox.worker import run

DEVICE = get_best_device()

# Model name -> (kind, loader, description)
MODELS = {
    "chatterbox": ("tts", lambda: ChatterboxTTS.from_pretrained(DEVICE), "Full English TTS (highest quality)"),
    "chatterbox-tts": ("tts", lambda: ChatterboxTTS.from_pretrained(DEVICE), "Full English TTS (highest quality)"),
    "chatterbox-turbo": ("turbo", lambda: ChatterboxTurboTTS.from_pretrained(DEVICE), "Turbo English TTS"),
    "chatterbox-nano": ("turbo", lambda: ChatterboxTurboTTS.from_pretrained(DEVICE, nano=True), "Nano English TTS"),
    "chatterbox-multilingual": ("mtl", lambda: ChatterboxMultilingualTTS.from_pretrained(DEVICE), "Multilingual TTS (23 languages)"),
    "chatterbox-vc": ("vc", lambda: ChatterboxVC.from_pretrained(DEVICE), "Voice conversion"),
}

CONTENT_TYPES = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}
VALID_FORMATS = set(CONTENT_TYPES)


# ---------------------------------------------------------------------------
# Model cache (accessed only from the single worker thread via run()).
# ---------------------------------------------------------------------------
_models: dict[str, object] = {}


def _get_model(name: str):
    """Load (on the worker thread) and return (kind, model)."""
    name = name or "chatterbox"
    if name not in MODELS:
        raise HTTPException(404, f"Unknown model '{name}'. Available: {sorted(MODELS)}")

    kind, loader, _ = MODELS[name]

    def load():
        if name not in _models:
            _models[name] = loader()
        return kind, _models[name]

    return run(load)


def _run_generation(model_name, fn):
    def run_job():
        return fn(_models[model_name])

    return run(run_job)


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
    y = wav.detach().cpu().numpy()
    if y.ndim == 1:
        y = librosa.effects.time_stretch(y, rate=speed)  # pitch-preserving
    else:
        y = np.stack([librosa.effects.time_stretch(ch, rate=speed) for ch in y])
    return torch.from_numpy(np.asarray(y, dtype=np.float32))


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
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


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
    if len(req.input) > 300:
        req.input = req.input[:300]
    if req.response_format not in VALID_FORMATS:
        raise HTTPException(422, f"response_format must be one of {sorted(VALID_FORMATS)}")

    ref_path = _resolve_voice_or_404(req.voice)
    kind, _ = _get_model(req.model)

    def generate(model):
        if kind == "mtl":
            language = req.language_id or "en"
            return model.generate(req.input, language, audio_prompt_path=ref_path)
        return model.generate(req.input, audio_prompt_path=ref_path)

    wav = _run_generation(req.model, generate)
    sr = _get_model(req.model)[1].sr

    wav = _apply_speed(wav, sr, req.speed)
    data = _tensor_to_bytes(wav, sr, req.response_format)

    return Response(
        content=data,
        media_type=CONTENT_TYPES[req.response_format],
        headers={"X-Chatterbox-Model": req.model, "X-Chatterbox-Voice": req.voice or "default"},
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
        kind, _ = _get_model("chatterbox-vc")

        def convert(model):
            return model.generate(tmp_path, target_voice_path=ref_path)

        wav = _run_generation("chatterbox-vc", convert)
    finally:
        os.unlink(tmp_path)

    sr = _get_model("chatterbox-vc")[1].sr
    data = _tensor_to_bytes(wav, sr, response_format)
    return Response(
        content=data,
        media_type=CONTENT_TYPES[response_format],
        headers={"X-Chatterbox-Model": "chatterbox-vc", "X-Chatterbox-Voice": target_voice},
    )