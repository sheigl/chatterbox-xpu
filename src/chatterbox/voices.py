"""Voice library: scan a folder of audio files and expose them as selectable voices.

Default folder is ``<repo_root>/voices`` (auto-created). Override with the
``CHATTERBOX_VOICES_DIR`` environment variable.

Drop any audio file (wav/mp3/flac/ogg/m4a) into the folder, call :func:`refresh_voices`
from the UI, and it becomes selectable in the app's voice dropdown.
"""

import os
from pathlib import Path

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def voices_dir() -> Path:
    """Return the voice library directory, creating it if needed."""
    override = os.environ.get("CHATTERBOX_VOICES_DIR")
    if override:
        path = Path(override)
    else:
        # <repo_root>/voices
        path = Path(__file__).resolve().parents[2] / "voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_voices() -> list[str]:
    """Return sorted names of audio files in the voice library."""
    return sorted(
        p.name
        for p in voices_dir().iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    )


def voice_path(name: str) -> str | None:
    """Return the full path for a voice name, or None if it doesn't exist."""
    if not name:
        return None
    p = voices_dir() / Path(name).name  # strip any path components
    return str(p) if p.is_file() else None


def resolve_voice(name: str) -> str | None:
    """Resolve an OpenAI-style voice name to a file in the voice library.

    Matches case-insensitively. Accepts a bare stem ("am_welch") or a filename
    ("am_welch.wav"). If multiple files share a stem, the lossless one (.wav /
    .flac) is preferred for the highest-quality voice reference.
    """
    if not name:
        return None
    d = voices_dir()
    name = str(name).strip()
    if not name:
        return None
    lower = name.lower()
    files = [
        p
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    ]

    # 1) bare stem match (case-insensitive), preferring lossless formats
    matches = sorted(
        (p for p in files if p.stem.lower() == lower),
        key=lambda p: (p.suffix.lower() not in {".wav", ".flac"}, p.name),
    )
    if matches:
        return str(matches[0])

    # 2) exact filename match (case-insensitive)
    for p in files:
        if p.name.lower() == lower:
            return str(p)

    return None


def refresh_voices() -> dict:
    """Gradio helper: returns an update with the latest voice choices."""
    import gradio as gr

    return gr.update(choices=list_voices())