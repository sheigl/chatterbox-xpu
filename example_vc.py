import torchaudio as ta

from chatterbox.vc import ChatterboxVC
from chatterbox.devices import get_best_device

# Automatically detect the best available device (cuda > xpu > mps > cpu)
device = get_best_device()

print(f"Using device: {device}")

AUDIO_PATH = "YOUR_FILE.wav"
TARGET_VOICE_PATH = "YOUR_FILE.wav"

model = ChatterboxVC.from_pretrained(device)
wav = model.generate(
    audio=AUDIO_PATH,
    target_voice_path=TARGET_VOICE_PATH,
)
ta.save("testvc.wav", wav, model.sr)
