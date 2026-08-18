import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.devices import get_best_device

# Load the Turbo model (auto-detects cuda > xpu > mps > cpu)
model = ChatterboxTurboTTS.from_pretrained(device=get_best_device())

# Generate with Paralinguistic Tags
text = "Oh, that's hilarious! [chuckle] Um anyway, we do have a new model in store. It's the SkyNet T-800 series and it's got basically everything. Including AI integration with ChatGPT and all that jazz. Would you like me to get some prices for you?"

# Generate audio (requires a reference clip for voice cloning)
# wav = model.generate(text, audio_prompt_path="your_10s_ref_clip.wav")
wav = model.generate(text)
ta.save("test-turbo.wav", wav, model.sr)
