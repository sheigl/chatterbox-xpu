import gradio as gr
from chatterbox.vc import ChatterboxVC
from chatterbox.devices import get_best_device
from chatterbox.voices import list_voices, resolve_voice, voice_path, voices_dir, refresh_voices
from chatterbox.worker import run


DEVICE = get_best_device()
VOICES_DIR = voices_dir()


model = run(lambda: ChatterboxVC.from_pretrained(DEVICE))
def generate(audio, target_voice_name, target_voice_upload):
    if target_voice_name:
        target_voice_path = resolve_voice(target_voice_name)
    else:
        target_voice_path = target_voice_upload
    if not target_voice_path:
        target_voice_path = None
    wav = run(lambda: model.generate(
        audio, target_voice_path=target_voice_path,
    ))
    return model.sr, wav.squeeze(0).numpy()


with gr.Blocks(title="Chatterbox Voice Conversion") as demo:
    gr.Markdown("# 🎙️ Chatterbox Voice Conversion")

    with gr.Row():
        with gr.Column():
            source_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Input audio file (the speech to convert)",
            )
        with gr.Column():
            with gr.Row():
                target_voice_lib = gr.Dropdown(
                    choices=list_voices(),
                    label="Target voice library",
                    info=f"Voices from {VOICES_DIR} — drop audio files there, then click Refresh.",
                    value=None,
                )
                refresh_btn = gr.Button("↻ Refresh voice library")
            refresh_btn.click(fn=refresh_voices, inputs=[], outputs=target_voice_lib)
            target_voice_upload = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="…or upload a target voice audio file (if neither, the default voice is used)",
                value=None,
            )

    convert_btn = gr.Button("Convert", variant="primary")
    audio_output = gr.Audio(label="Output Audio")

    convert_btn.click(
        fn=generate,
        inputs=[source_audio, target_voice_lib, target_voice_upload],
        outputs=audio_output,
    )

if __name__ == "__main__":
    demo.queue(
        max_size=50,
        default_concurrency_limit=1,
    ).launch()