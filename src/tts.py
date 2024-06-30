from textwrap import dedent
from TTS.utils.synthesizer import Synthesizer
from pathlib import Path


base_model_path = Path("models")
config_path = base_model_path / "config.json"
output_path = Path("output")

name_to_voice = {
    "Donald Trump": "trump",
    "David Attenborough": "david"
}

def tts(text, character, output_filename):
    voice = name_to_voice[character]

    model_file = base_model_path / f"{voice}.pth"
    synthesizer = Synthesizer(
        tts_config_path=config_path,
        tts_checkpoint=model_file
    )

    wav = synthesizer.tts(text)
    synthesizer.save_wav(wav, output_filename)
    print(f"Saved audio file in {output_filename}")


if __name__ == "__main__":
    text = dedent("""
        The quick brown fox jumps over the lazy dog
    """)

    tts(text, "Donald Trump", "output/trump_quick_brown_fox.wav")
