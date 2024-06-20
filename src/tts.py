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
        Uh, summa-lumma, dooma-lumma, you assumin' I'm a human
        What I gotta do to get it through to you I'm superhuman?
        Innovative and I'm made of rubber so that anything
        You say is ricochetin' off of me, and it'll glue to you and
        I'm devastating, more than ever demonstrating
        How to give a motherfuckin' audience a feeling like it's levitating
        Never fading, and I know the haters are forever waiting
        For the day that they can say I fell off, they'll be celebrating
    """)

    tts(text, "Donald Trump")
