import pandas as pd
from src.tts import tts, name_to_voice
from src.text_personification import personify

from tqdm import tqdm


CAPTION_CSV = "data/text/GPT4o/0021800013-dal-vs-phx_commentary_results2.csv"

if __name__ == "__main__":
    characters = ["David Attenborough", "Donald Trump"]
    original_captions = pd.read_csv(CAPTION_CSV)

    for character in characters:
        captions = []
        audio_files = []
        print(f"Running {character}")
        for video_file, caption in tqdm(original_captions[["video_file", "commentary"]].values, total=original_captions.shape[0]):
            personified_text, token_usage = personify(caption, character)
            print(token_usage)

            # save to file
            video_id = video_file.split("_")[-1].replace(".mp4", "")
            output_filename = f"output/{name_to_voice[character]}/{video_id}.wav"
            tts(personified_text, character, output_filename)
        
            captions.append(personified_text)
            audio_files.append(output_filename)

        original_captions[f"{name_to_voice[character]}_caption"] = captions
        original_captions[f"{name_to_voice[character]}_audio_file"] = audio_files

original_captions.to_csv("data/text/GPT4o/personified.csv", index=False)
