import helper
from pathlib import Path
import os

def main():
    frame_interval = 30  # Save every 30th frame
    video_path = './data/videos/game_0021801212-phi-vs-mia.csv_video_33589.mp4'
    output_dir = './data/extracted_frames'
    audio_dir = './data/audio'
    api_key = os.environ['OPENAI_API_KEY']

    base_path = Path.cwd()
    abs_video_path = (base_path / video_path).resolve()
    abs_audio_path = (base_path / audio_dir).resolve()
    
    # print('abs_video_path',abs_video_path)
    # print('abs_audio_path',abs_audio_path)
    
    encoded_frames = helper.extract_encode_frames(abs_video_path, output_dir)
    messages = helper.create_message(encoded_frames)
    commentary = helper.get_commentary_for_frames(api_key, messages, max_tokens=300)
    print('commentary:',commentary)
    helper.text_to_speech(commentary, abs_audio_path)

if __name__ == '__main__':
    main()