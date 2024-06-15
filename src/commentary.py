import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from video_processor import extract_encode_frames, create_message
from openai_client import get_commentary_for_frames

def main():
    load_dotenv()

    frame_interval = 60
    video_dir = './data/raw/NSVA_Video/0021800013-dal-vs-phx'
    text_dir = './data/text/GPT4o/commentary_results.csv'
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        logging.error("API key not found. Make sure to set the OPENAI_API_KEY environment variable.")
        return

    # Convert the directory string to a Path object
    abs_video_dir = Path(video_dir).resolve()
    logging.info(f"Absolute video directory: {abs_video_dir}")

    results = []
    commentary_buffer = []  # Initialize a buffer to store the last 10 lines of commentary

    # Use glob to find all .mp4 files in the directory
    for video_file in abs_video_dir.glob('*.mp4'):
        logging.info(f"Processing video: {video_file}")
        base64_frames = extract_encode_frames(video_file, frame_interval)
        messages = create_message(base64_frames, frame_interval, "\n".join(commentary_buffer))
        commentary = get_commentary_for_frames(api_key, messages)
        
        if commentary:
            results.append({'video_file': video_file.name, 'commentary': commentary})
            commentary_buffer.extend(commentary.split("\n"))  # Update the buffer with new lines
            commentary_buffer = commentary_buffer[-10:]  # Keep only the last 10 lines
        else:
            logging.error(f"Failed to get commentary for {video_file.name}")

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(text_dir, index=False)
    logging.info(f"Commentary saved to {text_dir}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
