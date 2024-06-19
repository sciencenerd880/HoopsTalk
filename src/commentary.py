import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from video_processor import extract_encode_frames, create_message
from openai_client import get_commentary_for_frames
from tqdm import tqdm
import re

def main():
    load_dotenv()

    frame_interval = 30
    metadata_file = './data/raw/NSVA_Data_Exploded/0021800013-dal-vs-phx-exploded.xlsx'
    match = re.search(r'/([^/]+)-exploded\.xlsx$', metadata_file)
    game = match.group(1)
    video_dir = './data/raw/NSVA_Video/' + game
    text_dir = f'./data/text/GPT4o/{game}_commentary_results.csv'
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        logging.error("API key not found. Make sure to set the OPENAI_API_KEY environment variable.")
        return

    # Convert the directory string to a Path object
    # abs_video_dir = Path(video_dir).resolve()
    # logging.info(f"Absolute video directory: {abs_video_dir}")

    results = []
    commentary_buffer = []  # Initialize a buffer to store the last 10 lines of commentary
    
    meta_df = pd.read_excel(metadata_file)
    meta_df = meta_df.loc[meta_df['ActionType1_Pred_Result'].isin(['Foul','Rebound','Made Shot','Missed Shot','Turnover'])]
    meta_df = meta_df.iloc[:10,:]

    for index, row in tqdm(meta_df.iterrows()):
        video_file = video_dir + '/' + game + '_' + str(row['video_id']) + '.mp4'
        logging.info(f"Processing video: {video_file}")
        base64_frames = extract_encode_frames(video_file, frame_interval)
        pred_actions = (row['ActionType1_Pred_Result'],row['ActionType1_Pred_Prob'],row['ActionType2_Pred_Result'],row['ActionType2_Pred_Prob'])
        messages = create_message(base64_frames, pred_actions, frame_interval, "\n".join(commentary_buffer))
        commentary = get_commentary_for_frames(api_key, messages)
        
        if commentary:
            results.append({'video_id': row['video_id'], 'gen_commentary': commentary})
            commentary_buffer.extend(commentary.split("\n"))  # Update the buffer with new lines
            commentary_buffer = commentary_buffer[-10:]  # Keep only the last 10 lines
        else:
            logging.error(f"Failed to get commentary for {video_file.name}")

    # Save results to CSV
    df = pd.DataFrame(results)
    merged_df = pd.merge(meta_df, df, on='video_id', how='left')
    merged_df.to_csv(text_dir, index=False)
    logging.info(f"Commentary saved to {text_dir}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # INFO, CRITICAL
    main()
