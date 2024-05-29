import glob
import pandas as pd
import urllib.request

list_of_games = glob.glob("./data/raw/NSVA_Data/NSVA_Data/*.csv")

for game in list_of_games:
    game_detail = pd.read_csv(game)
    game_name = game.split("/")[-1]
    for detail_row in game_detail.iterrows():
        game_sample = detail_row[1]
        # download the video
        videos = game_sample["video_url"].split("</>")
        video_ids = str(game_sample["video_id"]).split("</>")
        for video_url, video_id in zip(videos, video_ids):
            file_name = f"./data/raw/NSVA_Data/NSVA_Video/game_{game_name}_video_{video_id.strip()}.mp4"
            print(f"Downloading {file_name}")
            urllib.request.urlretrieve(video_url.strip(), file_name)
