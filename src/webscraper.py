import glob
import pandas as pd
import urllib.request

team = "dal"
list_of_games = glob.glob("./data/raw/NSVA_Data/NSVA_Data/*.csv")
#print(list_of_games)
dal_games =[game for game in list_of_games if team in game]
#print(dal_games)
print('Team {} has {} games'.format(team,
                                     str(len(dal_games))))

temp_dal_games = dal_games[:2]
print('Extracting {} games\n'.format(len(temp_dal_games)))
print(temp_dal_games)

for game in temp_dal_games:
    
    game_detail = pd.read_csv(game)
    game_name = game.split("\\")[-1] #modified from leo to put \\ instead of /
    game_name = game_name.split(".")[0]
    print(game_name)
    
    for detail_row in game_detail.iterrows():
        
        game_sample = detail_row[1]
        # download the video
        videos = game_sample["video_url"].split("</>")
        video_ids = str(game_sample["video_id"]).split("</>")
        print(videos,video_ids)
        
        for video_url, video_id in zip(videos, video_ids):
            file_name = f"./data/raw/NSVA_Data/NSVA_Video/{game_name}_{video_id.strip()}.mp4"
            print(f"Downloading {file_name}")
            urllib.request.urlretrieve(video_url.strip(), file_name)

            



'''
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
            
'''
