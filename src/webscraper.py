import os
import glob
import numpy as np
import pandas as pd
import urllib.request
import hashlib

# skvideo uses `np.float` and `np.int`, which are already deprecated
np.float = np.float64
np.int = np.int_
import skvideo.io

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

            
def compute_video_hash(video_path):
    # Initialize a hash object (MD5 in this case)
    hash_obj = hashlib.md5()

    # Read the video frame by frame
    videogen = skvideo.io.vreader(video_path)

    for frame in videogen:
        # Update the hash object with the frame data
        hash_obj.update(frame.tobytes())

    # Return the hex digest of the hash
    return hash_obj.hexdigest()


def unique_exploded(video_csv_file_names, video_output_dir):
    # this function does two things:
    #   1. Download ONLY the unique videos
    #   2. Created new CSV files to replace the previous ones, with ONLY the unique videos
    for csv_file in video_csv_file_names:
        match_video_details = pd.read_csv(csv_file)
        new_match_video_exploded = []

        for idx, match_snippet in match_video_details.iterrows():
            # check whether there are 2 actions in one snippet
            game_name = csv_file.split("/")[-1].replace(".csv", "")
            if len(match_snippet["actionType"].split("</>")) == 1:
                new_match_video_exploded.append(match_snippet.to_dict())
                # download the video
                video_id = match_snippet["video_id"]
                file_name = f"{video_output_dir}/{game_name}_{video_id.strip()}.mp4"
                urllib.request.urlretrieve(match_snippet["video_url"], file_name)
                print("Downloaded", file_name)
            else:
                # split the data into num elements for each columns
                num_elements = len(match_snippet["actionType"].split("</>"))
                new_dicts = [{} for _ in range(num_elements)]
                # put split result into different rows in new dictionaries
                for column in match_snippet.index:
                    split_value = match_snippet[column].split("</>")
                    for idx, value in enumerate(split_value):
                        new_dicts[idx][column] = value.strip()

                # download the videos
                video_ids = match_snippet["video_id"].split("</>")
                video_urls = match_snippet["video_url"].split("</>")

                video_contents = []
                video_file_names = []
                for video_id, video_url in zip(video_ids, video_urls):
                    file_name = f"{video_output_dir}/{game_name}_{video_id.strip()}.mp4"
                    urllib.request.urlretrieve(video_url, file_name)
                    print("Downloaded", file_name)
                    video_contents.append(compute_video_hash(file_name))
                    # identical videos will have the same hash values
                    video_file_names.append(file_name)

                # remove row if the video content is the same
                _, video_indices = np.unique(video_contents, axis=0, return_index=True)
                # delete videos from the drive that are non unique
                for idx, file_name in enumerate(video_file_names):
                    if idx not in video_indices:
                        # non-unique video, delete
                        os.remove(file_name)
                        print("Deleted Duplicate Video", file_name)
        
                # delete rows from new_dicts where the video is unique
                new_dicts = [new_dicts[i] for i in video_indices]

                new_match_video_exploded.extend(new_dicts)

        # save new csv file as `-exploded` file
        new_filename = csv_file.replace(".csv", "") + "-exploded.csv"
        new_match_video_exploded.to_csv(new_filename)
        print("Saved exploded CSV", new_filename)
