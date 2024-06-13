import cv2
import os
from pathlib import Path
import base64
from gtts import gTTS
import requests

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_encode_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    video_path = str(video_path).replace('\\', '/')
    video = cv2.VideoCapture(video_path)
    
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    # print(len(base64Frames), "frames read.")
    
    # display_handle = display(None, display_id=True)
    # for img in base64Frames:
    #     display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
    #     time.sleep(0.025)
    
    return base64Frames

def create_message(encoded_frames):
    
    messages = [
        {
            "role": "user",
            "content": [
                "These are frames from an NBA video that I want to upload. You are an NBA Commentator. Provide a commentary for these images as whole instead of frame by frame and approximately one sentence per 10 frames is sufficient:",
                *map(lambda x: {"image": x, "resize": 768}, encoded_frames[0::30]),
            ],
        },
    ]
    
    return messages

def get_commentary_for_frames(api_key, messages, max_tokens=300):
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model":"gpt-4o",
        "messages":messages,
        "max_tokens":max_tokens
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    print(response)
    
    response_json = response.json()
    
    if response.status_code == 200:
        text_response = response_json['choices'][0]['message']['content']
        return text_response
    else:
        error_message = response_json.get('error', 'Unknown error')
        print(f"Request failed with status code {response.status_code}, Error Msg: {error_message}")
        return None
    
    # return response.choices[0]

def text_to_speech(text, output_audio_path):
    output_audio_path = str(output_audio_path).replace('\\', '/')
    tts = gTTS(text=text, lang='en')
    tts.save(output_audio_path)