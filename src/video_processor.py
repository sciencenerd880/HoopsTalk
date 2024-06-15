import cv2
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_encode_frames(video_path, frame_interval=50):
    video_path = str(video_path).replace('\\', '/')
    video = cv2.VideoCapture(video_path)
    
    base64_frames = []
    frame_count = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64_frames.append(base64.b64encode(buffer).decode("utf-8"))
        frame_count += 1

    video.release()
    
    return base64_frames

def create_message(encoded_frames, frame_interval=50, previous_commentary=""):
    content = [
        "You are an NBA commentator. Here are frames from a recent NBA game. Your task is to provide a brief and engaging commentary summarizing the key events depicted in these frames. Please maintain a flow and continuity with the previous commentary provided. Ensure that the events described are logically sequenced and contextually appropriate for different parts of the game:",
        *map(lambda x: {"image": x, "resize": 768}, encoded_frames[0::frame_interval]),
        f"Previous commentary:\n{previous_commentary}",
        "Consider the following types of events to help structure your commentary:",
        """
        1. Play Start: Describe the beginning of an offensive or defensive play.
        2. Shot Attempt: [Player] attempts a [shot type] from [distance].
        3. Rebound: [Player] grabs the rebound.
        4. Successful Shot: [Player] scores a [shot type] from [distance].
        5. Assist: Assist by [Player].
        6. Foul: [Player] commits a [foul type].
        7. Turnover: [Player] loses the ball.
        8. Significant Moment: Describe any significant moment like a game-changing play or a remarkable defensive action.
        9. End of Period: End of [period number].
        Combine these elements to provide a cohesive and engaging summary of the key events depicted in these frames in approximately two sentences.
        """
    ]
    
    messages = [{"role": "user", "content": content}]
    
    return messages
