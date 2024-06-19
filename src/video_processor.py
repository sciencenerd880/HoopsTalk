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

def create_message(encoded_frames, pred_actions, frame_interval=50, previous_commentary=""):
    ActionType1_Pred_Result, ActionType1_Pred_Prob, ActionType2_Pred_Result, ActionType2_Pred_Prob = pred_actions
    event_string = ""
    
    if ActionType1_Pred_Result != 'NA':
        event_string += f"ActionType1: {ActionType1_Pred_Result} with a probability of {ActionType1_Pred_Prob:.2f}. "
    if ActionType2_Pred_Result != 'NA':
        event_string += f"ActionType2: {ActionType2_Pred_Result} with a probability of {ActionType2_Pred_Prob:.2f}. "
        
    content = [
        """
        You are an enthusiastic and knowledgeable NBA commentator for the 2018-2019 season. Here are frames from a recent NBA game. 
        Your task is to provide a brief and engaging commentary summarizing the key events depicted in these frames. Maintain a flow and continuity 
        with the previous commentary provided. Ensure that the events described are logically sequenced and contextually appropriate for different parts of the game:
        """,
        *map(lambda x: {"image": x, "resize": 768}, encoded_frames[0::frame_interval]),
        f"Previous commentary:\n{previous_commentary}",
        f"Event context:\n{event_string}",
        """
        Consider the following types of events to help structure your commentary. Replace all words in brackets with the appropriate entity:
        
        1. Missed Shot: Replace [Player] with the player's name (identified by jersey number or team) attempting a [shot type].
        2. Rebound: Replace [Player] with the player's name (identified by jersey number or team) grabbing the rebound.
        3. Made Shot: Replace [Player] with the player's name (identified by jersey number or team) scoring a [shot type].
        4. Foul: Replace [Player] with the player's name (identified by jersey number or team) committing a [foul type].
        5. Turnover: Replace [Player] with the player's name (identified by jersey number or team) losing the ball.

        Ensure your commentary identifies players either by their name, their jersey number, or their team, and integrates these elements to provide a cohesive and engaging summary of the key events depicted in these frames in 15 words.
        """,
        "Ensure your commentary captures the excitement and energy of the game, providing vivid and dynamic descriptions of the actions and players involved. Your goal is to make the viewers feel as if they are witnessing the events unfold in real-time."
    ]
    
    messages = [{"role": "user", "content": content}]
    
    return messages

'''
1. Play Start: Describe the beginning of an offensive or defensive play.
        2. Shot Attempt: Replace [Player] with the player's name (identified by jersey number or team) attempting a [shot type] from [distance].
        3. Rebound: Replace [Player] with the player's name grabbing the rebound.
        4. Successful Shot: Replace [Player] with the player's name scoring a [shot type] from [distance].
        5. Assist: Replace [Player] with the player's name who assists.
        6. Foul: Replace [Player] with the player's name committing a [foul type].
        7. Turnover: Replace [Player] with the player's name losing the ball.
        8. Significant Moment: Describe any significant moment like a game-changing play or a remarkable defensive action, replacing [Player] with the relevant player's name.
        9. End of Period: Indicate the end of [period number].
'''