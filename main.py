from src.tts import tts
from src.text_personification import personify


if __name__ == "__main__":
    # PLACEHOLDER CAPTION
    # TODO: later change this one to video input and caption output
    caption = "Jump Ball: Adebayo vs. O'Quinn for the tip-off. The Heat win possession and begin to set up their offense as they move the ball down the court, looking for an early scoring opportunity against the Sixers' defense."
    character = "David Attenborough"

    personified_text, token_usage = personify(caption, character)
    print(token_usage)

    tts(personified_text, character)
