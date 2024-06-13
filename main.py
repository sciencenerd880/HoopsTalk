from src.tts import tts
from src.text_personification import personify


if __name__ == "__main__":
    # PLACEHOLDER CAPTION
    # TODO: later change this one to video input and caption output
    caption = "After playing 36 total minutes against the Kings, the 21-year veteran exited the floor with around four minutes left on the clock."
    character = "Donald Trump"

    personified_text, token_usage = personify(caption, character)
    print(token_usage)

    tts(personified_text, character)
