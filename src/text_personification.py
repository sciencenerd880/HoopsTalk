import re
from textwrap import dedent
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(base_url="http://localhost:1234/v1", api_key='not_needed', temperature=0.7)

def personify(text, character):
    prompt_template = """
    Rewrite the following NBA game commentary in the style of {character_name} in less than 25 words:

    Here are some examples of conversion in basketball context:
    {examples}

    Now rewrite this one:
    "{prompt}" -->
    """

    examples = {
        "Donald Trump": dedent("""
            1. "Kobe Bryant just came back from his 4 months of injury, better than ever" --> "Folks, let me tell you, Kobe Bryant, tremendous comeback, believe me. He's a real winner, best player on the court, and I know more about great comebacks than any expert."

            2. "Lebron's shooting from just outside of the three point line, sadly he missed" --> "LeBron, a great player, shooting from just outside the three-point line. Sad! He missed. But he'll come back stronger, believe me!"

            3. "Celtics hold off desperate Mavs, go up 3-0 in Finals" --> "Celtics, fantastic team, hold off the desperate Mavs. Now up 3-0 in the Finals. Total dominance, folks!"
            """),
        "David Attenborough": dedent("""
            1. "Kobe Bryant just came back from his 4 months of injury, better than ever" --> "After four months, Kobe Bryant returns, displaying remarkable resilience and skill, more formidable than ever. A true marvel of athleticism and determination."

            2. "Lebron's shooting from just outside of the three point line, sadly he missed" --> "Observe LeBron, poised just outside the three-point line. He takes the shot, but alas, it misses. A rare moment in his illustrious journey."

            3. "Celtics hold off desperate Mavs, go up 3-0 in Finals" --> "The Celtics, with remarkable tenacity, hold off the desperate Mavericks, advancing to a commanding 3-0 lead in the Finals. An extraordinary display of dominance."
            """)
    }

    if character not in examples:
        raise ValueError(f"Character name {character} not in ['Donald Trump', 'David Attenborough']")

    
    result = llm.invoke(prompt_template.format(character_name=character, examples=examples[character], prompt=text))

    text_result = re.findall(r"\"(.*)\"", result.content)[0]
    token_usage = result.response_metadata["token_usage"]

    return text_result, token_usage


if __name__ == "__main__":
    text_result, token_usage = personify("After playing 36 total minutes against the Kings, the 21-year veteran exited the floor with around four minutes left on the clock.", "Donald Trump")
    
    print(text_result)
    print("TOKEN USAGE", token_usage)
