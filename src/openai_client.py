import openai
import logging

def get_commentary_for_frames(api_key, messages, max_tokens=300, model="gpt-4o"):
    openai.api_key = api_key
    
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens
        )
        # logging.info(f"GPT Response: {response.choices[0].message.content}")
        return response.choices[0].message.content
    except openai.error.OpenAIError as e:
        logging.error(f"OpenAI API error: {e}")
        return None