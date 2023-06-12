import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_llm_response(prompt, engine='davinci'):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.9,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response