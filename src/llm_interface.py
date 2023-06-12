import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_llm_response(prompt, engine='davinci', temperature=0.9):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=temperature,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["\n"]
    )
    return response

def generate_prompt_generic(task_list):
    # few-shot learning description prompt
    # just for testing -- TODO refine, get better examples
        # QUESTION: how to maintain context across multiple prompts, if possible?
    task_dict = {
        'get_tomato'    : 'get tomatoes from the fridge',
        'get_onion'     : 'get onions from the fridge',
        'place_tomato'  : 'place tomatoes in cooking pot',
        'place_onion'   : 'place onions in cooking pot',
        'get_dish'      : 'get dish from the cupboard',
        'place_dish'    : 'place dish on the table',
        'plate_dish'    : 'plate dish',
        'serve_dish'    : 'serve dish to customer'
    }

    tasks_nl = "".join([task_dict[task] + ", " for task in task_list])

    prompt = '''Let's say we're playing the game Overcooked together.\
                Suggest a role description for the task list.
                 
                Task list: get tomatoes from the fridge, get onions from the fridge, place tomatoes in cooking pot, place onions in cooking pot,
                Role: prep chef

                Task list: get tomatoes from the fridge, place tomatoes in cooking pot,
                Role: tomato chef

                Task list: {}
                Role:'''.format(tasks_nl)

    return prompt