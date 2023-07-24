import openai
import yaml

class GPTRolePrompter:
    def __init__(self):
        with open("config.yaml", "r") as stream:
            self.api_key = yaml.safe_load(stream)["OPENAI_API_KEY"]
            openai.api_key = self.api_key

    def get_llm_response(self, prompt, temperature=0.6):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=prompt,
            temperature=0.0
        )
        return response

    def generate_role_prompt(self, n_players, subtask_list):
        prompt = '''Given a list of tasks in the game Overcooked, suggest a few different ways I could divide up the tasks into {} roles.

                    Tasks: {}

                    Format each suggestion as [Role name]: [Task list]
                    '''.format(str(n_players), subtask_list)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return messages
    
    def prompt(self, subtask_list):
        query = self.generate_role_prompt(2, subtask_list)
        response = self.get_llm_response(query)
        print(response)

    def parse_gpt_response(self, text):
        # TODO: parse the response from GPT to get the role suggestions
        # maybe use regex?
        raise NotImplementedError


if __name__ == "__main__":
    gpt = GPTRolePrompter()
    gpt.prompt("""['Grabbing an onion from dispenser', 'Grabbing a tomato from dispenser,
                         'Putting onion in pot', 'Putting tomato in pot',
                         'Grabbing dish from dispenser', 'Grabbing dish from counter',
                         'Placing dish closer to pot', 'Getting the soup',
                         'Grabbing soup from counter', 'Placing soup closer',
                         'Serving the soup']""")
