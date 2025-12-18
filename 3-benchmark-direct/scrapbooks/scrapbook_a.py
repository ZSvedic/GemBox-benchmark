import dotenv
if not dotenv.load_dotenv():
    raise FileExistsError(".env file not found or empty")

from openai import OpenAI
client = OpenAI()

padding_text = 'p932765a023474d692363d429823i675937n0121174g-'
limit = 50_000 * (4+1)  # 4 bytes per character + 1 for safety
n_repeats = limit // len(padding_text) + 1
prompt = padding_text*n_repeats + "\n Answer what is the height of Eiffel Tower in meters?"

print(f'n_repeats: {n_repeats}, len(prompt): {len(prompt)}')
print(f'prompt[:100]: {prompt[:100]}...')
print(f'prompt[limit:]: {prompt[limit:]}\n')

response = client.responses.create(model="gpt-5-mini", reasoning={"effort": "low"}, input=prompt)

print(response.output_text)
