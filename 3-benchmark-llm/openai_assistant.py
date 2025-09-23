import dotenv
from openai import OpenAI

dotenv.load_dotenv()
assert dotenv.dotenv_values().values(), ".env file not found or empty"

client = OpenAI()

response = client.responses.create(
    # model="pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2",
    prompt={
        "id": "pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2",
        "version": "3"
    },
    input=f"How do you set the value of cell A1 to Hello?\nworksheet.Cells[???].??? = ???;\n"
)

print(response.output_text)