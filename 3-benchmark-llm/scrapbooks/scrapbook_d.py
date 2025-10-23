import dotenv
from openai import OpenAI

def main():
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")

    client = OpenAI()

    response = client.responses.create(
        model="gpt-5-mini",
        tools=[{"type": "web_search"}],
        input="What was a positive news story from today?"
    )

    print(response.output_text)

if __name__ == "__main__":
    main()