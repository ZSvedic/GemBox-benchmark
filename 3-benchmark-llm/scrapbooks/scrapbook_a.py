# Calling the OpenAI API responses API (not chat API) works with a web search tool.

import dotenv
if not dotenv.load_dotenv():
    raise FileExistsError(".env file not found or empty")

from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    tools=[{"type": "web_search_preview"}],
    input="What was a positive news story from today?"
)

print(response.output_text)