import dotenv
if not dotenv.load_dotenv():
    raise FileExistsError(".env file not found or empty")

from openai import OpenAI
from openai.types.responses import ResponsePromptParam
client = OpenAI()

from pydantic import BaseModel
class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

prompt_dict: ResponsePromptParam = {
        "id": "pmpt_abc123",
        "version": "2",
        "variables": {
            "customer_name": "Jane Doe",
            "product": "40oz juice box"
        },
    }

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    prompt=prompt_dict,
    input=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    text_format=CalendarEvent)

response = client.responses.parse(
    model="gpt-4o-2024-08-06",
    prompt={
        "id": "pmpt_abc123",
        "version": "2",
        "variables": {
            "customer_name": "Jane Doe",
            "product": "40oz juice box"
        },
    },
    input=[
        {"role": "system", "content": "Extract the event information."},
        {
            "role": "user",
            "content": "Alice and Bob are going to a science fair on Friday.",
        },
    ],
    text_format=CalendarEvent)


event = response.output_parsed