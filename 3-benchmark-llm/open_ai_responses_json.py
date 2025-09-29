import dotenv
import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel

class ListOfStrings(BaseModel):
    completions: list[str]

questions = [
    'How do you access the printing options of the "Sheet1" worksheet?\nvar printingOpt = workbook.???["Sheet1"].???;\n',
    'How do you set the worksheet zoom level to 140%?\nworksheet.???=???;\n',
]

async def main():
    dotenv.load_dotenv()
    assert dotenv.dotenv_values().values(), ".env file not found or empty"

    client = AsyncOpenAI()

    async_responses = [client.responses.parse(
        prompt={"id": "pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2"},
        # model="gpt-4o-2024-08-06",
        input=[{"role": "user", "content": q}],
        text_format=ListOfStrings,
    ) for q in questions]

    responses = await asyncio.gather(*async_responses)
    for res in responses:
        print(res.output_parsed.completions)

if __name__ == "__main__":
    asyncio.run(main())