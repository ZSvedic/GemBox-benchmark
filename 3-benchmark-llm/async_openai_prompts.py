import asyncio
import dotenv
from openai import AsyncOpenAI
from pydantic_ai import Agent

class MyPromptAgent(Agent):
    def __init__(self, prompt: dict):
        self.client = AsyncOpenAI()
        self.prompt = prompt

    async def run(self, input: str):
        r = await self.client.responses.create(prompt=self.prompt, input=input)
        return r.output_text

questions = [
    "How to set value of A1?",
    "How to format bold?",
    "How to print sheet?"
]

async def main():
    dotenv.load_dotenv()
    assert dotenv.dotenv_values().values(), ".env file not found or empty"

    prompt_agen = MyPromptAgent(prompt={"id": "pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2", "version": "3"})

    results = await asyncio.gather(*(prompt_agen.run(q) for q in questions))
    for res in results:
        print(res)

if __name__ == "__main__":
    asyncio.run(main())

# OLD CODE:
# async def ask(question: str):
#     r = await client.responses.create(
#         prompt={"id": "pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2", "version": "3"},
#         input=question
#     )
#     return r.output_text

# async def main():
#     questions = [
#         "How to set value of A1?",
#         "How to format bold?",
#         "How to print sheet?"
#     ]
#     results = await asyncio.gather(*(ask(q) for q in questions))
#     for res in results:
#         print(res)

# asyncio.run(main())