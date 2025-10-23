import asyncio
import dotenv
import httpx

from typing import Type
from pydantic import BaseModel
from openai import AsyncOpenAI

class OpenAIPromptAgent:
    def __init__(self, prompt_id: str, passed_output_type: Type[BaseModel]):
        self.client = AsyncOpenAI()
        self.prompt = {"id": prompt_id}
        self.passed_output_type = passed_output_type

    async def run(self, input: str) -> httpx.Response:
        return await self.client.responses.parse(
            prompt=self.prompt, 
            input=[{"role": "user", "content": input}],
            tools=[{"type": "web_search_preview"}],
            text_format=self.passed_output_type)

    def response_2_results_tokens(self, response: httpx.Response) -> tuple[list[str], int, int]:
        return response.output_parsed.completions, response.usage.input_tokens, response.usage.output_tokens

questions = [
    "How to set value of A1 to 'Abracadabra'?",
    # "How to format B2 to bold?",
    # "How to print sheet?"
]

class ListOfStrings(BaseModel):
    completions: list[str]

async def main():
    dotenv.load_dotenv()
    assert dotenv.dotenv_values().values(), ".env file not found or empty"

    agent = OpenAIPromptAgent(
            prompt_id="pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2",
            passed_output_type=ListOfStrings)

    async_responses = [agent.run(q) for q in questions]

    responses = await asyncio.gather(*async_responses)
    for res in responses:
        usage, results = agent.response_2_usage_results(res)
        print(f"\nUsage: {usage}\nResults: {results}")

if __name__ == "__main__":
    asyncio.run(main())