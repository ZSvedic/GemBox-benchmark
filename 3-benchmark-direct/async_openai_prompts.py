import dotenv
import asyncio
import httpx

from pprint import pprint
from typing import Type, Protocol
from pydantic import BaseModel
from openai import AsyncOpenAI, Omit, omit

# ToDO: how to implement Init or factory method? 
class AgentProtocol(Protocol):
    async def run(self, input: str) -> httpx.Response: ...
    def response_2_results_tokens(self, response: httpx.Response) -> tuple[list[str], int, int]: ...

class OpenAIPromptAgent(AgentProtocol):
    def __init__(
        self, *, 
        model: str | Omit = omit, 
        prompt_id: str | Omit = omit, 
        system_prompt: str | Omit = omit,
        parse_type: Type[BaseModel] | Omit = omit,
        web_search: bool = False,
        verbose: bool = True):

        if bool(model) == bool(prompt_id):
            raise ValueError("Provide exactly one of model_name or prompt_id")

        self.model = model
        self.prompt = {"id": prompt_id} if prompt_id else omit
        self.system_prompt = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self.parse_type = parse_type
        self.tools = [{"type": "web_search_preview"}] if web_search else omit
        self.verbose = verbose
        self.client = AsyncOpenAI()

    async def run(self, input: str) -> httpx.Response:
        input = [*self.system_prompt, {"role": "user", "content": input}]
        if self.verbose:
            pprint(input)
        return await self.client.responses.parse(
            model=self.model,
            prompt=self.prompt, 
            input=input,
            tools=self.tools,
            text_format=self.parse_type)

    def response_2_results_tokens(self, response: httpx.Response) -> tuple[list[str], int, int]:
        result = response.output_parsed.completions if response.output_parsed else [response.output_text]
        usage = response.usage 
        return result, usage.input_tokens, usage.output_tokens

class ListOfStrings(BaseModel):
    completions: list[str]

PROGRAMMING_PROMPT = """Answer a coding question related to GemBox Software .NET components.
Return a JSON object with a 'completions' array containing only the code strings that should replace the ??? marks, in order. 
Completions array should not contain any extra whitespace as results will be used for string comparison.

Example question: 
How do you set the value of cell A1 to "Hello"?
worksheet.Cells[???].??? = ???;
Your response:
{'completions': ['"A1"', 'Value', '"Hello"']}

Below '--- QUESTION AND MASKED CODE:' line is the question and masked code. Return only the JSON object with no explanations, comments, or additional text.

--- QUESTION AND MASKED CODE: """

_QUESTIONS = [
    "How to set value of A1 to 'Abracadabra'?",
    "How to format B2 to bold?",
]

async def run_agent(agent: OpenAIPromptAgent):
    async_responses = [agent.run(q) for q in _QUESTIONS]
    responses = await asyncio.gather(*async_responses)
    for res in responses:
        results, input_tokens, output_tokens = agent.response_2_results_tokens(res)
        print(f"\nResults: {results}\nInput tokens: {input_tokens}\nOutput tokens: {output_tokens}")

async def main():
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")

    try:
        agent = OpenAIPromptAgent()
    except ValueError as e:
        print('PASS: Calling OpenAIPromptAgent without model or prompt_id throws a ValueError.')

    # Test with model and system prompt.
    agent = OpenAIPromptAgent(model="gpt-5-mini", system_prompt=PROGRAMMING_PROMPT, parse_type=ListOfStrings)
    await run_agent(agent)

    # Test with prompt_id.
    agent = OpenAIPromptAgent(prompt_id="pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2", parse_type=ListOfStrings)
    await run_agent(agent)

if __name__ == "__main__":
    asyncio.run(main())