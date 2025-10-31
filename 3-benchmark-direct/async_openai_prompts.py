import asyncio
from pprint import pprint
from typing import override

import dotenv
from openai import AsyncOpenAI, Omit, omit
from pydantic import BaseModel

import base_classes as bc

class OpenAIHandler(bc.LLMHandler):
    @override
    def __init__(
        self, 
        model_info: bc.ModelInfo, 
        *,
        system_prompt: str | Omit = omit, 
        parse_type: type[BaseModel] | Omit = omit,
        web_search: bool | Omit = omit,
        verbose: bool = True): 

        self.model_info = model_info
        if model_info.prompt_id: # OpenAI prompt.
            self.model = omit
            self.prompt = {"id": model_info.prompt_id}
        else: # OpenAI model.
            self.model = model_info.name
            self.prompt = omit

        self.system_prompt = [{"role": "system", "content": system_prompt}] if system_prompt else []
        self.parse_type = parse_type
        self.tools = [{"type": "web_search_preview"}] if web_search else []
        self.verbose = verbose
        self.client = AsyncOpenAI()

    @override
    async def call(self, input: str) -> tuple[list[str], int, int]: 
        input = [
            *self.system_prompt, 
            {"role": "user", "content": input}]

        if self.verbose:
            pprint(input)

        response = await self.client.responses.parse(model=self.model, prompt=self.prompt, input=input, tools=self.tools, text_format=self.parse_type)
        result = response.output_parsed.completions if response.output_parsed else [response.output_text]
        usage = response.usage 

        return result, usage.input_tokens, usage.output_tokens

_OPENAI_MODELS = [
    # ModelInfo('name', prompt_id, input_cost, output_cost, context_length, direct_class, tags),
    # TEMPLATE:
    # ModelInfo('', None, 0.0, 0.0, 0, None, {''}),
    # OpenAI models: https://openrouter.ai/provider/openai
    bc.ModelInfo('gpt-3.5-turbo', None, 0.50, 1.50, 16_385, OpenAIHandler, {'openai', 'fast'}),
    bc.ModelInfo('gpt-4.1', None, 2.0, 8.0, 1_050_000, OpenAIHandler, {'openai', 'fast'}),
    bc.ModelInfo('gpt-4o-2024-11-20', None, 2.5, 10.0, 128_000, OpenAIHandler, {'openai'}),
    bc.ModelInfo('gpt-4o-mini', None, 0.15, 0.60, 128_000, OpenAIHandler, {'openai', 'fast'}), 
    bc.ModelInfo('gpt-5-nano', None, 0.05, 0.40, 400_000, OpenAIHandler, {'openai', 'fast'}),            
    bc.ModelInfo('gpt-5-mini', None, 0.25, 2.00, 400_000, OpenAIHandler, {'openai', 'fast',}), 
    bc.ModelInfo('gpt-5', None, 1.25, 10.00, 400_000, OpenAIHandler, {'openai', 'accurate'}),  
    bc.ModelInfo('gpt-5-codex', None, 1.25, 10.0, 400_000, OpenAIHandler, {'openai'}), # Doesn't work directly?      
    # OpenAIPrompt models (Zel's private account): 
    # https://platform.openai.com/chat/edit?prompt=pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2&version=4
    # Prompt version 4 uses gpt-5-mini.
    bc.ModelInfo('prompt-GBS-examples-GPT5mini', 'pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2', 0.25, 2.00, 400_000, OpenAIHandler, {'openai', 'prompt'}),
    bc.ModelInfo('prompt-GBS-examples-GPT5', 'pmpt_68ee4f81f8d4819786ff5301af701ced0843964564bf8684', 1.25, 10.00, 400_000, OpenAIHandler, {'openai', 'prompt'}),
]
    
bc.Models._MODEL_REGISTRY += _OPENAI_MODELS

async def call_handler(handler: OpenAIHandler):
    async_responses = [handler.call(q) for q in bc._TEST_QUESTIONS]
    responses = await asyncio.gather(*async_responses)
    for results, input_tokens, output_tokens in responses:
        print(f"\nResults: {results}\nInput tokens: {input_tokens}\nOutput tokens: {output_tokens}")

async def main_test():
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")

    # Test with model and system prompt.
    handler = bc.Models().by_name('gpt-5-mini').create_handler(system_prompt=bc._DEFAULT_SYSTEM_PROMPT, parse_type=bc.ListOfStrings)
    await call_handler(handler)

    # Test with prompt_id.
    handler = bc.Models().by_name('prompt-GBS-examples-GPT5mini').create_handler(parse_type=bc.ListOfStrings)
    await call_handler(handler)

if __name__ == "__main__":
    asyncio.run(main_test())