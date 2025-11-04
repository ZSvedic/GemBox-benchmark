import asyncio
import dataclasses as dc
from pprint import pprint
from typing import Any, override

import dotenv
from openai import AsyncOpenAI, Omit, omit
from pydantic import BaseModel

import base_classes as bc

if not dotenv.load_dotenv():
    raise FileNotFoundError(".env file not found or empty")

# OpenAIHandler class.

@dc.dataclass
class OpenAIHandler(bc.LLMHandler):
    model_info: bc.ModelInfo
    system_prompt: str | Omit = omit
    parse_type: type[BaseModel] | Omit = omit
    web_search: bool | Omit = omit
    verbose: bool = False
    client: AsyncOpenAI = AsyncOpenAI()

    @override
    async def call(self, input_dict: str) -> tuple[Any, bc.CallDetailsType, bc.UsageType]: 
        if self.model_info.prompt_id: # OpenAI prompt.
            model = omit
            prompt_dict = {"id": self.model_info.prompt_id}
        else: # OpenAI model.
            model = self.model_info.name
            prompt_dict = omit

        input_dict = [{"role": "user", "content": input_dict}]

        if self.system_prompt:
            input_dict = [{"role": "system", "content": self.system_prompt}] + input_dict

        if self.web_search:
            tools_dict = [{"type": "web_search"}]
            include_list = ["web_search_call.action.sources"]
        else:
            tools_dict = include_list = omit

        if self.verbose:
            pprint(input_dict)

        response = await self.client.responses.parse(
            model=model, prompt=prompt_dict, input=input_dict, tools=tools_dict, include=include_list, text_format=self.parse_type)

        result = response.output_parsed if self.parse_type else response.output_text
        links = self.get_web_search_links(response)
        usage = response.usage 

        if self.verbose:
            print(f"result: {result}\nlinks: {links}")

        return result, links, (usage.input_tokens, usage.output_tokens)

    def get_web_search_links(self, response) -> bc.CallDetailsType:
        links_dict = {
            f'web_search_call: {item.action.query}': [source.url for source in item.action.sources] 
            for item in response.output if item.type == "web_search_call" 
            }

        if self.web_search and not links_dict and self.verbose:
            print("WARNING: web_search is True but no links were returned.")
            
        return links_dict

# OpenAI models.

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

# Main test functions.

async def main_test():
    # Test plain text response for question about today's news.
    handler = bc.Models().by_name('gpt-5-mini').create_handler(web_search=True)
    await bc._test_call_handler(handler, ["What are the latest tech news today, be concise?"])

    # Test with model default system prompt and web search.
    handler = bc.Models().by_name('gpt-5-mini').create_handler(system_prompt=bc._DEFAULT_SYSTEM_PROMPT, web_search=True, parse_type=bc.ListOfStrings)
    await bc._test_call_handler(handler, bc._TEST_QUESTIONS)

    # Test with prompt_id.
    handler = bc.Models().by_name('prompt-GBS-examples-GPT5mini').create_handler(parse_type=bc.ListOfStrings)
    await bc._test_call_handler(handler, bc._TEST_QUESTIONS)

if __name__ == "__main__":
    asyncio.run(main_test())