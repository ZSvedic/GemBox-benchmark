import asyncio
import threading
from pprint import pprint
from typing import Any, override

import dotenv
from openai import AsyncOpenAI, omit

import base_classes as bc

class OpenAIHandler(bc.LLMHandler):
    # Client is a singleton that requires close()...
    _client = None
    # ... and is protected by a lock.
    _lock = threading.Lock()

    @classmethod
    def get_client(cls):
        with cls._lock:
            if cls._client is None:
                cls._client = AsyncOpenAI()
            return cls._client

    @override
    @classmethod
    async def close(cls):
        with cls._lock:
            if cls._client:
                await cls._client.close()
                cls._client = None

    @override
    async def call(self, input: str) -> tuple[Any, bc.CallDetailsType, bc.UsageType]: 
        if self.model_info.prompt_id: # OpenAI prompt.
            model = omit
            prompt_dict = {"id": self.model_info.prompt_id}
        else: # OpenAI model.
            model = self.model_info.name
            prompt_dict = omit

        input_dict = [{"role": "user", "content": input}]

        if self.system_prompt:
            input_dict = [{"role": "system", "content": self.system_prompt}] + input_dict

        if self.web_search:
            tools_dict = [{"type": "web_search"}]
            include_list = ["web_search_call.action.sources"]
        else:
            tools_dict = include_list = omit

        if self.verbose:
            pprint(input_dict)

        if self.parse_type:
            format = self.parse_type
        else:
            format = omit

        response = await OpenAIHandler.get_client().responses.parse(
            model=model, prompt=prompt_dict, input=input_dict, tools=tools_dict, include=include_list, text_format=format)

        result = response.output_parsed if self.parse_type else response.output_text
        links = self.get_web_search_links(response)
        usage = response.usage 

        if self.verbose:
            print(f"result: {result}\nlinks: {links}")

        return result, links, (usage.input_tokens, usage.output_tokens)

    def get_web_search_links(self, response) -> bc.CallDetailsType:
        if not self.web_search:
            return None

        links = {}
        for item in response.output:
            if item.type == "web_search_call":
                action = item.action
                match action.type:
                    case "search":
                        links[f"search(query={action.query})"] = [s.url for s in (action.sources or [])]
                    case "open_page":
                        links["open_page"] = action.url
                    case other:
                        print(f"WARNING: Unexpected web_search_call action type: {other}")

        print(f"DEBUG: Found {sum(len(v) for v in links.values())} web search links."
              if links else f"WARNING: web_search True but no links were returned.")
        
        return links

# OpenAI models.

_OPENAI_MODELS = [
    # ModelInfo('name', prompt_id, input_cost, output_cost, context_length, direct_class, tags),
    # TEMPLATE:
    # ModelInfo('', None, 0.0, 0.0, 0, None, {''}),
    # OpenAI models: https://openrouter.ai/provider/openai
    bc.ModelInfo('gpt-3.5-turbo', None, 0.50, 1.50, 16_385, OpenAIHandler, False, {'openai', 'fast', 'old'}),
    bc.ModelInfo('gpt-4.1', None, 2.0, 8.0, 1_050_000, OpenAIHandler, False, {'openai', 'fast', 'old'}),
    bc.ModelInfo('gpt-4o-2024-11-20', None, 2.5, 10.0, 128_000, OpenAIHandler, False, {'openai', 'old'}),
    bc.ModelInfo('gpt-4o-mini', None, 0.15, 0.60, 128_000, OpenAIHandler, True, {'openai', 'fast', 'old'}), 
    bc.ModelInfo('gpt-5-nano', None, 0.05, 0.40, 400_000, OpenAIHandler, True, {'openai', 'fast'}),            
    bc.ModelInfo('gpt-5-mini', None, 0.25, 2.00, 400_000, OpenAIHandler, True, {'openai', 'fast',}), 
    bc.ModelInfo('gpt-5', None, 1.25, 10.00, 400_000, OpenAIHandler, True, {'openai'}),  
    bc.ModelInfo('gpt-5-codex', None, 1.25, 10.0, 400_000, OpenAIHandler, True, {'openai'}),  
    bc.ModelInfo('gpt-5.1', None, 1.25, 10.00, 400_000, OpenAIHandler, True, {'openai'}), 
    bc.ModelInfo('gpt-5.1-codex', None, 1.25, 10.00, 400_000, OpenAIHandler, True, {'openai'}),
    bc.ModelInfo('gpt-5.1-codex-mini', None, 0.25, 2.00, 400_000, OpenAIHandler, True, {'openai'}),   
    # OpenAIPrompt models (Zel's private account): 
    # https://platform.openai.com/chat/edit?prompt=pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2&version=4
    # Prompt version 4 uses gpt-5-mini.
    bc.ModelInfo('prompt-GBS-examples-GPT5mini', 'pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2', 
                 0.25, 2.00, 400_000, OpenAIHandler, False, {'openai', 'prompt'}),
    bc.ModelInfo('prompt-GBS-examples-GPT5', 'pmpt_68ee4f81f8d4819786ff5301af701ced0843964564bf8684', 
                 1.25, 10.00, 400_000, OpenAIHandler, False, {'openai', 'prompt'}),
]
    
bc.Models._MODEL_REGISTRY += _OPENAI_MODELS

# Main test functions.

async def main_test():
    print("\n===== async_openai_prompts.main_test() =====")

    # Test with model default system prompt and web search.
    handler = bc.Models().by_name('gpt-5-mini').create_handler(system_prompt=bc._DEFAULT_SYSTEM_PROMPT, web_search=True, parse_type=bc.ListOfStrings)
    await bc._test_call_handler(handler, bc._TEST_QUESTIONS)

    # Test with prompt_id.
    handler = bc.Models().by_name('prompt-GBS-examples-GPT5mini').create_handler(parse_type=bc.ListOfStrings)
    await bc._test_call_handler(handler, bc._TEST_QUESTIONS)

if __name__ == "__main__":
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")
        
    asyncio.run(main_test())