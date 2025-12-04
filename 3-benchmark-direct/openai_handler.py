# Python stdlib.
import asyncio
import dataclasses as dc
from pprint import pprint
import threading
from typing import Any, override

# Third-party.
import dotenv
from openai import AsyncOpenAI, omit

# Local modules.
import base_classes as bc

# OpenAI LLM handler.

class OpenAIHandler(bc.LLMHandler):
    # Client is a singleton that requires close()...
    _client = None
    # ... and is protected by a lock.
    _lock = threading.Lock()

    @override
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
    @classmethod
    def provider_name(cls) -> str:
        return "OpenAI"

    @override
    async def call(self, input: str) -> tuple[Any, bc.CallDetailsType, bc.UsageType]: 
        inputs = []
        tools = include_list = omit

        if self.model_info.prompt_id: # OpenAI prompt.
            model = omit
            prompt_dict = {"id": self.model_info.prompt_id}
        else: # OpenAI model.
            model = self.model_info.name
            prompt_dict = omit
            if self.system_ins:
                inputs.append({"role": "system", "content": self.system_ins})

        inputs.append({"role": "user", "content": input})

        if self.web:
            if self.model_info.web is False:
                raise ValueError(f"Model {model} does not support web search")
            else:
                tools = [{"type": "web_search"}]
                if self.include_domains:
                    tools[0]["filters"] = {"allowed_domains": [self.include_domains]}
                include_list = ["web_search_call.action.sources"]

        if self.verbose:
            print(f"=== PROMPT DICT:\n{prompt_dict}\n\n=== TOOLS:\n{tools}\n\n=== INCLUDE LIST:\n{include_list}\n\n=== INPUTS:\n{inputs}")

        # if self.parse_type:
        #     format = self.parse_type
        # else:
        #     format = omit

        response = await OpenAIHandler.get_client().responses.create(
            model=model, prompt=prompt_dict, input=inputs, 
            tools=tools, include=include_list, # text_format=format,
            )

        result = (self.parse_type.model_validate_json(response.output_text) if self.parse_type 
                  else response.output_text)
        # result = response.output_parsed if self.parse_type else response.output_text
        links = self.get_web_search_links(response)
        usage = response.usage 

        if self.verbose:
            print(f"result: {result}\nlinks: {links}")

        return result, links, (usage.input_tokens, usage.output_tokens)

    def get_web_search_links(self, response) -> bc.CallDetailsType:
        if not self.web:
            return None

        links = {}
        for item in response.output:
            if item.type == "web_search_call":
                action = item.action
                match action.type:
                    case "search":
                        links[f"search(query={action.query})"] = [s.url for s in (action.sources or [])]
                    case "open_page":
                        links["open_page"] = [action.url]
                    case "find_in_page":
                        links[f"find_in_page({action.pattern})"] = [action.url]
                    case other:
                        print(f"WARNING: Unexpected web_search_call action type: {other}")

        if links:
            print(f"DEBUG: Found {sum(len(v) for v in links.values())} links:")
            pprint(links, compact=True)
        else:
            print("WARNING: web_search True but no links were returned.")
        
        return links

# OpenAI models.

_base = bc.ModelInfo(handler=OpenAIHandler, tags={'openai'})
_base_prompt = bc.ModelInfo(handler=OpenAIHandler, tags={'openai', 'prompt'})

_OPENAI_MODELS = [
    # OpenAI models: https://openrouter.ai/provider/openai
    dc.replace(_base, name='gpt-4.1',           in_usd=2.00, out_usd= 8.00, context_len=1_050_000, web=False),
    dc.replace(_base, name='gpt-4o-mini',       in_usd=0.15, out_usd= 0.60, context_len=  128_000, web=True), 
    dc.replace(_base, name='gpt-5-nano',        in_usd=0.05, out_usd= 0.40, context_len=  400_000, web=True),
    dc.replace(_base, name='gpt-5-mini',        in_usd=0.25, out_usd= 2.00, context_len=  400_000, web=True),
    dc.replace(_base, name='gpt-5',             in_usd=1.25, out_usd=10.00, context_len=  400_000, web=True),
    dc.replace(_base, name='gpt-5-codex',       in_usd=1.25, out_usd=10.00, context_len=  400_000, web=True),
    dc.replace(_base, name='gpt-5.1',           in_usd=1.25, out_usd=10.00, context_len=  400_000, web=True),
    dc.replace(_base, name='gpt-5.1-codex',     in_usd=1.25, out_usd=10.00, context_len=  400_000, web=True),
    dc.replace(_base, name='gpt-5.1-codex-mini',in_usd=0.25, out_usd= 2.00, context_len=  400_000, web=True),
    # OpenAIPrompt models (Zel's private account): 
    # https://platform.openai.com/chat/edit?prompt=pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2&version=4
    # Prompt version 4 uses gpt-5-mini.
    dc.replace(_base_prompt, name='prompt-GBS-examples-GPT5mini', prompt_id='pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2',
               in_usd=0.25, out_usd= 2.00, context_len=400_000, web=False),
    dc.replace(_base_prompt, name='prompt-GBS-examples-GPT5', prompt_id='pmpt_68ee4f81f8d4819786ff5301af701ced0843964564bf8684',
               in_usd=1.25, out_usd=10.00, context_len=400_000, web=False),
    dc.replace(_base_prompt, name='prompt-web-search-GPT-5-mini', prompt_id='pmpt_6925f3a9ab3c81948b94c2528a16da620b3e3169c57b2f60',
               in_usd=0.25, out_usd= 2.00, context_len=400_000, web=True),
]
    
bc.Models._MODEL_REGISTRY += _OPENAI_MODELS

# Main test.

async def main_test():
    print("\n===== openai_handler.main_test() =====")

    # Test with model default system instructions and web search.
    handler = bc.Models().by_name('gpt-5-mini').create_handler(
        system_ins=bc.DEFAULT_SYSTEM_INS, web=True, parse_type=bc.ListOfStrings, verbose=True)
    await bc._test_call_handler(handler, bc.TEST_QUESTIONS)

    # Test with prompt_id.
    handler = bc.Models().by_name('prompt-GBS-examples-GPT5mini').create_handler(parse_type=bc.ListOfStrings)
    await bc._test_call_handler(handler, bc.TEST_QUESTIONS)

if __name__ == "__main__":
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")
        
    asyncio.run(main_test())