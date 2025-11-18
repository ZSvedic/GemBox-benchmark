import asyncio
import threading
import os
from typing import Any, override

import dotenv
from openai import AsyncOpenAI

import base_classes as bc

class OpenRouterHandler(bc.LLMHandler):
    # Client is a singleton that requires close()...
    _client = None
    # ... and is protected by a lock.
    _lock = threading.Lock()

    @classmethod
    def get_client(cls):
        with cls._lock:
            if cls._client is None:
                cls._client = AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ.get("OPENROUTER_API_KEY")
                )
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
        messages = []
        model_name = self.model_info.name + (":online" if self.web_search else "")
        
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        messages.append({"role": "user", "content": input})

        if self.verbose:
            print(f"Calling model: {model_name}")
            print(f"Messages: {messages}")

        # Call OpenRouter API
        response = await OpenRouterHandler.get_client().chat.completions.create(model=model_name, messages=messages)

        # Extract response
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON if parse_type is specified
        if self.parse_type:
            result_text = bc.LLMHandler.strip_code_fences(result_text)
            result = self.parse_type.model_validate_json(result_text)
        else:
            result = result_text
        
        # Extract usage
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        if self.verbose:
            print(f"result: {result}")
            print(f"input_tokens: {input_tokens}, output_tokens: {output_tokens}")

        # OpenRouter doesn't provide web search links in the same way
        links = None

        return result, links, (input_tokens, output_tokens)

# OpenRouter models registry.

_OPENROUTER_MODELS = [
    # ModelInfo('name', prompt_id, input_cost, output_cost, context_length, direct_class, web_search, tags),
    # TEMPLATE:
    # ModelInfo('', None, 0.0, 0.0, 0, None, False, {''}),
    
    # Anthropic models: https://openrouter.ai/provider/anthropic
    bc.ModelInfo('anthropic/claude-3-5-haiku', None, 0.25, 1.35, 200_000, OpenRouterHandler, False, {'anthropic', 'openrouter'}),
    # bc.ModelInfo('anthropic/claude-sonnet-4.5', None, 3.0, 15.00, 1_000_000, OpenRouterHandler, True, {'anthropic', 'openrouter'}), # Too expensive.
    # bc.ModelInfo('anthropic/claude-opus-4.1', None, 15.00, 75.00, 200_000, OpenRouterHandler, True, {'anthropic', 'openrouter'}), # Too expensive.
    
    # Mistral models: https://openrouter.ai/provider/mistral
    bc.ModelInfo('mistralai/codestral-2508', None, 0.30, 0.90, 256_000, OpenRouterHandler, False, {'mistral', 'openrouter'}),
    bc.ModelInfo('mistralai/devstral-medium', None, 0.40, 2.00, 131_000, OpenRouterHandler, True, {'mistral', 'openrouter'}),
    bc.ModelInfo('mistralai/mistral-large', None, 2.0, 6.0, 128_000, OpenRouterHandler, True, {'mistral', 'openrouter'}),
    
    # DeepSeek models: https://openrouter.ai/provider/deepseek
    bc.ModelInfo('deepseek/deepseek-chat', None, 0.14, 0.28, 64_000, OpenRouterHandler, True, {'deepseek', 'openrouter'}),
    bc.ModelInfo('deepseek/deepseek-r1', None, 0.55, 2.19, 64_000, OpenRouterHandler, True, {'deepseek', 'openrouter'}),

    # MoonshotAI models: https://openrouter.ai/provider/moonshotai
    bc.ModelInfo('moonshotai/kimi-k2', None, 0.60, 2.5, 131_100, OpenRouterHandler, True, {'moonshotai', 'openrouter'}),
]

bc.Models._MODEL_REGISTRY += _OPENROUTER_MODELS

# Main test functions.

async def main_test():
    print("===== open_router.main_test() =====")

    # Load questions from the test questions
    test_questions = bc._TEST_QUESTIONS[:5]  # Use only 5 questions for testing
    
    # Test web search.
    for model in _OPENROUTER_MODELS:
        if model.has_web_search:
            print(f"\n--- Testing model: {model.name} (web_search=True) ---")
            handler = bc.Models().by_name(model.name).create_handler(web_search=True)
            await bc._test_call_handler(handler, ["Give me the second news item from news.ycombinator.com right now?"])

    # Test with DeepSeek (fast and cheap)
    model = bc.Models().by_name('deepseek/deepseek-chat')
    print(f"\nTesting model: {model.name}")
    handler = model.create_handler(
        system_prompt=bc._DEFAULT_SYSTEM_PROMPT, 
        parse_type=bc.ListOfStrings,
        web_search=False,
        verbose=True)
    await bc._test_call_handler(handler, test_questions)
    await handler.close()
    
    # Test with Mistral Codestral
    model = bc.Models().by_name('mistralai/codestral-2508')
    print(f"\nTesting model: {model.name}")
    handler = model.create_handler(
        system_prompt=bc._DEFAULT_SYSTEM_PROMPT, 
        parse_type=bc.ListOfStrings,
        web_search=False,
        verbose=True)
    await bc._test_call_handler(handler, test_questions)
    await handler.close()

if __name__ == "__main__":
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")
        
    asyncio.run(main_test())
