# Python stdlib.
import asyncio
import dataclasses as dc
import os
import threading
from typing import Any, override

# Third-party.
import dotenv
from openai import AsyncOpenAI

# Local modules.
import base_classes as bc

class OpenRouterHandler(bc.LLMHandler):
    # Client is a singleton that requires close()...
    _client = None
    # ... and is protected by a lock.
    _lock = threading.Lock()

    @override
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
    @classmethod
    def provider_name(cls) -> str:
        return "OpenRouter"

    @override
    async def call(self, input: str) -> tuple[Any, bc.CallDetailsType, bc.UsageType]:
        messages = []
        model_name = self.model_info.name 
        
        if self.web:
            if self.model_info.web is False:
                raise ValueError(f"Model {model_name} does not support web search")
            else:
                model_name += ':online'
        
        if self.system_ins:
            messages.append({"role": "system", "content": self.system_ins})
        
        messages.append({"role": "user", "content": input})

        if self.verbose:
            print(f"Calling model: {model_name}")
            print(f"Messages: {messages}")

        # Call OpenRouter API
        response = await OpenRouterHandler.get_client().chat.completions.create(
            model=model_name, messages=messages)

        # Extract response
        result_text = response.choices[0].message.content.strip()
        
        # Parse JSON if parse_type is specified
        if self.parse_type:
            result_text = bc.LLMHandler.strip_code_fences(result_text)
            result = self.parse_type.model_validate_json(result_text)
        else:
            result = result_text
        
        # Extract usage
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        if self.verbose:
            print(f"result: {result}")
            print(f"input_tokens: {input_tokens}, output_tokens: {output_tokens}")

        # OpenRouter doesn't provide web web links in the same way
        links = None

        return result, links, (input_tokens, output_tokens)

# OpenRouter models.

_base = bc.ModelInfo(handler=OpenRouterHandler)

_OPENROUTER_MODELS = [
    # Anthropic models: https://openrouter.ai/provider/anthropic
    dc.replace(_base, name='anthropic/claude-3-5-haiku',    in_usd= 0.25, out_usd= 1.35, context_len=  200_000, web=False, tags={'anthropic', 'openrouter'}),
    # dc.replace(_base, name='anthropic/claude-sonnet-4.5',   in_usd= 3.00, out_usd=15.00, context_len=1_000_000, web=True,  tags={'anthropic', 'openrouter'}), # Too expensive.
    # dc.replace(_base, name='anthropic/claude-opus-4.1',     in_usd=15.00, out_usd=75.00, context_len=  200_000, web=True,  tags={'anthropic', 'openrouter'}), # Too expensive.
    # dc.replace(_base, name='anthropic/claude-opus-4.5',     in_usd= 5.00, out_usd=25.00, context_len=  200_000, web=True,  tags={'anthropic', 'openrouter'}), # Too expensive.

    # Mistral models: https://openrouter.ai/provider/mistral
    dc.replace(_base, name='mistralai/codestral-2508',      in_usd= 0.30, out_usd= 0.90, context_len=  256_000, web=False, tags={'mistral', 'openrouter'}),
    dc.replace(_base, name='mistralai/devstral-medium',     in_usd= 0.40, out_usd= 2.00, context_len=  131_000, web=True,  tags={'mistral', 'openrouter'}),
    dc.replace(_base, name='mistralai/mistral-large',       in_usd= 2.00, out_usd= 6.00, context_len=  128_000, web=True,  tags={'mistral', 'openrouter'}),

    # DeepSeek models: https://openrouter.ai/provider/deepseek
    dc.replace(_base, name='deepseek/deepseek-chat',        in_usd= 0.14, out_usd= 0.28, context_len=   64_000, web=True,  tags={'deepseek', 'openrouter'}),
    dc.replace(_base, name='deepseek/deepseek-r1',          in_usd= 0.55, out_usd= 2.19, context_len=   64_000, web=True,  tags={'deepseek', 'openrouter'}),        
    
    # MoonshotAI models: https://openrouter.ai/provider/moonshotai
    dc.replace(_base, name='moonshotai/kimi-k2',            in_usd= 0.60, out_usd= 2.50, context_len=  131_100, web=True,  tags={'moonshotai', 'openrouter'}),
    
    # Google's Gemini 3 Pro seems to be available via OpenRouter:
    # https://openrouter.ai/google/gemini-3-pro-preview
    dc.replace(_base, name='google/gemini-3-pro-preview',   in_usd= 2.00, out_usd=12.00, context_len=1_050_000, web=True,  tags={'google', 'openrouter'}),
    
    # xAI models: https://openrouter.ai/provider/xai
    dc.replace(_base, name='x-ai/grok-4.1-fast',            in_usd= 0.00, out_usd= 0.00, context_len=2_000_000, web=False,  tags={'xai', 'openrouter'}),
]

bc.Models._MODEL_REGISTRY += _OPENROUTER_MODELS

# Main test functions.

async def main_test():
    print("\n===== openrouter_handler.main_test() =====")

    # Contexts.
    contexts = [ ('deepseek/deepseek-chat', True), ('mistralai/codestral-2508', False) ]
    
    # Test contexts.
    for name, web in contexts:
        model = bc.Models().by_name(name)
        print(f"\nTesting model: {name}")
        handler = model.create_handler(
            system_ins=bc.DEFAULT_SYSTEM_INS, web=web, parse_type=bc.ListOfStrings, verbose=True)
        await bc._test_call_handler(handler, bc.TEST_QUESTIONS)
        await handler.close()

if __name__ == "__main__":
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")
        
    asyncio.run(main_test())
