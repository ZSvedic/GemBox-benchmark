import asyncio

import dotenv

import async_openai_prompts # Required to populate model registry.
import async_google_prompt # Required to populate model registry.
import base_classes as bc

async def main_test():
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")

    all_models = bc.Models()

    all_models.print_by_tags()

    model = all_models.by_name('gpt-5-mini')
    handler = model.create_handler(system_prompt="Answer questions about geography.", web_search=False)
    results, input_tokens, output_tokens = await handler.call("What is the capital of France?")
    print(f"results: {results}, input_tokens: {input_tokens}, output_tokens: {output_tokens}")

    model = all_models.by_name('gemini-2.5-flash')
    handler = model.create_handler(system_prompt="Answer questions about mathematics.", web_search=False)
    results, input_tokens, output_tokens = await handler.call("Calculate 3 * 100.")
    print(f"results: {results}, input_tokens: {input_tokens}, output_tokens: {output_tokens}")
    
if __name__ == "__main__":
    asyncio.run(main_test())