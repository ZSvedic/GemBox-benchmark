# Python stdlib.
import asyncio

# Third-party.
import dotenv

# Local modules.
import metrics
import questions
import tee_logging
import base_classes
import openai_handler
import google_handler
import openrouter_handler
import benchmark
import dotnet_cli

# Main test.

async def main_test():
    # Plain calls.
    metrics.main_test()
    questions.main_test()
    tee_logging.main_test()
    dotnet_cli.main_test()
    
    # Async calls.
    await base_classes.main_test()
    await openai_handler.main_test()
    await google_handler.main_test()
    await openrouter_handler.main_test()
    await benchmark.main_test()

if __name__ == "__main__":
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")
        
    asyncio.run(main_test())