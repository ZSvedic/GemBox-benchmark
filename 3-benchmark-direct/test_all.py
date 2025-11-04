import asyncio

import dotenv

import async_google_prompt as agp
import async_openai_prompts as aop
import base_classes as bc
import benchmark as bm

async def main_test():
    await bc.main_test()
    await aop.main_test()
    await agp.main_test()
    await bm.main_test()

if __name__ == "__main__":
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")
        
    asyncio.run(main_test())