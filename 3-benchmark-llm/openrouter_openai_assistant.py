import dotenv
import asyncio

from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

# Pydantic model for structured output.
class CodeCompletion(BaseModel):
    completions: List[str]

# Async main.
async def main():
    # Load environment variables from parent directory .env.
    dotenv.load_dotenv()
    assert dotenv.dotenv_values().values(), ".env file not found or empty"

    # Use OpenAI Responses API to call an assistant with ID "asst_9NmTbjXkB2t7PxvQZJwgD9Pf"
    # model_name = "asst_9NmTbjXkB2t7PxvQZJwgD9Pf"
    # model_name = "openai:gpt-4o-mini"
    model_name = "pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2"
    agent_code = Agent(model_name, output_type=CodeCompletion)
    full_prompt = f"How do you set the value of cell A1 to Hello?\nworksheet.Cells[???].??? = ???;\n"
    result = await asyncio.wait_for(agent_code.run(full_prompt), timeout=30)
    print(str(result.output.completions))

if __name__ == "__main__":
    asyncio.run(main())