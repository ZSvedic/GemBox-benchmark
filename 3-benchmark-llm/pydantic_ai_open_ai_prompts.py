import dotenv
import asyncio

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel

# 1. Define your structured output
class Joke(BaseModel):
    setup: str
    punchline: str

async def main():
    dotenv.load_dotenv()
    assert dotenv.dotenv_values().values(), ".env file not found or empty"

    # 2. Create an agent with OpenAIResponsesModel
    agent = Agent(
        # model=OpenAIResponsesModel("gpt-5-mini"),
        model="pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2",
        output_type=Joke,
    )

    # 3. Run it
    response = await agent.run("Tell me a short joke about programmers")
    print(response.output)  # Joke(setup='...', punchline='...')

if __name__ == "__main__":
    asyncio.run(main())