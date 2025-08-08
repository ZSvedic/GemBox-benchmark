import dotenv
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

dotenv.load_dotenv()

model = OpenAIModel('openai/gpt-4o', provider=OpenRouterProvider())
# model = OpenAIModel('anthropic/claude-3.5-sonnet', provider=OpenRouterProvider())

agent = Agent(model)

async def main():
    result = await agent.run("What is the meaning of life? Be concise. At the end say '-- by <YourModelNameAndVersion>'")
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
