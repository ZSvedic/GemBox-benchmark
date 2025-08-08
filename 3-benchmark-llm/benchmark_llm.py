# Question prompts
QUESTIONS = [
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "Explain the theory of relativity in simple terms.",
]

# Setup
import dotenv
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

dotenv.load_dotenv()

model = OpenAIModel('openai/gpt-4o-mini', provider=OpenRouterProvider())
# model = OpenAIModel('anthropic/claude-3.5-sonnet', provider=OpenRouterProvider())

agent = Agent(model)

async def main():
    print("Starting LLM Benchmark...")
    print("=" * 50)
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 30)
        
        try:
            result = await agent.run(question)
            print(f"Answer: {result.output}")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    print("Benchmark completed!")

if __name__ == "__main__":
    asyncio.run(main())
