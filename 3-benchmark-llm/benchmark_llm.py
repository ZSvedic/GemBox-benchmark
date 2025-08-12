import dotenv
import asyncio
from typing import Literal
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

dotenv.load_dotenv()

# Question prompts.
QUESTIONS = [
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "Explain the theory of relativity in simple terms.",
]

# Models with input and output costs.
MODELS_WITH_COSTS = [
    ('openai/gpt-4o-mini', 0.15, 0.60),
    ('openai/gpt-5-mini', 0.25, 2.00),
    ('openai/gpt-5-nano', 0.05, 0.40),
    ('anthropic/claude-3-haiku', 0.25, 1.35),
    ('google/gemini-2.5-flash', 0.30, 2.50),
    ('google/gemini-2.5-flash-lite', 0.10, 0.40),
    ('mistralai/codestral-2508', 0.30, 0.90),
    ('deepseek/deepseek-chat-v3-0324', 0.18, 0.72),
]

# Extract just the names for your existing functions.
MODELS = [model[0] for model in MODELS_WITH_COSTS]

def get_model(model_string: str):
    """Get a model instance from a model string."""
    return OpenAIModel(model_string, provider=OpenRouterProvider())

async def benchmark_model_questions(model_name: str):
    """Benchmark one model on all 3 questions."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name} on all questions")
    print(f"{'='*60}")
    
    model = get_model(model_name)
    agent = Agent(model)
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 40)
        
        try:
            result = await agent.run(question)
            print(f"Answer: {result.output}")
        except Exception as e:
            print(f"Error: {e}")

async def benchmark_models_question(model_names: list[str], question_index: int = 1):
    """Benchmark multiple models on a specific question (default: second question - Mockingbird)."""
    question = QUESTIONS[question_index]
    print(f"\n{'='*60}")
    print(f"Benchmarking all models on question {question_index + 1}: {question}")
    print(f"{'='*60}")
    
    for model_name in model_names:
        print(f"\n--- Testing {model_name} ---")
        print("-" * 40)
        
        try:
            model = get_model(model_name)
            agent = Agent(model)
            result = await agent.run(question)
            print(f"Answer: {result.output}")
        except Exception as e:
            print(f"Error with {model_name}: {e}")

async def main():
    print("Starting LLM Benchmark Suite...")
    
    # Benchmark 1: Single model on all questions
    # await benchmark_model_questions('openai/gpt-4o-mini')
    
    # Benchmark 2: All models on the Mockingbird question (index 1)
    await benchmark_models_question(MODELS, question_index=1)

if __name__ == "__main__":
    asyncio.run(main())