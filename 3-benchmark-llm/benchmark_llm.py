import dotenv
import asyncio
from typing import Literal
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

dotenv.load_dotenv()

# General question prompts.
GENERAL_QUESTIONS = [
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "Explain the theory of relativity in simple terms.",
]

# C# programming questions with masked code snippets.
PROGRAMMING_QUESTIONS = [
    "Complete this C# method to calculate the factorial of a number: public static int Factorial(int n) { if (n <= 1) return 1; return ???; }",
    "Fill in the missing part of this C# LINQ query: var result = numbers.Where(x => ???).Select(x => x * 2).ToList();",
    "Complete this C# async method: public async Task<string> GetDataAsync() { var response = await httpClient.GetAsync(url); return await ???; }",
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

async def benchmark_models_questions(models: list[str], questions: list[str]):
    """Benchmark multiple models on multiple questions."""
    print(f"Benchmarking {len(models)} models on {len(questions)} questions")
    
    for model_name in models:
        print(f"\n--- Testing {model_name} ---")
        model = get_model(model_name)
        agent = Agent(model)
        
        for i, question in enumerate(questions, 1):
            print(f"Q{i}: {question[:50]}...")
            try:
                result = await agent.run(question)
                print(f"A{i}: {result.output[:100]}...")
            except Exception as e:
                print(f"Error: {e}")

async def main():
    print("Starting LLM Benchmark Suite...")
    
    # Test one model on all questions
    await benchmark_models_questions(['openai/gpt-4o-mini'], GENERAL_QUESTIONS)
    
    # Test all models on one question
    await benchmark_models_questions(MODELS, [GENERAL_QUESTIONS[1]])

if __name__ == "__main__":
    asyncio.run(main())