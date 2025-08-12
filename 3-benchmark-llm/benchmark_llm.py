import dotenv
import asyncio
import time
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

# Models with input and output costs (per 1M tokens).
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

def get_model_costs(model_name: str):
    """Get input and output costs for a specific model."""
    for model, input_cost, output_cost in MODELS_WITH_COSTS:
        if model == model_name:
            return input_cost, output_cost
    return 0.0, 0.0

def calculate_cost(input_tokens: int, output_tokens: int, input_cost_per_1m: float, output_cost_per_1m: float):
    """Calculate the cost of an API call based on token usage."""
    input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
    return input_cost + output_cost

def calculate_speed(total_tokens: int, duration: float):
    """Calculate tokens per second."""
    if duration > 0:
        return total_tokens / duration
    return 0

def estimate_input_tokens(text: str):
    """Rough estimation of input tokens (words + punctuation)."""
    # Simple estimation: roughly 1.3 tokens per word
    words = len(text.split())
    return int(words * 1.3)

async def run_single_question(agent: Agent, question: str, question_num: int, input_cost_per_1m: float, output_cost_per_1m: float):
    """Run a single question and return performance metrics."""
    print(f"Q{question_num}: {question[:50]}...")
    
    try:
        start_time = time.time()
        result = await agent.run(question)
        duration = time.time() - start_time
        
        # Extract token usage
        usage = result.usage() if callable(result.usage) else None
        input_tokens = getattr(usage, 'request_tokens', 0) if usage else 0
        output_tokens = getattr(usage, 'response_tokens', 0) if usage else 0
        
        if input_tokens == 0:
            input_tokens = estimate_input_tokens(question)
        
        total_tokens = input_tokens + output_tokens
        cost = calculate_cost(input_tokens, output_tokens, input_cost_per_1m, output_cost_per_1m)
        speed = calculate_speed(total_tokens, duration)
        
        print(f"A{question_num}: {result.output[:100]}...")
        print(f"  Tokens: {input_tokens} input + {output_tokens} output = {total_tokens} total")
        print(f"  Time: {duration:.2f}s | Speed: {speed:.1f} tokens/sec")
        print(f"  Cost: ${cost:.6f}")
        
        return {
            'cost': cost,
            'tokens': total_tokens,
            'time': duration,
            'speed': speed
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {'cost': 0, 'tokens': 0, 'time': 0, 'speed': 0}

def print_model_summary(model_name: str, metrics: list):
    """Print summary for a single model."""
    total_cost = sum(m['cost'] for m in metrics)
    total_tokens = sum(m['tokens'] for m in metrics)
    total_time = sum(m['time'] for m in metrics)
    avg_speed = calculate_speed(total_tokens, total_time)
    
    print(f"\n  Model Summary: {model_name}")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Total Tokens: {total_tokens}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Average Speed: {avg_speed:.1f} tokens/sec")
    
    return {
        'model': model_name,
        'total_cost': total_cost,
        'total_tokens': total_tokens,
        'total_time': total_time,
        'avg_speed': avg_speed
    }

def print_benchmark_summary(performance_data: list):
    """Print overall benchmark summary and rankings."""
    print("\n" + "="*60)
    print("OVERALL BENCHMARK SUMMARY")
    print("="*60)
    
    total_cost = sum(p['total_cost'] for p in performance_data)
    total_tokens = sum(p['total_tokens'] for p in performance_data)
    total_time = sum(p['total_time'] for p in performance_data)
    
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Overall Speed: {calculate_speed(total_tokens, total_time):.1f} tokens/sec")
    
    # Cost efficiency ranking
    print("\nCost Efficiency Ranking (lowest cost per token first):")
    efficiency_data = [(p['model'], p['total_cost'] / p['total_tokens'] if p['total_tokens'] > 0 else float('inf')) for p in performance_data]
    efficiency_data.sort(key=lambda x: x[1])
    
    for i, (model, cost_per_token) in enumerate(efficiency_data, 1):
        print(f"{i}. {model}: ${cost_per_token*1000:.6f} per 1K tokens")
    
    # Speed ranking
    print("\nSpeed Ranking (fastest first):")
    speed_data = [(p['model'], p['avg_speed']) for p in performance_data]
    speed_data.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, speed) in enumerate(speed_data, 1):
        print(f"{i}. {model}: {speed:.1f} tokens/sec")

async def benchmark_models_questions(models: list[str], questions: list[str]):
    """Benchmark multiple models on multiple questions."""
    print(f"Benchmarking {len(models)} model(s) on {len(questions)} question(s).")
    
    performance_data = []
    
    for model_name in models:
        print(f"\n--- Testing {model_name} ---")
        agent = Agent(get_model(model_name))
        input_cost_per_1m, output_cost_per_1m = get_model_costs(model_name)
        
        # Run all questions for this model
        model_metrics = []
        for i, question in enumerate(questions, 1):
            metrics = await run_single_question(agent, question, i, input_cost_per_1m, output_cost_per_1m)
            model_metrics.append(metrics)
        
        # Print model summary and store data
        model_data = print_model_summary(model_name, model_metrics)
        performance_data.append(model_data)
    
    # Print overall summary
    print_benchmark_summary(performance_data)
    return performance_data

async def main():
    print("Starting LLM Benchmark Suite...")
    
    # Test one model on all questions
    await benchmark_models_questions(['openai/gpt-4o-mini'], GENERAL_QUESTIONS)
    
    # Test all models on one question
    await benchmark_models_questions(MODELS, [GENERAL_QUESTIONS[1]])

if __name__ == "__main__":
    asyncio.run(main())