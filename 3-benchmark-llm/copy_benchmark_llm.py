import dotenv
import asyncio
import time
from dataclasses import dataclass
from typing import List
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

dotenv.load_dotenv()

# Data classes for cleaner structure.
@dataclass
class ModelInfo:
    name: str
    input_cost: float
    output_cost: float

@dataclass
class Metrics:
    cost: float
    tokens: int
    time: float
    speed: float

@dataclass
class ModelSummary:
    model: str
    total_cost: float
    total_tokens: int
    total_time: float
    avg_speed: float

# Pydantic model for structured output.
class CodeCompletion(BaseModel):
    completions: List[str]

# Questions and prompts.
GENERAL_QUESTIONS = [
    "What is the capital of France?",
    "Who wrote 'To Kill a Mockingbird'?",
    "Explain the theory of relativity in simple terms.",
]

PROGRAMMING_QUESTIONS = [
    "Complete this C# method to calculate the factorial of a number: public static int Factorial(int n) { if (n <= 1) return 1; return ???; }",
    "Fill in the missing parts of this C# LINQ query: var result = numbers.Where(x => ???).Select(x => ???).ToList();",
    "Complete this C# async method: public async Task<string> GetDataAsync() { var response = await httpClient.GetAsync(url); if (response.IsSuccessStatusCode) { var content = await ???; return ???; } return ???; }",
    "Complete this C# class constructor and method: public class Calculator { private int ???; public Calculator(int initialValue) { ??? = initialValue; } public int Add(int value) { return ??? + value; } }",
]

PROGRAMMING_PROMPT = "Return a JSON object with a 'completions' array containing only the code strings that should replace the ??? marks, in order. Do not include any explanations, comments, or additional text."

# Models with costs.
MODELS = [
    ModelInfo('openai/gpt-4o-mini', 0.15, 0.60),
    ModelInfo('openai/gpt-5-mini', 0.25, 2.00),
    ModelInfo('openai/gpt-5-nano', 0.05, 0.40),
    ModelInfo('anthropic/claude-3-haiku', 0.25, 1.35),
    ModelInfo('google/gemini-2.5-flash', 0.30, 2.50),
    ModelInfo('google/gemini-2.5-flash-lite', 0.10, 0.40),
    ModelInfo('mistralai/codestral-2508', 0.30, 0.90),
    ModelInfo('deepseek/deepseek-chat-v3-0324', 0.18, 0.72),
]

MODEL_NAMES = [model.name for model in MODELS]

# Utility functions.
def get_model(model_name: str):
    return OpenAIModel(model_name, provider=OpenRouterProvider())

def get_model_costs(model_name: str) -> tuple[float, float]:
    for model in MODELS:
        if model.name == model_name:
            return model.input_cost, model.output_cost
    return 0.0, 0.0

def calculate_cost(input_tokens: int, output_tokens: int, input_cost: float, output_cost: float) -> float:
    return (input_tokens / 1_000_000) * input_cost + (output_tokens / 1_000_000) * output_cost

def calculate_speed(total_tokens: int, duration: float) -> float:
    return total_tokens / duration if duration > 0 else 0

def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)

def display_text(text: str, max_length: int) -> str:
    return text[:max_length] + "..." if len(text) > max_length else text

# Benchmarking functions.
async def run_single_question(agent: Agent, question: str, question_num: int, input_cost_per_1m: float, output_cost_per_1m: float, truncate_length: int = 100) -> Metrics:
    """Run a single question and return performance metrics."""
    print(f"Q{question_num}: {display_text(question, truncate_length)}")
    
    try:
        start_time = time.time()
        
        # Prepare question and get response
        if "???" in question:
            full_question = f"{question}\n\n{PROGRAMMING_PROMPT}"
            result = await agent.run(full_question, output_type=CodeCompletion)
            answer_text = str(result.output.completions)
        else:
            full_question = question
            result = await agent.run(full_question)
            answer_text = result.output
            
        duration = time.time() - start_time
        
        # Extract token usage
        usage = result.usage() if callable(result.usage) else None
        input_tokens = getattr(usage, 'request_tokens', 0) if usage else 0
        output_tokens = getattr(usage, 'response_tokens', 0) if usage else 0
        
        if input_tokens == 0:
            input_tokens = estimate_tokens(full_question)
        
        total_tokens = input_tokens + output_tokens
        cost = calculate_cost(input_tokens, output_tokens, input_cost_per_1m, output_cost_per_1m)
        speed = calculate_speed(total_tokens, duration)
        
        # Display results
        print(f"A{question_num}: {display_text(answer_text, truncate_length)}")
        print(f"  Tokens: {input_tokens} input + {output_tokens} output = {total_tokens} total")
        print(f"  Time: {duration:.2f}s | Speed: {speed:.1f} tokens/sec")
        print(f"  Cost: ${cost:.6f}")
        
        return Metrics(cost, total_tokens, duration, speed)
        
    except Exception as e:
        print(f"Error: {e}")
        return Metrics(0, 0, 0, 0)

def print_model_summary(model_name: str, metrics: list[Metrics]) -> ModelSummary:
    """Print summary for a single model."""
    total_cost = sum(m.cost for m in metrics)
    total_tokens = sum(m.tokens for m in metrics)
    total_time = sum(m.time for m in metrics)
    avg_speed = calculate_speed(total_tokens, total_time)
    
    if len(metrics) > 1:
        print(f"\n  Model Summary: {model_name}")
        print(f"  Total Cost: ${total_cost:.6f}")
        print(f"  Total Tokens: {total_tokens}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Speed: {avg_speed:.1f} tokens/sec")
    
    return ModelSummary(model_name, total_cost, total_tokens, total_time, avg_speed)

def print_benchmark_summary(performance_data: list[ModelSummary]):
    """Print overall benchmark summary and rankings."""
    print("\n" + "="*60)
    print("OVERALL BENCHMARK SUMMARY")
    print("="*60)
    
    total_cost = sum(p.total_cost for p in performance_data)
    total_tokens = sum(p.total_tokens for p in performance_data)
    total_time = sum(p.total_time for p in performance_data)
    
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Overall Speed: {calculate_speed(total_tokens, total_time):.1f} tokens/sec")
    
    # Rankings
    print("\nCost Efficiency Ranking (lowest cost per token first):")
    efficiency = [(p.model, p.total_cost / p.total_tokens if p.total_tokens > 0 else float('inf')) for p in performance_data]
    efficiency.sort(key=lambda x: x[1])
    
    for i, (model, cost_per_token) in enumerate(efficiency, 1):
        print(f"{i}. {model}: ${cost_per_token*1000:.6f} per 1K tokens")
    
    print("\nSpeed Ranking (fastest first):")
    speed_data = [(p.model, p.avg_speed) for p in performance_data]
    speed_data.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, speed) in enumerate(speed_data, 1):
        print(f"{i}. {model}: {speed:.1f} tokens/sec")

async def benchmark_models_questions(models: list[str], questions: list[str], truncate_length: int = 150):
    """Benchmark multiple models on multiple questions."""
    print(f"\nBenchmarking {len(models)} model(s) on {len(questions)} question(s).")
    
    performance_data = []
    
    for model_name in models:
        print(f"\n--- Testing {model_name} ---")
        agent = Agent(get_model(model_name))
        input_cost_per_1m, output_cost_per_1m = get_model_costs(model_name)
        
        # Run all questions for this model
        model_metrics = []
        for i, question in enumerate(questions, 1):
            metrics = await run_single_question(agent, question, i, input_cost_per_1m, output_cost_per_1m, truncate_length)
            model_metrics.append(metrics)
        
        # Print model summary and store data
        model_data = print_model_summary(model_name, model_metrics)
        performance_data.append(model_data)
    
    # Print overall summary only if multiple models
    if len(models) > 1:
        print_benchmark_summary(performance_data)
    return performance_data

async def main():
    # Test one model on a programming question with multiple masked areas
    await benchmark_models_questions(['openai/gpt-4o-mini'], [PROGRAMMING_QUESTIONS[2]], truncate_length=500)
    
    # Test all models on one question
    await benchmark_models_questions(MODEL_NAMES, [GENERAL_QUESTIONS[1]])

if __name__ == "__main__":
    asyncio.run(main())