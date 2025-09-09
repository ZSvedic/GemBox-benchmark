import dotenv
import asyncio
import time

from typing import List, Union
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from cached_agent_proxy import CachedAgentProxy

@dataclass
class Context:
    # Timing:
    timeout_seconds: int = 15
    delay_ms: int = 0
    # Display:
    verbose: bool = True
    truncate_length: int = 150  # For display text truncation
    # Calling:
    max_parallel_questions: int = None  # Limit parallel question execution
    max_retries: int = 0        # Number of retries for failed requests
    # Caching:
    use_caching: bool = False   # Whether to use CachedAgentProxy
    cache_dir: str = "cache"    # Directory for cache files
    # Model:
    use_open_router: bool = True

# Data classes for cleaner structure.
@dataclass
class ModelInfo:
    openrouter_name: str
    input_cost: float
    output_cost: float

@dataclass
class Metrics:
    cost: float
    tokens: int
    time: float
    speed: float
    accuracy: float = None
    was_cached: bool = False
    had_error: bool = False

# Data structure for JSONL questions
class QuestionData(BaseModel):
    category: str
    question: str
    masked_code: str
    answers: List[str]

# Pydantic model for structured output.
class CodeCompletion(BaseModel):
    completions: List[str]

PROGRAMMING_PROMPT = """Answer a coding question related to GemBox Software .NET components.
Return a JSON object with a 'completions' array containing only the code strings that should replace the ??? marks, in order. 
Completions array should not contain any extra whitespace as results will be used for string comparison.

Example question: 
How do you set the value of cell A1 to Hello?
worksheet.Cells[???].??? = ???;
Your response:
{"completions": ["A1", "Value", "Hello"]}

Return only the JSON object with no explanations, comments, or additional text."""

# Models with OpenRouter name, input costs, and output costs.
MODELS = {
    'gpt-4o-mini': ModelInfo('openai/gpt-4o-mini', 0.15, 0.60),
    # 'gpt-5-mini': ModelInfo('openai/gpt-5-mini', 0.25, 2.00),
    # 'gpt-5-nano': ModelInfo('openai/gpt-5-nano', 0.05, 0.40),
    # 'claude-3-haiku': ModelInfo('anthropic/claude-3-haiku', 0.25, 1.35),
    # 'gemini-2.5-flash': ModelInfo('google/gemini-2.5-flash', 0.30, 2.50),
    'gemini-2.5-flash-lite': ModelInfo('google/gemini-2.5-flash-lite', 0.10, 0.40),
    # 'codestral-2508': ModelInfo('mistralai/codestral-2508', 0.30, 0.90),
    # 'deepseek-chat-v3-0324': ModelInfo('deepseek-chat-v3-0324', 0.18, 0.72),
}

def load_questions_from_jsonl(file_path: str) -> List[QuestionData]:
    """Load questions from a JSONL file using Pydantic for automatic parsing and validation."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            questions = [
                QuestionData.model_validate_json(line.strip())
                for line in file
                if line.strip()
            ]
            print(f"Loaded {len(questions)} questions from {file_path}")
            return questions
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error loading questions: {e}")
        return []

# Utility functions.
def get_model_costs(model_info: ModelInfo) -> tuple[float, float]:
    return model_info.input_cost, model_info.output_cost

def calculate_cost(input_tokens: int, output_tokens: int, input_cost: float, output_cost: float) -> float:
    return (input_tokens / 1_000_000) * input_cost + (output_tokens / 1_000_000) * output_cost

def calculate_speed(total_tokens: int, duration: float) -> float:
    return total_tokens / duration if duration > 0 else 0

def display_text(text: str, max_length: int) -> str:
    return text[:max_length] + "..." if len(text) > max_length else text

def validate_answer(question: str, answer_text: str, result, expected_answers: List[str] = None) -> float:
    """Validate model response against expected answers and return accuracy."""
    assert expected_answers is not None
    
    try:
        # Calculate accuracy - now result.output is directly a list[str]
        if len(result.output) == len(expected_answers):
            correct_completions = sum(1 for m, e in zip(result.output, expected_answers) 
                                   if str(m).strip() == e.strip())
            return correct_completions / len(expected_answers)
        else:
            return 0.0
    except Exception as parse_error:
        print(f"  Warning: Could not parse model response for validation: {parse_error}")
        return 0.0

# Benchmarking functions.
async def run_single_question(
    context: Context,
    agent_code: Union[Agent, CachedAgentProxy],
    question: str, 
    question_num: int, 
    input_cost_per_1m: float, 
    output_cost_per_1m: float, 
    expected_answers: List[str] = None) -> Metrics:
    """Run a single question and return performance metrics."""
    if context.verbose:
        print(f"Q{question_num}: {display_text(question, context.truncate_length)}")
    
    try:
        # Prepare question and get response
        full_question = f"{question}\n\n{PROGRAMMING_PROMPT}"
        
        # Start timing immediately before the API call (pure model inference time)
        start_time = time.time()
        result = await asyncio.wait_for(agent_code.run(full_question), timeout=context.timeout_seconds)
        answer_text = str(result.output)
        
        # End timing immediately after the API call (pure model inference time)
        duration = time.time() - start_time
        
        # Extract token usage
        usage = result.usage() if callable(result.usage) else None
        input_tokens = getattr(usage, 'request_tokens', 0) if usage else int(len(full_question.split()) * 1.3)
        output_tokens = getattr(usage, 'response_tokens', 0) if usage else 0
        
        total_tokens = input_tokens + output_tokens
        cost = calculate_cost(input_tokens, output_tokens, input_cost_per_1m, output_cost_per_1m)
        speed = calculate_speed(total_tokens, duration)
        
        # Validate answer if expected answers are provided
        accuracy = validate_answer(question, answer_text, result, expected_answers)
        
        # Display results with model identifier for parallel execution clarity
        if context.verbose:
            print(f"A{question_num}: {display_text(answer_text, context.truncate_length)}")
            if accuracy is not None:
                if accuracy == 1.0:
                    status = "✓ CORRECT"
                    print(f"  {status}")
                elif accuracy > 0:
                    status = f"✗ PARTIAL ({accuracy:.1%})"
                    print(f"  {status}")
                else:
                    status = "✗ INCORRECT"
                    print(f"  {status}")
                
                if accuracy < 1.0 and expected_answers:
                    print(f"  Expected: {expected_answers}")
                
            print(f"  Tokens: {input_tokens} input + {output_tokens} output = {total_tokens} total")
            print(f"  Time: {duration:.2f}s (pure inference) | Speed: {speed:.1f} tokens/sec")
            print(f"  Cost: ${cost:.6f}")
        
        # Check if the result was cached
        was_cached = getattr(result, '_was_cached', False)
        
        return Metrics(cost, total_tokens, duration, speed, accuracy, was_cached, False)
        
    except Exception as e:
        print(f"Error: {repr(e)}")
        return Metrics(0, 0, 0, 0, None, False, True)

def calculate_model_accuracy(metrics: list[Metrics]) -> float:
    """Calculate overall accuracy for a model."""
    # Exclude questions with errors from accuracy calculation
    accuracy_scores = [m.accuracy for m in metrics if m.accuracy is not None and not m.had_error]
    return sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else None

def print_model_summary(model_name: str, metrics: list[Metrics]) -> dict:
    """Print summary for a single model."""
    total_cost = sum(m.cost for m in metrics)
    total_tokens = sum(m.tokens for m in metrics)
    total_time = sum(m.time for m in metrics)
    avg_speed = calculate_speed(total_tokens, total_time)
    error_count = sum(1 for m in metrics if m.had_error)
    total_calls = len(metrics)
    
    # Calculate accuracy
    overall_accuracy = calculate_model_accuracy(metrics)
    
    # Check if any responses were cached
    was_cached = any(m.was_cached for m in metrics)
    cache_status = " (CACHED)" if was_cached else ""
    display_cost = 0.0 if was_cached else total_cost
    
    if len(metrics) > 1:
        print(f"\n  Model Summary: {model_name}{cache_status}")
        print(f"  Total Cost: ${display_cost:.6f}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Speed: {avg_speed:.1f} tokens/sec")
        if overall_accuracy is not None:
            print(f"  Overall Accuracy: {overall_accuracy:.1%}")
        print(f"  Errors: {error_count} out of {total_calls} API calls")
    
    return {"model": model_name, "total_cost": display_cost, "total_tokens": total_tokens, "total_time": total_time, "avg_speed": avg_speed, "overall_accuracy": overall_accuracy, "errors": error_count}

def print_benchmark_summary(performance_data: list[dict], total_questions: int):
    """Print overall benchmark summary and rankings."""
    print("\n" + "="*60)
    print("OVERALL BENCHMARK SUMMARY")
    print("="*60)
    
    total_cost = sum(p['total_cost'] for p in performance_data)
    total_tokens = sum(p['total_tokens'] for p in performance_data)
    total_time = sum(p['total_time'] for p in performance_data)
    total_errors = sum(p.get('errors', 0) for p in performance_data)
    
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Overall Speed: {calculate_speed(total_tokens, total_time):.1f} tokens/sec")
    if total_errors:
        print(f"Total Errors: {total_errors} out of {total_questions} API calls")
    
    # Rankings
    print("\nCost Efficiency Ranking (lowest cost per token first):")
    efficiency = [(p['model'], p['total_cost'] / p['total_tokens'] if p['total_tokens'] > 0 else float('inf')) for p in performance_data]
    efficiency.sort(key=lambda x: x[1])
    
    for i, (model, cost_per_token) in enumerate(efficiency, 1):
        print(f"{i}. {model}: ${cost_per_token*1000:.6f} per 1K tokens")
    
    print("\nSpeed Ranking (fastest first):")
    speed_data = [(p['model'], p['avg_speed']) for p in performance_data]
    speed_data.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, speed) in enumerate(speed_data, 1):
        print(f"{i}. {model}: {speed:.1f} tokens/sec")
    
    print("\nAccuracy Ranking (highest accuracy first):")
    accuracy_data = [(p['model'], p.get('overall_accuracy', 0.0)) for p in performance_data if p.get('overall_accuracy') is not None]
    accuracy_data.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model, accuracy) in enumerate(accuracy_data, 1):
        print(f"{i}. {model}: {accuracy:.1%}")
    
    print("\nError Rate Ranking (lowest error rate first):")
    error_data = [(p['model'], p.get('errors', 0), total_questions) for p in performance_data]
    error_data.sort(key=lambda x: x[1] / x[2] if x[2] > 0 else 0)
    
    for i, (model, errors, total) in enumerate(error_data, 1):
        error_rate = errors / total if total > 0 else 0
        print(f"{i}. {model}: {errors}/{total} API calls ({error_rate:.1%})")

async def run_model_benchmark(context: Context, model_info: ModelInfo, questions: list) -> dict:
    """Run benchmark for a single model on all questions in parallel."""
    if context.use_open_router:
        model_name = model_info.openrouter_name
        print(f"\n--- Testing {model_name} ---")
        model = OpenAIModel(model_name, provider=OpenRouterProvider())
        agent_code = Agent(model, output_type=list[str])
    else:
        model_name = model_info.openrouter_name.split('/')[-1] # Extract model name from OpenRouter name.
        print(f"\n--- Testing {model_name} ---")
        agent_code = Agent(model_name, output_type=list[str])

    # Create model-specific cache file in cache folder.
    cache_file = f"{context.cache_dir}/cached_responses_{model_name.replace('/', '_')}.json"
    # with CachedAgentProxy(Agent(model), cache_file) as agent:
    input_cost_per_1m, output_cost_per_1m = get_model_costs(model_info)
    
    # Create tasks for all questions to run in parallel.    
    question_tasks = []
    for i, question in enumerate(questions, 1):
        # Handle both QuestionData objects and string questions.
        if hasattr(question, 'question') and hasattr(question, 'masked_code'):
            # QuestionData object
            question_text = f"{question.question}\n{question.masked_code}"
            expected_answers = question.answers
        else:
            # String question (fallback).
            question_text = str(question)
            expected_answers = None
        
        task = run_single_question(
            context,
            agent_code,
            question_text,
            i,
            input_cost_per_1m,
            output_cost_per_1m,
            expected_answers,
        )
        question_tasks.append(task)
    
    # Apply parallel limit if specified.
    if context.max_parallel_questions and context.max_parallel_questions < len(question_tasks):
        # Process in batches
        model_metrics = []
        for i in range(0, len(question_tasks), context.max_parallel_questions):
            batch = question_tasks[i:i + context.max_parallel_questions]
            batch_results = await asyncio.gather(*batch)
            model_metrics.extend(batch_results)
    else:
        # Run all questions for this model in parallel.
        model_metrics = await asyncio.gather(*question_tasks)

    # END with CachedAgentProxy(Agent(model)).
    
    # Print model summary and return data.
    model_data = print_model_summary(model_name, model_metrics)
    return model_data

async def main():
    # Load environment variables from parent directory .env.
    dotenv.load_dotenv() 
    
    # Load questions from JSONL file.
    jsonl_path = "../2-bench-filter/test.jsonl"
    questions = load_questions_from_jsonl(jsonl_path)
    assert questions is not None

    # Create context.
    context = Context() 
    context.verbose = False
    context.use_open_router = False

    # Define models to benchmark.
    models = [MODELS['gpt-4o-mini'], MODELS['gemini-2.5-flash-lite']]
    
    print(f"\nBenchmarking {len(models)} model(s) on {len(questions)} question(s) sequentially.")
    
    # Run models sequentially.
    performance_data = [await run_model_benchmark(context, model, questions) for model in models]
    
    # Print overall summary only if multiple models.
    if len(models) > 1 and context.verbose:
        print_benchmark_summary(performance_data, len(questions))

if __name__ == "__main__":
    asyncio.run(main())