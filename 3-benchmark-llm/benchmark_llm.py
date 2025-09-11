import dotenv
import asyncio
import time

from pprint import pprint
from typing import List, Union
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from cached_agent_proxy import CachedAgentProxy

# Data classes for cleaner structure.
@dataclass
class Context:
    # Timing:
    timeout_seconds: int = 15
    delay_ms: int = 0
    # Display:
    verbose: bool = True
    truncate_length: int = 150          # Display text truncation.
    # Calling:
    max_parallel_questions: int = 50    # Limit parallel question execution.
    max_retries: int = 0                # Number of retries for failed requests.
    # Caching:
    use_caching: bool = False           # Whether to use CachedAgentProxy.
    # Model:
    use_open_router: bool = True        # Whether to use OpenRouter or direct calls.

@dataclass
class ModelInfo:
    openrouter_name: str
    direct_name: str
    input_cost: float
    output_cost: float

@dataclass
class Metrics:
    name: str
    cost: float
    tokens: int
    time: float
    was_cached: bool = False
    accuracy: float = None # None for complete failures.
    error_count: int = 0

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

Below is the question and masked code. Return only the JSON object with no explanations, comments, or additional text. """

# Models with OpenRouter name, direct name, input costs, and output costs.
ModelInfos = [
    ModelInfo('openai/gpt-4o-mini', 'openai:gpt-4o-mini', 0.15, 0.60),
    ModelInfo('openai/gpt-5-mini', 'openai:gpt-5-mini', 0.25, 2.00),
    ModelInfo('openai/gpt-5-nano', 'openai:gpt-5-nano', 0.05, 0.40),
    ModelInfo('anthropic/claude-3-haiku', 'anthropic:claude-3-5-haiku-latest', 0.25, 1.35),
    ModelInfo('google/gemini-2.5-flash', 'google-gla:gemini-2.5-flash', 0.30, 2.50),
    ModelInfo('google/gemini-2.5-flash-lite', 'google-gla:gemini-2.5-flash-lite', 0.10, 0.40),
    ModelInfo('mistralai/codestral-2508', 'mistral:codestral-latest', 0.30, 0.90),
    ModelInfo('deepseek/deepseek-chat-v3-0324', 'deepseek-chat-v3-0324', 0.18, 0.72),
]
MODELS = {m.openrouter_name.split('/')[-1]: m for m in ModelInfos}

def load_questions_from_jsonl(file_path: str) -> list[QuestionData]:
    """Load questions from a JSONL file using Pydantic for automatic parsing and validation."""
    with open(file_path, 'r', encoding='utf-8') as file:
        questions = [
            QuestionData.model_validate_json(line.strip())
            for line in file
            if line.strip()
        ]
        print(f"Loaded {len(questions)} questions from {file_path}")
        return questions

# Utility functions.
def calculate_cost(input_tokens: int, output_tokens: int, model: ModelInfo) -> float:
    return (input_tokens / 1_000_000) * model.input_cost + (output_tokens / 1_000_000) * model.output_cost

def calculate_speed(total_tokens: int, duration: float) -> float:
    return total_tokens / duration if duration > 0 else 0

def display_text(text: str, max_length: int) -> str:
    return text[:max_length] + "..." if len(text) > max_length else text

def calculate_accuracy(question_num: int, results: list[str], expected_answers: List[str] = None) -> float:
    """Validate model response against expected answers and return accuracy."""
    max_length = max(len(results), len(expected_answers))
    try:
        correct_completions = sum(1 for res, exp in zip(results, expected_answers) 
                                if str(res).strip() == exp.strip())
        return correct_completions / max_length
    except Exception as parse_ex:
        print(f"  Warning: Q{question_num} failed with exception: {repr(parse_ex)}")
        return 0.0

def calculate_model_accuracy(metrics: list[Metrics]) -> float:
    """Calculate overall accuracy for a model."""
    # Exclude questions with errors from accuracy calculation
    accuracy_scores = [m.accuracy for m in metrics if m.accuracy is not None]
    return sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else None

# Benchmarking functions.
async def get_question_task(context: Context, model: ModelInfo, agent_code: Union[Agent, CachedAgentProxy],
    question_num: int, question: str, expected_answers: List[str] = None) -> Metrics:
    """Run a single question and return performance metrics."""
    if context.verbose:
        print(f"Q{question_num}: {display_text(question, context.truncate_length)}")
    
    try:
        # Prepare question and get response
        full_question = f"{PROGRAMMING_PROMPT}\n\n{question}"
        start_time = time.time()
        result = await asyncio.wait_for(agent_code.run(full_question), timeout=context.timeout_seconds)
        duration = time.time() - start_time
        # Calculate tokens and cost.
        usage = result.usage()
        total_tokens = usage.input_tokens + usage.output_tokens
        cost = calculate_cost(usage.input_tokens, usage.output_tokens, model)
        # Calculate accuracy.
        accuracy = calculate_accuracy(question_num, result.output.completions, expected_answers)
        # Display results.  
        if context.verbose:
            print(f"A{question_num}: {display_text(str(result.output.completions), context.truncate_length)}")
            if accuracy == 1.0:
                print("✓ CORRECT")
            elif accuracy > 0:
                print(f"✗ PARTIAL ({accuracy:.1%}), expected: {expected_answers}")
            else:
                print(f"✗ INCORRECT, expected: {expected_answers}")
        # Check if the result was cached.
        was_cached = getattr(result, '_was_cached', False)
        # Return metrics.
        return Metrics(f"Q{question_num}", cost, total_tokens, duration, was_cached, accuracy, 0)

    except asyncio.CancelledError as e:
        # Treat cancellations as a handled outcome so they don't bubble to gather/main.
        if context.verbose:
            print(f"Q{question_num}: cancelled: {repr(e)}")
        return Metrics(f"Q{question_num}", 0.0, 0, 0, False, None, 1)

    except Exception as e:
        print(f"Error: {repr(e)}")
        return Metrics("Error", 0.0, 0, 0, False, 0.0, 1)

def print_model_summary(model_name: str, metrics: list[Metrics]) -> Metrics:
    """Print summary for a single model."""
    assert len(metrics) > 0, "No metrics to print summary for."
    # Calculate totals.
    total_cost = sum(m.cost for m in metrics)
    total_tokens = sum(m.tokens for m in metrics)
    total_time = sum(m.time for m in metrics)
    avg_speed = calculate_speed(total_tokens, total_time)
    error_count = sum(m.error_count for m in metrics)
    total_calls = len(metrics)
    total_accuracy = calculate_model_accuracy(metrics)
    # Check if any responses were cached
    was_cached = any(m.was_cached for m in metrics)
    cache_status = " (CACHED)" if was_cached else ""
    display_cost = 0.0 if was_cached else total_cost
    # Print summary.
    print(f"\n  Model Summary: {model_name}{cache_status}")
    print(f"  Total Cost: ${display_cost:.6f}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Average Speed: {avg_speed:.1f} tokens/sec")
    print(f"  Overall Accuracy: {total_accuracy:.1%}")
    print(f"  Errors: {error_count} out of {total_calls} API calls")
    # Return metrics.
    return Metrics(model_name, display_cost, total_tokens, total_time, was_cached, total_accuracy, error_count)

def print_benchmark_summary(metrics: list[Metrics], total_questions: int):
    """Print overall benchmark summary and rankings."""
    print("\n=== OVERALL BENCHMARK SUMMARY ===")
    # Calculate totals.
    total_cost = sum(m.cost for m in metrics)
    total_time = sum(m.time for m in metrics)
    total_errors = sum(m.error_count for m in metrics)
    # Print totals.
    print(f"Total Cost: ${total_cost:.6f}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Errors: {total_errors} out of {total_questions*len(metrics)} API calls")
    # Accuracy Ranking
    print("\nAccuracy Ranking (highest accuracy first):")
    accuracy_data = [(m.name, m.accuracy) for m in metrics if m.accuracy is not None]
    accuracy_data.sort(key=lambda x: x[1], reverse=True)
    for i, (model, accuracy) in enumerate(accuracy_data, 1):
        print(f"{i}. {model}: {accuracy:.1%}")
    # Error Rate Ranking
    if total_errors > 0:
        print("\nError Rate Ranking (lowest error rate first):")
        error_data = [(m.name, float(m.error_count)/total_questions) for m in metrics]
        error_data.sort(key=lambda x: x[1])
        for i, (model, error_rate) in enumerate(error_data, 1):
            print(f"{i}. {model}: {error_rate:.1%}")

def get_model_agent(context: Context, model_info: ModelInfo) -> (str, Union[Agent, CachedAgentProxy]):
    """Get a model name and agent."""
    if context.use_open_router:
        model_name = model_info.openrouter_name
        model = OpenAIChatModel(model_name, provider=OpenRouterProvider())
        agent_code = Agent(model, output_type=CodeCompletion)
    else:
        model_name = model_info.direct_name
        agent_code = Agent(model_name, output_type=CodeCompletion)
    return model_name, agent_code

async def run_model_benchmark(context: Context, model_info: ModelInfo, questions: list) -> Metrics:
    """Run benchmark for a single model on all questions in parallel."""
    # Initialize model and agent.
    try:
        model_name, agent = get_model_agent(context, model_info)
    except Exception as e:
        print(f"\n--- Can't get model agent: {repr(e)}")
        return Metrics(f"ERROR-{model_info.openrouter_name}", 0.0, 0, 0, False, 0.0, len(questions))
    
    print(f"\n--- Testing {model_name} ---")

    # Create model-specific cache file in cache folder.
    if context.use_caching:
        cache_file = f"cache/cached_responses_{model_name.replace('/', '_')}.json"
        agent = CachedAgentProxy(agent, cache_file, context.verbose)
    
    # Create tasks for all questions to run in parallel.
    question_tasks = [
        get_question_task(
            context, model_info, agent, 
            qi, f"{question.question}\n{question.masked_code}", question.answers)
        for qi, question in enumerate(questions, 1)
    ]
    
    # Process in batches
    model_metrics = []
    for i in range(0, len(question_tasks), context.max_parallel_questions):
        batch = question_tasks[i:i + context.max_parallel_questions]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        for r in batch_results:
            if isinstance(r, Metrics):
                model_metrics.extend([r])
            else:
                # Normalize unexpected exceptions so they don't crash the run.
                print(f"Unexpected task error: {repr(r)}")
                model_metrics.extend([Metrics("Error", 0.0, 0, 0, False, 0.0, 1)])

    # Close CachedAgentProxy.
    if context.use_caching:
        agent.close()
    
    # Print model summary and return data.
    model_data = print_model_summary(model_name, model_metrics)
    return model_data

async def main():
    # Load environment variables from parent directory .env.
    dotenv.load_dotenv() 

    # Define models to benchmark.
    # models = [
    #     MODELS['gpt-4o-mini'], 
    #     MODELS['gemini-2.5-flash-lite']
    #     ]
    models = MODELS.values()

    # Load questions from JSONL file.
    jsonl_path = "../2-bench-filter/test.jsonl"
    questions = load_questions_from_jsonl(jsonl_path)
    questions = questions[:5]
    
    print(f"\nBenchmarking {len(models)} model(s) on {len(questions)} question(s) sequentially.\n")

    # Create context.
    context = Context(
        timeout_seconds=30, 
        delay_ms=0, 
        verbose=False, 
        truncate_length=150, 
        max_parallel_questions=50, 
        max_retries=0, 
        use_caching=True, 
        use_open_router=True)
    pprint(context)

    # Run models sequentially.
    performance_data = [
        await run_model_benchmark(context, model, questions) 
        for model in models]

    # Print overall summary only if multiple models.
    if len(models) > 1:
        print_benchmark_summary(performance_data, len(questions))

if __name__ == "__main__":
    asyncio.run(main())