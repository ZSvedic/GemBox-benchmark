import dotenv
import asyncio
import time
import dataclasses

from pprint import pprint
from typing import List, Union
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from cached_agent_proxy import CachedAgentProxy

# Data classes for cleaner structure:

@dataclass(frozen=True)
class Context:
    timeout_seconds: int = 20           # Timeout for each question.
    delay_ms: int = 50                  # Delay between question calls.
    verbose: bool = True                # Verbose or quiet output?
    truncate_length: int = 150          # Display text truncation.
    max_parallel_questions: int = 30    # Limit parallel question execution.
    retry_failures: bool = True         # Retry failed requests?
    use_caching: bool = False           # Use CachedAgentProxy?
    use_open_router: bool = True        # Use OpenRouter or direct calls?
    benchmark_n_times: int = 1          # Benchmark n times?
    reasoning_effort: str = "low"       # Reasoning effort.
    web_search: bool = False            # Use web search?

    def __post_init__(self):
        assert self.benchmark_n_times==1 or not self.use_caching, "Caching only makes sense for single-run benchmarks."

    def with_changes(self, **kwargs):
        """Create a new Context with specified changes"""
        return dataclasses.replace(self, **kwargs)

@dataclass(frozen=True)
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
    was_cached: bool
    accuracy: float # None for complete failures.
    error_count: int = 0
    api_calls: int = 0

# Data structure for JSONL questions.
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
    ModelInfo('openai/gpt-5-mini', 'openai:gpt-5-mini', 0.25, 2.00), 
    ModelInfo('google/gemini-2.5-flash', 'google-gla:gemini-2.5-flash', 0.30, 2.50),
    ModelInfo('mistralai/codestral-2508', 'mistral:codestral-latest', 0.30, 0.90),
    ModelInfo('google/gemini-2.5-flash-lite', 'google-gla:gemini-2.5-flash-lite', 0.10, 0.40), 
    ModelInfo('openai/gpt-5-nano', 'openai:gpt-5-nano', 0.05, 0.40), # Low accuracy.
    ModelInfo('anthropic/claude-3-haiku', 'anthropic:claude-3-5-haiku-latest', 0.25, 1.35), # Low accuracy.
    ModelInfo('openai/gpt-4o-mini', 'openai:gpt-4o-mini', 0.15, 0.60), # Low accuracy.
]

MODELS = {m.openrouter_name.split('/')[-1]: m for m in ModelInfos}
PRIMARY_MODELS = [
    MODELS['gpt-5-mini'], 
    MODELS['gemini-2.5-flash'], 
    MODELS['codestral-2508'],  
]

# Utility functions:

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

def calculate_model_accuracy(metrics: list[Metrics]) -> Union[float, None]:
    """Calculate overall accuracy for a model."""
    accuracy_scores = [m.accuracy for m in metrics if m.accuracy is not None] # Exclude errors.
    return (
        sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores 
        else None)

# Summary functions:

def get_model_summary(model_name: str, metrics: list[Metrics], verbose: bool) -> Metrics:
    """Print summary for a single model."""
    assert len(metrics) > 0, "No metrics to get summary for."
    # Calculate totals.
    total_tokens = sum(m.tokens for m in metrics)
    total_cost = sum(m.cost for m in metrics)
    total_time = sum(m.time for m in metrics)
    total_errors = sum(m.error_count for m in metrics)
    total_calls = sum(m.api_calls for m in metrics)
    total_accuracy = calculate_model_accuracy(metrics)
    # Check if any responses were cached
    was_cached = any(m.was_cached for m in metrics)
    cache_status = " (CACHED)" if was_cached else ""
    display_cost = 0.0 if was_cached else total_cost
    # Print summary.
    if verbose:
        print(f"\n  Model Summary: {model_name}{cache_status}")
        print(f"  Total Tokens: {total_tokens}")
        print(f"  Total Cost: ${display_cost:.6f}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Overall Accuracy: {total_accuracy:.1%}")
        print(f"  Errors: {total_errors} out of {total_calls} API calls")
    # Return metrics.
    return Metrics(
        name=model_name,
        cost=display_cost,
        tokens=total_tokens,
        time=total_time,
        was_cached=was_cached,
        accuracy=total_accuracy,
        error_count=total_errors,
        api_calls=total_calls
    )

def print_benchmark_summary(model_metrics: dict[str, list[Metrics]], n_questions: int, verbose: bool) -> list[Metrics]:
    """Print overall benchmark summary and rankings."""
    print("\n=== OVERALL SUMMARY ===")
    # Flatten and average metrics per model.
    total_metrics = [
        get_model_summary(model, metrics, verbose) 
        for model, metrics in model_metrics.items()]
    # Calculate totals.
    total_cost = sum(m.cost for m in total_metrics)
    total_time = sum(m.time for m in total_metrics)
    total_errors = sum(m.error_count for m in total_metrics)
    total_calls = sum(m.api_calls for m in total_metrics)
    # Print totals.
    print(f"\nTotal Cost: ${total_cost:.6f}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Total Errors: {total_errors} out of {total_calls} API calls = {total_errors/total_calls:.1%} error rate")
    # Error Rate Ranking
    if verbose and total_errors > 0:
        print("\nError Rate Ranking (lowest error rate first):")
        error_data = [(m.name, m.error_count/n_questions) for m in total_metrics]
        error_data.sort(key=lambda x: x[1])
        for i, (model, error_rate) in enumerate(error_data, 1):
            print(f"{i}. {model}: {error_rate:.1%}")
    # Accuracy Ranking
    print("\nAccuracy Ranking (highest accuracy first):")
    accuracy_data = [(m.name, m.accuracy) for m in total_metrics if m.accuracy is not None]
    accuracy_data.sort(key=lambda x: x[1], reverse=True)
    for i, (model, accuracy) in enumerate(accuracy_data, 1):
        print(f"{i}. {model}: {accuracy:.1%}")

    # Return total metrics.
    return total_metrics

# Benchmarking functions:

def get_model_agent(ctx: Context, model_info: ModelInfo) -> str | Union[Agent, CachedAgentProxy]:
    """Get a model name and agent."""
    if ctx.use_open_router:
        model_name = model_info.openrouter_name + (":online" if ctx.web_search else "")
        model = OpenAIChatModel(
            model_name, 
            provider=OpenRouterProvider(),
            settings=OpenAIChatModelSettings(
                openai_reasoning_effort=ctx.reasoning_effort)
            )
        agent_code = Agent(model, output_type=CodeCompletion)
    else:
        model_name = model_info.direct_name
        agent_code = Agent(model_name, output_type=CodeCompletion)
    return model_name, agent_code

async def get_question_task(ctx: Context, model: ModelInfo, agent_code: Union[Agent, CachedAgentProxy],
    question_num: int, question: QuestionData) -> Metrics:
    """Run a single question and return performance metrics."""
    # Prepare question and prompt.
    question_text = f"{question.question}\n{question.masked_code}"
    full_prompt = f"{PROGRAMMING_PROMPT}\n\n{question_text}"

    if ctx.verbose:
        print(f"Q{question_num}: {display_text(question_text, ctx.truncate_length)}")
    
    try:
        # Run the async call and retry if needed.
        for attempt in range(1+ctx.retry_failures):  # First try + one retry.
            try:
                if ctx.delay_ms and not ctx.use_caching: # Don't delay if caching.
                    await asyncio.sleep(ctx.delay_ms / 1000)
                result = await asyncio.wait_for(agent_code.run(full_prompt), timeout=ctx.timeout_seconds)
                break
            except Exception as e:
                if attempt == 0:  # First failure.
                    print(f"Retrying because of exception: {repr(e)}")
                    continue
                raise
        
        # Calculate tokens, cost, and accuracy.
        usage = result.usage()
        total_tokens = usage.input_tokens + usage.output_tokens
        cost = calculate_cost(usage.input_tokens, usage.output_tokens, model)
        accuracy = calculate_accuracy(question_num, result.output.completions, question.answers)
        
        # Display results.  
        if ctx.verbose:
            print(f"A{question_num}: {display_text(str(result.output.completions), ctx.truncate_length)}")
            if accuracy == 1.0:
                print("✓ CORRECT")
            elif accuracy > 0:
                print(f"✗ PARTIAL ({accuracy:.1%}), expected: {question.answers}")
            else:
                print(f"✗ INCORRECT, expected: {question.answers}")
        
        # Check if the result was cached.
        was_cached = getattr(result, '_was_cached', False)
        
        # Return metrics.
        return Metrics(f"Q{question_num}", cost, total_tokens, 0, was_cached, accuracy, 0, 1)

    except Exception as e:
        print(f"Error: {repr(e)}")
        return Metrics("Error", 0.0, 0, 0, False, 0.0, 1, 1)

async def run_model_benchmark(ctx: Context, model_info: ModelInfo, questions: list) -> Metrics:
    """Run benchmark for a single model on all questions in parallel."""
    # Initialize model and agent.
    try:
        model_name, agent = get_model_agent(ctx, model_info)
        print(f"\n--- Testing {model_name} ---")
    except Exception as e:
        print(f"\n--- Can't get model agent: {repr(e)}")
        return Metrics(f"ERROR-{model_info.openrouter_name}", 0.0, 0, 0, False, 0.0, len(questions))
    
    # Create model-specific cache file in cache folder.
    if ctx.use_caching:
        cache_file = f"cache/responses_{model_name.replace('/', '_')}.json"
        agent = CachedAgentProxy(agent, cache_file, ctx.verbose)
    
    # Create tasks for all questions to run in parallel.
    question_tasks = [
        get_question_task(ctx, model_info, agent, qi, question) 
        for qi, question in enumerate(questions, 1)
    ]
    
    # Process in batches, with timing and error handling.
    model_start_time = time.time()
    task_metrics = []
    for i in range(0, len(question_tasks), ctx.max_parallel_questions):
        batch = question_tasks[i:i + ctx.max_parallel_questions]
        batch_results = await asyncio.gather(*batch, return_exceptions=True)
        for r in batch_results:
            if isinstance(r, Metrics):
                task_metrics.append(r)
            elif isinstance(r, asyncio.CancelledError):
                raise r   # Let cancellation terminate everything.
            else:
                print(f"Unexpected task error: {repr(r)}")
                task_metrics.append(Metrics("Error", 0.0, 0, 0, False, 0.0, 1))
    model_time = time.time() - model_start_time
    # Hack: last task metric has the model time, others have 0. That way summation works in print_model_summary().
    task_metrics[-1].time = model_time

    # Close CachedAgentProxy.
    if ctx.use_caching:
        agent.close()
    
    # Print model summary and return data.
    model_data = get_model_summary(model_name, task_metrics, True)
    return model_data

async def benchmark_models_n_times(ctx: Context, models: list[ModelInfo], questions: list[QuestionData]) -> Metrics:
    """Benchmark models N times."""
    print(f"\n===== Benchmarking {len(models)} model(s) on {len(questions)} question(s) {ctx.benchmark_n_times} times. =====\n")
    pprint(ctx)

    perf_data = {model.openrouter_name: [] for model in models}
    for run_idx in range(ctx.benchmark_n_times):
        print(f"\n=== Run {run_idx + 1} of {ctx.benchmark_n_times} ===")
        for model in models:
            perf_data[model.openrouter_name].append(
                await run_model_benchmark(ctx, model, questions))

    all_metrics = [
        m for model_metrics in perf_data.values() 
        for m in model_metrics ]
    if len(all_metrics) > 1: # If more than one measure, calculate averages.
        return get_model_summary(str(ctx), all_metrics, True)
        # return print_benchmark_summary(perf_data, len(questions), True)
    elif len(all_metrics) == 1: # If only one measure, return it.
        return all_metrics[0]
    else: # Should never happen.
        raise ValueError(f"Invalid number of measurements: {len(all_metrics)}")

# Async main.
async def main():
    # Load environment variables from parent directory .env.
    dotenv.load_dotenv()
    assert dotenv.dotenv_values().values(), ".env file not found or empty"

    # Define models to benchmark.
    models = [ MODELS['gpt-5-mini'], MODELS['gemini-2.5-flash'] ]
    # models = PRIMARY_MODELS
    # models = MODELS.values()

    # Load questions from JSONL file.
    jsonl_path = "../2-bench-filter/test.jsonl"
    questions = load_questions_from_jsonl(jsonl_path)
    questions = questions[:5]
    
    # Create contexts.
    default_context = Context(
        timeout_seconds=30, 
        delay_ms=10, 
        verbose=False, 
        truncate_length=150, 
        max_parallel_questions=30, 
        retry_failures=True, 
        use_caching=True, 
        use_open_router=True,
        benchmark_n_times=1, 
        reasoning_effort="low", 
        web_search=False)

    contexts = [
        default_context.with_changes(reasoning_effort=reason, timeout_seconds=timeout) 
        for timeout, reason in [(30, "low"), (60, "medium")]
        # for timeout, reason in [(30, "low"), (60, "medium"), (90, "high")]
    ]

    # Benchmark each context.
    perf_data = [
        await benchmark_models_n_times(ctx, models, questions)
        for ctx in contexts
    ]
    # Then print benchmark summaries again.
    for pd in perf_data:
        pprint(pd)

    perf_dict = {ctx.reasoning_effort: [pd] for ctx, pd in zip(contexts, perf_data)}
    print_benchmark_summary(perf_dict, len(questions), True)


if __name__ == "__main__":
    asyncio.run(main())