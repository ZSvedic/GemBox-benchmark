import dotenv
import asyncio
import time
import dataclasses

from typing import List, Union
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from questions import QuestionData, load_questions_from_jsonl
from metrics import Metrics, get_accuracy, summarize_metrics, print_metrics
from cached_agent_proxy import CachedAgentProxy

# Data classes:

@dataclass(frozen=True)
class Context:
    timeout_seconds: int = 20           # Timeout for each question.
    delay_ms: int = 10                  # Delay between question calls.
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

def calculate_cost(input_tokens: int, output_tokens: int, model: ModelInfo) -> float:
    return (input_tokens / 1_000_000) * model.input_cost + (output_tokens / 1_000_000) * model.output_cost

def display_text(text: str, max_length: int) -> str:
    return text[:max_length] + "..." if len(text) > max_length else text

# Benchmarking functions:

def get_model_agent(ctx: Context, model_info: ModelInfo) -> str | Agent:
    """Get a model name and agent."""
    if ctx.use_open_router:
        model_name = model_info.openrouter_name + (":online" if ctx.web_search else "")
        model = OpenAIChatModel(
            model_name, 
            provider=OpenRouterProvider(),
            settings=OpenAIChatModelSettings(
                openai_reasoning_effort=ctx.reasoning_effort)
            )
    else:
        model_name = model_info.direct_name
        model = model_name

    if ctx.use_caching:
        agent_code = CachedAgentProxy(
            model, CodeCompletion, f"cache/responses_{model_name.replace('/', '_')}.json", ctx.verbose)
    else:
        agent_code = Agent(model_name, output_type=CodeCompletion)

    return model_name, agent_code

async def get_question_task(ctx: Context, model: ModelInfo, agent_code: Agent,
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
        accuracy = get_accuracy(f"Q{question_num}", result.output.completions, question.answers)
        
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
    # Hack: last task metric has the model time, others have 0. That way summation works in summarize_metrics().
    task_metrics[-1].time = model_time

    # Close CachedAgentProxy.
    if ctx.use_caching:
        agent.close()
    
    # Summarize model metrics and return data.
    if ctx.verbose:
        print_metrics(model_name, task_metrics)
    return summarize_metrics(model_name, task_metrics)

async def benchmark_models_n_times(name: str, ctx: Context, models: list[ModelInfo], questions: list[QuestionData]) -> Metrics:
    """Benchmark models N times."""
    print(f"\n===== Benchmarking {len(models)} model(s) on {len(questions)} question(s) {ctx.benchmark_n_times} times. =====\n")
    print(ctx)

    perf_data = {model.openrouter_name: [] for model in models}
    for run_idx in range(ctx.benchmark_n_times):
        print(f"\n=== Run {run_idx + 1} of {ctx.benchmark_n_times} ===")
        for model in models:
            perf_data[model.openrouter_name].append(
                await run_model_benchmark(ctx, model, questions))

    # Flatten list of lists.
    all_metrics = [ 
        summarize_metrics(model_name, model_metrics) 
        for model_name, model_metrics in perf_data.items()]

    if len(all_metrics) > 1: # If more than one measure, calculate averages.
        print_metrics(f" SUMMARY OF: {name} ", all_metrics)
        return summarize_metrics(name, all_metrics)
    else: # If only one measure, return it.
        return all_metrics[0]

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
    
    # Default context.
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

    # Benchmark reasoning effort.
    perf_data = [
        await benchmark_models_n_times(
            f"{timeout}s, {reason} REASONING", 
            default_context.with_changes(reasoning_effort=reason, timeout_seconds=timeout), 
            models, 
            questions)
        # for timeout, reason in [(30, "low"), (60, "medium"), (100, "high")]
        for timeout, reason in [(30, "low")]
    ]

    # # Benchmark web search.
    # perf_data = [
    #     await benchmark_models_n_times(
    #         f"WEB SEARCH: {web}",
    #         default_context.with_changes(web_search=web),
    #         models,
    #         questions)
    #     for web in [False, True]
    # ]

    # Print summary.
    print_metrics("=== SUMMARY OF ALL TESTS ===", perf_data)
    print_metrics("=== SUMMARY OF: TOTAL ===", [summarize_metrics("TOTAL", perf_data)])

if __name__ == "__main__":
    asyncio.run(main())