import dotenv
import asyncio
import time
import dataclasses
import httpx
import re
import textwrap

from typing import Protocol
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from questions import QuestionData, load_questions_from_jsonl
from models import ModelInfo, Models
from metrics import Metrics, get_accuracy, summarize_metrics, print_metrics
from cached_agent_proxy import CachedAgentProxy
from async_openai_prompts import OpenAIPromptAgent
from async_google_prompt import GeminiPromptAgent

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

    def with_changes(self, **kwargs):
        """Creates a new Context with specified changes"""
        return dataclasses.replace(self, **kwargs)

# Pydantic model for structured output.
class CodeCompletion(BaseModel):
    completions: list[str]

# Protocol for agents.
class AgentProtocol(Protocol):
    async def run(self, input: str) -> httpx.Response: ...
    def response_2_results_tokens(self, response: httpx.Response) -> tuple[list[str], int, int]: ...

# Methods run() and __init__() are inherited from Agent, but response_2_usage_results() is implemented.
class OpenRouterAgent(Agent, AgentProtocol):
    def response_2_results_tokens(self, response: httpx.Response) -> tuple[list[str], int, int]:
        return response.output.completions, response.usage().input_tokens, response.usage().output_tokens

PROGRAMMING_PROMPT = """Answer a coding question related to GemBox Software .NET components.
Return a JSON object with a 'completions' array containing only the code strings that should replace the ??? marks, in order. 
Completions array should not contain any extra whitespace as results will be used for string comparison.

Example question: 
How do you set the value of cell A1 to Hello?
worksheet.Cells[???].??? = ???;
Your response:
{"completions": ["A1", "Value", "Hello"]}

Below is the question and masked code. Return only the JSON object with no explanations, comments, or additional text. """

# Utility functions:


def display_text(text: str, max_length: int) -> str:
    return text[:max_length] + "..." if len(text) > max_length else text

# Benchmarking functions:

def get_model_agent(ctx: Context, model_info: ModelInfo, run_index: int = 0) -> tuple[str, AgentProtocol]:
    """Get a model name and agent."""
    if model_info.openrouter_name.startswith("openaiprompt"): # OpenAIPromptAgent.
        model_name = model_info.direct_name
        print(f"\n--- Creating OpenAIPromptAgent for {model_name} ---")
        agent_code = OpenAIPromptAgent(model_name, CodeCompletion)
    elif model_info.openrouter_name.startswith("googlevertexai"): # GeminiPromptAgent.
        model_name = model_info.direct_name
        print(f"\n--- Creating GeminiPromptAgent for {model_name} ---")
        agent_code = GeminiPromptAgent(model_name)
    else:
        if ctx.use_open_router: # OpenRouterAgent.
            model_name = model_info.openrouter_name + (":online" if ctx.web_search else "")
            model = OpenAIChatModel(
                model_name, 
                provider=OpenRouterProvider(),
                settings=OpenAIChatModelSettings(
                    openai_reasoning_effort=ctx.reasoning_effort)
                )
        else: # OpenRouterAgent but with direct name.
            model_name = model_info.direct_name
            model = model_name

        if ctx.use_caching:
            cache_name = f"cache/responses_{re.sub(r'[^A-Za-z0-9._-]', '_', model_name)}_run{run_index + 1}.json"
            print(f"\n--- Creating CachedAgentProxy for {model_name} (run {run_index + 1}) ---")
            agent_code = CachedAgentProxy(
                model, CodeCompletion, cache_name, ctx.verbose)
        else:
            print(f"\n--- Creating Agent for {model_name} ---")
            agent_code = OpenRouterAgent(model, output_type=CodeCompletion)

    return model_name, agent_code

async def get_question_task(ctx: Context, model: ModelInfo, agent: AgentProtocol,
    question_num: int, question: QuestionData) -> Metrics:
    """Run a single question and return performance metrics."""
    # Prepare question and prompt.
    question_text = f"{question.question}\n{question.masked_code}"
    full_prompt = f"{PROGRAMMING_PROMPT}\n\n{question_text}"

    if ctx.verbose:
        print(f"Q{question_num}: {textwrap.shorten(question_text, ctx.truncate_length)}")
    
    try:
        # Run the async call and retry if needed.
        for attempt in range(1+ctx.retry_failures):  # First try + one retry.
            try:
                if ctx.delay_ms and not ctx.use_caching: # Don't delay if caching.
                    await asyncio.sleep(ctx.delay_ms / 1000)
                response = await asyncio.wait_for(agent.run(full_prompt), timeout=ctx.timeout_seconds)
                break
            except Exception as e:
                if attempt == 0:  # First failure.
                    print(f"Retrying because of exception: {repr(e)}")
                    continue
                raise
        
        # Convert specific response to usage and results.
        results, input_tokens, output_tokens = agent.response_2_results_tokens(response)
        
        # Calculate tokens, cost, and accuracy.
        cost = model.calculate_cost(input_tokens, output_tokens)
        accuracy = get_accuracy(f"Q{question_num}", results, question.answers)
        
        # Display results.  
        if ctx.verbose:
            print(f"A{question_num}: {textwrap.shorten(str(results), ctx.truncate_length)}")
            if accuracy == 1.0:
                print("✓ CORRECT")
            elif accuracy > 0:
                print(f"✗ PARTIAL ({accuracy:.1%}), expected: {question.answers}")
            else:
                print(f"✗ INCORRECT, expected: {question.answers}")
        
        # Check if the result was cached.
        was_cached = getattr(response, '_was_cached', False)
        
        # Return metrics.
        return Metrics(f"Q{question_num}", cost, input_tokens+output_tokens, 0, was_cached, accuracy, 0, 1)

    except Exception as e:
        print(f"Error: {repr(e)}")
        return Metrics("Error", 0.0, 0, 0, False, 0.0, 1, 1)

async def run_model_benchmark(ctx: Context, model_info: ModelInfo, questions: list, run_index: int = 0) -> Metrics:
    """Run benchmark for a single model on all questions in parallel."""
    # Initialize model and agent.
    try:
        model_name, agent = get_model_agent(ctx, model_info, run_index)
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

    # If CachedAgentProxy, close it.
    if isinstance(agent, CachedAgentProxy):
        agent.close()
    
    # Summarize model metrics, print, and return data.
    sum_metrics = summarize_metrics(model_name, task_metrics)
    if ctx.verbose:
        print_metrics(model_name, task_metrics)
    else:
        print_metrics(model_name, [sum_metrics])
    return sum_metrics

async def benchmark_models_n_times(name: str, ctx: Context, models: list[ModelInfo], questions: list[QuestionData]) -> Metrics:
    """Benchmark models N times."""
    print(f"\n===== Benchmarking {len(models)} model(s) on {len(questions)} question(s) {ctx.benchmark_n_times} times. =====\n")
    print(ctx)

    perf_data = {model.openrouter_name: [] for model in models}
    for run_idx in range(ctx.benchmark_n_times):
        print(f"\n=== Run {run_idx + 1} of {ctx.benchmark_n_times} ===")
        for model in models:
            perf_data[model.openrouter_name].append(
                await run_model_benchmark(ctx, model, questions, run_idx))

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

    # Load questions from JSONL file.
    questions = load_questions_from_jsonl("../2-bench-filter/test.jsonl")

    # Filter models.
    models = Models().by_tags(exclude={'prompt'})
    # models = Models().by_names(['gpt-5-codex', 'mistral-large'])
    print(f"Filtered models ({len(models)}): {models}")
    
    # Create starting context.
    start_ctx = Context(
        timeout_seconds=30, 
        delay_ms=50, 
        verbose=False, 
        truncate_length=150, 
        max_parallel_questions=30, 
        retry_failures=True, 
        use_caching=False, 
        use_open_router=True,
        benchmark_n_times=1, 
        reasoning_effort="low", 
        web_search=False)

    # Benchmark models.
    perf_data = [
        await benchmark_models_n_times(
            # f"WEB SEARCH: {web}",
            # start_ctx.with_changes(web_search=web),
            f"{timeout}s, {reason} REASONING", 
            start_ctx.with_changes(reasoning_effort=reason, timeout_seconds=timeout), 
            models, 
            questions)
        # for web in [False, True]
        # for timeout, reason in [(30, "low"), (60, "medium"), (100, "high")]
        for timeout, reason in [(60, "medium")]
    ]

    # Print summary.
    print_metrics("=== SUMMARY OF ALL TESTS ===", perf_data)
    # print_metrics("=== SUMMARY OF: TOTAL ===", [summarize_metrics("TOTAL", perf_data)])

if __name__ == "__main__":
    asyncio.run(main())