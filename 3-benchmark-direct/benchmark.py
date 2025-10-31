import asyncio
import dataclasses as dc
from pathlib import Path
import time
import textwrap

import dotenv

import async_google_prompt
import async_openai_prompts
import base_classes as bc
import metrics as mt
from questions import QuestionData, load_questions_from_jsonl

# Data classes:

@dc.dataclass(frozen=True)
class BenchmarkContext:
    timeout_seconds: int = 20           # Timeout for each question.
    delay_ms: int = 10                  # Delay between question calls.
    verbose: bool = True                # Verbose or quiet output?
    truncate_length: int = 150          # Display text truncation.
    max_parallel_questions: int = 30    # Limit parallel question execution.
    retry_failures: bool = True         # Retry failed requests?
    use_open_router: bool = True        # Use OpenRouter or direct calls?
    benchmark_n_times: int = 1          # Benchmark n times?
    reasoning_effort: str = "low"       # Reasoning effort.
    web_search: bool = False            # Use web search?
    context: str = ""                   # Context with documentation/examples.

    def __str__(self):
        return f'''BenchmarkContext:
        timeout_seconds: {self.timeout_seconds}
        delay_ms: {self.delay_ms}
        verbose: {self.verbose}
        truncate_length: {self.truncate_length}
        max_parallel_questions: {self.max_parallel_questions}
        retry_failures: {self.retry_failures}
        use_open_router: {self.use_open_router}
        benchmark_n_times: {self.benchmark_n_times}
        reasoning_effort: {self.reasoning_effort}
        web_search: {self.web_search}
        context: {display_text(self.context, self.truncate_length)}
        '''

    def with_changes(self, **kwargs):
        """Creates a new Context with specified changes"""
        return dc.replace(self, **kwargs)

# Utility functions:

def display_text(text: str, max_length: int) -> str:
    return text[:max_length] + "..." if len(text) > max_length else text

def load_txt_file(file_path: str) -> tuple[str, int]:
    """Load a text file, then return its content and approximate number of tokens."""
    p = Path(file_path)
    size_tokens = p.stat().st_size // 3 # Code usually uses 3 chars per token.
    txt = p.read_text(encoding="utf-8")
    return txt, size_tokens

# Benchmarking functions:

def get_model_agent(ctx: BenchmarkContext, model_info: bc.ModelInfo, run_index: int = 0) -> tuple[str, bc.LLMHandler]:
    """Get a model name and agent."""
    raise NotImplementedError("Not implemented")

async def get_question_task(ctx: BenchmarkContext, model: bc.ModelInfo, agent: bc.LLMHandler,
    question_num: int, question: QuestionData) -> mt.Metrics:
    """Run a single question and return performance metrics."""
    # Prepare question and prompt.
    question_text = f"{question.question}\n{question.masked_code}"
    full_prompt = f"{bc._DEFAULT_SYSTEM_PROMPT}\n\n{question_text}"

    if ctx.context:
        full_prompt += f"\n--- END OF QUESTION AND MASKED CODE ---\nBelow '--- DOCUMENTATION:' line is the documentation, which are all GemBox Software .NET components examples.\n--- DOCUMENTATION: \n{ctx.context}\n--- END OF DOCUMENTATION ---\n Answer the question based on the documentation, return only the JSON object with no explanations, comments, or additional text.\n"

    if ctx.verbose:
        print(f"Q{question_num}: {textwrap.shorten(question_text, ctx.truncate_length)}")
    
    try:
        # Run the async call and retry if needed.
        for attempt in range(1+ctx.retry_failures):  # First try + one retry.
            try:
                if ctx.delay_ms:
                    await asyncio.sleep(ctx.delay_ms / 1000)
                results, input_tokens, output_tokens = await asyncio.wait_for(agent.run(full_prompt), timeout=ctx.timeout_seconds)
                break
            except Exception as e:
                if attempt == 0:  # First failure.
                    print(f"Retrying because of exception: {repr(e)}")
                    continue
                raise
        
        # Calculate tokens, cost, and accuracy.
        cost = model.calculate_cost(input_tokens, output_tokens)
        accuracy = mt.get_accuracy(f"Q{question_num}", results, question.answers)
        
        # Display results.  
        if ctx.verbose:
            print(f"A{question_num}: {textwrap.shorten(str(results), ctx.truncate_length)}")
            if accuracy == 1.0:
                print("✓ CORRECT")
            elif accuracy > 0:
                print(f"✗ PARTIAL ({accuracy:.1%}), expected: {question.answers}")
            else:
                print(f"✗ INCORRECT, expected: {question.answers}")
        
        # Return metrics.
        return mt.Metrics(f"Q{question_num}", cost, input_tokens+output_tokens, 0, accuracy, 0, 1)

    except Exception as e:
        print(f"Error: {repr(e)}")
        return mt.Metrics("Error", 0.0, 0, 0, 0.0, 1, 1)

async def run_model_benchmark(ctx: BenchmarkContext, model_info: bc.ModelInfo, questions: list, run_index: int = 0) -> mt.Metrics:
    """Run benchmark for a single model on all questions in parallel."""
    # Initialize model and agent.
    try:
        model_name, agent = get_model_agent(ctx, model_info, run_index)
    except Exception as e:
        print(f"\n--- Can't get model agent: {repr(e)}")
        return mt.Metrics(f"ERROR-{model_info.name}", 0.0, 0, 0, 0.0, len(questions))
    
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
            if isinstance(r, mt.Metrics):
                task_metrics.append(r)
            elif isinstance(r, asyncio.CancelledError):
                raise r   # Let cancellation terminate everything.
            else:
                print(f"Unexpected task error: {repr(r)}")
                task_metrics.append(mt.Metrics("Error", 0.0, 0, 0, False, 0.0, 1))
    model_time = time.time() - model_start_time
    # Hack: last task metric has the model time, others have 0. That way summation works in summarize_metrics().
    task_metrics[-1].time = model_time
    
    # Summarize model metrics, print, and return data.
    sum_metrics = mt.summarize_metrics(model_name, task_metrics)
    if ctx.verbose:
        mt.print_metrics(model_name, task_metrics)
    else:
        mt.print_metrics(model_name, [sum_metrics])
    return sum_metrics

async def benchmark_models_n_times(name: str, ctx: BenchmarkContext, models: list[bc.ModelInfo], questions: list[QuestionData]) -> mt.Metrics:
    """Benchmark models N times."""
    print(f"\n===== Benchmarking {len(models)} model(s) on {len(questions)} question(s) {ctx.benchmark_n_times} times. =====\n")
    print(ctx)

    perf_data = {model.name: [] for model in models}
    for run_idx in range(ctx.benchmark_n_times):
        print(f"\n=== Run {run_idx + 1} of {ctx.benchmark_n_times} ===")
        for model in models:
            perf_data[model.name].append(
                await run_model_benchmark(ctx, model, questions, run_idx))

    # Flatten list of lists.
    all_metrics = [ 
        mt.summarize_metrics(model_name, model_metrics) 
        for model_name, model_metrics in perf_data.items()]

    if len(all_metrics) > 1: # If more than one measure, calculate averages.
        mt.print_metrics(f" SUMMARY OF: {name} ", all_metrics)
        return mt.summarize_metrics(name, all_metrics)
    else: # If only one measure, return it.
        return all_metrics[0]

# Async main.
async def main_test():
    # Load environment variables from parent directory .env.
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")

    # Load questions from JSONL file.
    questions = load_questions_from_jsonl("../2-bench-filter/test.jsonl")[:3]

    # Load documentation.
    context_txt, context_approx_tokens = load_txt_file("GemBox-Spreadsheet-examples.txt")
    print(f"Documentation 1 of ~length: {context_approx_tokens} tokens, starting with: {context_txt[:100]}")

    # Filter models.
    # models = bc.Models().by_tags(include={'prompt'})
    # models = bc.Models().by_min_context_length(context_approx_tokens).by_max_price(0.25, 2.0).by_tags(exclude={'prompt'})
    models = bc.Models().by_names(['gpt-5-nano']) # For testing web search.
    # models = bc.Models().by_names(['gpt-5-mini']) # 'gemini-2.5-flash' Good long context models.
    print(f"Filtered models ({len(models)}): {models}")
    
    # Create starting context.
    start_bench_ctx = BenchmarkContext(
        timeout_seconds=60, 
        delay_ms=50, 
        verbose=False, 
        truncate_length=150, 
        max_parallel_questions=30, 
        retry_failures=True, 
        use_open_router=False,
        benchmark_n_times=1, 
        reasoning_effort="medium", 
        web_search=False, 
        context="")

    # Benchmark models.
    perf_data = [
        await benchmark_models_n_times(
            f"WEB SEARCH: {web}",
            start_bench_ctx.with_changes(web_search=web),
            # f"{timeout}s, {reason} REASONING, {len(context) if context else 0} bytes of documentation", 
            # start_bench_ctx.with_changes(timeout_seconds=timeout, reasoning_effort=reason, context=context), 
            models, 
            questions)
        for web in [True]
        # for timeout, reason in [(30, "low"), (60, "medium"), (100, "high")]
        # for timeout, reason, context in [(40, "medium", context_txt)]
    ]

    # Print summary.
    mt.print_metrics("=== SUMMARY OF ALL TESTS ===", perf_data)

    # OLD CODE:
    # print_metrics("=== SUMMARY OF: TOTAL ===", [summarize_metrics("TOTAL", perf_data)])
    # if not dotenv.load_dotenv():
    #     raise FileNotFoundError(".env file not found or empty")

    # all_models = bc.Models()

    # all_models.print_by_tags()

    # model = all_models.by_name('gpt-5-mini')
    # handler = model.create_handler(system_prompt="Answer questions about geography.", web_search=False)
    # results, input_tokens, output_tokens = await handler.call("What is the capital of France?")
    # print(f"results: {results}, input_tokens: {input_tokens}, output_tokens: {output_tokens}")

    # model = all_models.by_name('gemini-2.5-flash')
    # handler = model.create_handler(system_prompt="Answer questions about mathematics.", web_search=False)
    # results, input_tokens, output_tokens = await handler.call("Calculate 3 * 100.")
    # print(f"results: {results}, input_tokens: {input_tokens}, output_tokens: {output_tokens}")
    
if __name__ == "__main__":
    asyncio.run(main_test())