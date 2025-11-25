import asyncio
import dataclasses as dc
import pathlib
import textwrap
import time
from collections.abc import Collection

import dotenv

import google_handler, openai_handler, openrouter_handler  # Required to populate bc.Models._MODEL_REGISTRY.
import base_classes as bc
import metrics as mt
import questions as qs
from tee_logging import logging_context

# Data classes:

@dc.dataclass(frozen=True)
class BenchmarkContext:
    timeout_sec: int = 20               # Timeout for each question.
    delay_sec: int = 0.010              # Delay between question calls.
    verbose: bool = True                # Verbose or quiet output?
    truncate_length: int = 150          # Display text truncation.
    max_parallel_questions: int = 30    # Limit parallel question execution.
    retry_failures: bool = True         # Retry failed requests?
    benchmark_n_times: int = 1          # Benchmark n times?
    reasoning_effort: str = "low"       # Reasoning effort.
    web_search: bool = False            # Use web search?
    context: str = ""                   # Context with documentation/examples.

    def __str__(self):
        return f'''BenchmarkContext:
        timeout_sec: {self.timeout_sec}
        delay_sec: {self.delay_sec}
        verbose: {self.verbose}
        truncate_length: {self.truncate_length}
        max_parallel_questions: {self.max_parallel_questions}
        retry_failures: {self.retry_failures}
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
    p = pathlib.Path(file_path)
    size_tokens = p.stat().st_size // 3 # Code usually uses 3 chars per token.
    txt = p.read_text(encoding="utf-8")
    return txt, size_tokens

# Benchmarking functions:

async def get_question_task(ctx: BenchmarkContext, model: bc.ModelInfo, agent: bc.LLMHandler,
    question_num: int, question: qs.QuestionData) -> mt.Metrics:
    """Run a single question and return performance metrics."""
    # Prepare question and prompt.
    question_text = f"{question.question}\n{question.masked_code}"
    full_prompt = f"{bc._DEFAULT_SYSTEM_PROMPT}\n\n{question_text}"

    if ctx.context:
        full_prompt += f"\n--- END OF QUESTION AND MASKED CODE ---\nBelow '--- DOCUMENTATION:' line is the documentation, which are all GemBox Software .NET components examples.\n--- DOCUMENTATION: \n{ctx.context}\n--- END OF DOCUMENTATION ---\n Answer the question based on the documentation, return only the JSON object with no explanations, comments, or additional text.\n"

    if ctx.verbose:
        print(f"Q{question_num}: {textwrap.shorten(question_text, ctx.truncate_length)}")
    
    try:
        delay, t0 = 0.0, time.perf_counter()
        # Run the async call and retry if needed.
        for attempt in range(1+ctx.retry_failures):  # First try + one retry.
            try:
                if ctx.delay_sec:
                    await asyncio.sleep(ctx.delay_sec)
                    delay += ctx.delay_sec
                response = await asyncio.wait_for(agent.call(full_prompt), timeout=ctx.timeout_sec)
                break
            except Exception as e:
                if attempt == 0:  # First failure.
                    print(f"Retrying because of exception: {repr(e)}")
                    continue
                raise
        
        dt = time.perf_counter() - t0 - delay
        str_list, links, (input_tokens, output_tokens) = response
        results = str_list.completions

        # Calculate tokens, cost, and accuracy.
        cost = model.calculate_cost(input_tokens, output_tokens)
        error_rate = mt.get_error_rate(f"Q{question_num}", results, question.answers)
        
        # Display results.  
        if ctx.verbose:
            print(f"A{question_num}: {textwrap.shorten(str(results), ctx.truncate_length)}")
            if error_rate == 0.0:
                print("✓ CORRECT")
            elif error_rate < 1.0:
                print(f"✗ PARTIAL (error:{error_rate:.1%}), expected:{question.answers}")
            else:
                print(f"✗ INCORRECT, expected: {question.answers}")
        
        # Return metrics.
        return mt.Metrics(
            name=f"Q{question_num}", 
            provider_name=agent.provider_name(),
            cost_mdn=cost, 
            tokens_mdn=input_tokens+output_tokens, 
            time_avg=dt, 
            error_rate_avg=error_rate, 
            api_issues=0, 
            api_calls=1 )

    except Exception as e:
        print(f"Error: {repr(e)}")
        return mt.Metrics.get_error(agent.provider_name()) 

async def run_model_benchmark(ctx: BenchmarkContext, model_info: bc.ModelInfo, questions: list, run_index: int = 0) -> mt.Metrics:
    """Run benchmark for a single model on all questions in parallel."""
    print(f"\n==={model_info.name}===")
    # Initialize model and agent.
    try:
        handler = model_info.create_handler(
            system_prompt=bc._DEFAULT_SYSTEM_PROMPT, 
            parse_type=bc.ListOfStrings,
            web_search=ctx.web_search, 
            verbose=False)
    except Exception as e:
        print(f"\n--- Can't get model handler: {repr(e)}")
        return mt.Metrics(
            name=f"ERROR-{model_info.name}", 
            provider_name=model_info.provider_name(),
            cost_mdn=0.0, 
            tokens_mdn=0, 
            time_avg=0.0, 
            error_rate_avg=1.0, 
            api_issues=len(questions), 
            api_calls=len(questions) )
    
    # Create tasks for all questions to run in parallel.
    question_tasks = [
        get_question_task(ctx, model_info, handler, qi, question) 
        for qi, question in enumerate(questions, 1)
    ]
    
    # Process in batches, with timing and error handling.
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
                task_metrics.append(mt.Metrics.get_error(model_info.provider_name()))
    
    # Summarize model metrics, print, and return data.
    sum_metrics = mt.summarize_metrics(model_info.name, task_metrics)
    if ctx.verbose:
        mt.print_metrics(task_metrics)
    else:
        mt.print_metrics([sum_metrics])

    # Close handler to clean up resources.
    await handler.close()

    return sum_metrics

async def benchmark_models_n_times(
        name: str, 
        ctx: BenchmarkContext, 
        models: Collection[bc.ModelInfo], 
        questions: list[qs.QuestionData] ) -> mt.Metrics:
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

    print(f"\n=== SUMMARY OF: {name} ===")
    mt.print_metrics(all_metrics, csv_format=True)
    
    return (mt.summarize_metrics(name, all_metrics) if len(all_metrics) > 1 
            else all_metrics[0])

# Main test functions.

async def main_test():
    print("\n===== benchmark.main_test() =====")
    
    # Load questions from JSONL file.
    questions = qs.load_questions_from_jsonl("../2-bench-filter/test.jsonl")
    print(f"Using {len(questions)} questions.")

    # Load documentation.
    context_txt, context_approx_tokens = load_txt_file("GemBox-Spreadsheet-examples.txt")
    print(f"Documentation 1 of ~length: {context_approx_tokens} tokens, starting with: {context_txt[:100]}")

    print(f"PROMPT:\n{bc._DEFAULT_SYSTEM_PROMPT}\n")

    # Filter models.
    models_all = (
        bc.Models()
        # .by_web_search(True)
        # .by_min_context_length(context_approx_tokens)
        # .by_tags(include={'openrouter'})
        .by_names(['anthropic/claude-opus-4.5', 'x-ai/grok-4.1-fast']) 
    )

    print(f"Filtered models ({len(models_all)}): {models_all}")
    
    # Create starting context.
    start_bench_ctx = BenchmarkContext(
        timeout_sec=60, 
        delay_sec=0.070, 
        verbose=True, 
        truncate_length=150, 
        max_parallel_questions=14, 
        retry_failures=True, 
        benchmark_n_times=1, 
        reasoning_effort="medium", 
        web_search=True, 
        context="")
    
    # Create testing contents.
    bench_contexts = [
        ('OpenRouter + low reasoning', False, 'low', 30, '', models_all),
        ('OpenRouter Web search + medium reasoning', True, 'medium', 60, '', models_all),
        # ('Context + medium reasoning', False, 'medium', 60, context_txt, models_all),
        ]
    
    # Benchmark models.
    perf_data = [
        await benchmark_models_n_times(
            text,
            start_bench_ctx.with_changes(
                web_search=web, 
                reasoning_effort=reasoning,
                timeout_sec=timeout,
                context=context),
            models, 
            questions)
        for text, web, reasoning, timeout, context, models in bench_contexts
        ]

    # Print summary.
    print(f"\n=== SUMMARY OF ALL TESTS ===")
    mt.print_metrics(perf_data, True)

if __name__ == "__main__":
    # Load environment variables from parent directory .env.
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")
    
    with logging_context():
        asyncio.run(main_test())