# Python stdlib.
import asyncio
import dataclasses as dc
import pathlib
import textwrap as tw
import time

# Third-party.
import dotenv

# Local modules.
import google_handler, openai_handler, openrouter_handler  # Required to populate bc.Models._MODEL_REGISTRY.
import base_classes as bc
import metrics as mt
import questions as qs

# Constants.

TRUNCATE_LENGTH: int = 150  

# Data classes.

@dc.dataclass(frozen=True)  
class BenchmarkContext:
    description: str = '?'                      # Description of the benchmark context.
    models: tuple[bc.ModelInfo] = ()            # Models to benchmark.
    reasoning: str = 'low'                      # Reasoning effort.
    web: bool = False                           # Use web search?
    include_domains: str = ''                   # Domains to include in web search.
    bench_n_times: int = 1                      # Benchmark n times, for better precision.
    timeout_sec: int = 30                       # Timeout for each question.
    delay_sec: int = 0.050                      # Delay between calls, to avoid rate limits.
    verbose: bool = True                        # Verbose output?
    max_parallel_questions: int = 14            # Limit parallel question execution.
    retry_failures: bool = True                 # Retry failed requests?
    system_ins: str = bc.DEFAULT_SYSTEM_INS    # System instructons.
    system_doc: str = ''                        # System documentation/examples.
    questions: tuple[qs.QuestionData] = ()      # Questions to benchmark.

    def __str__(self):
        '''Indented multi-line string representation of the dataclass.'''
        return type(self).__name__ + ":\n" + \
            "\n".join(f"\t{f.name}={tw.shorten(str(getattr(self, f.name)), TRUNCATE_LENGTH)}" 
                      for f in dc.fields(self) )
    
# Utility functions.

def load_txt_file(file_path: str) -> tuple[str, int]:
    """Load a text file, then return its content and approximate number of tokens."""
    p = pathlib.Path(file_path)
    size_tokens = p.stat().st_size // 3 # Code usually uses 3 chars per token.
    txt = p.read_text(encoding="utf-8")
    return txt, size_tokens

# Benchmarking functions.

async def get_question_task(ctx: BenchmarkContext, model: bc.ModelInfo, agent: bc.LLMHandler,
    question_num: int, question: qs.QuestionData) -> mt.Metrics:
    """Run a single question and return performance metrics."""
    # Prepare question and prompt.
    question_text = f"{question.question}\n{question.masked_code}"
    full_prompt = f"{bc.DEFAULT_SYSTEM_INS}\n\n{question_text}"

    if ctx.system_doc:
        full_prompt += f"\n--- END OF QUESTION AND MASKED CODE ---\nBelow '--- DOCUMENTATION:' line is the documentation, which are all GemBox Software .NET components examples.\n--- DOCUMENTATION: \n{ctx.system_doc}\n--- END OF DOCUMENTATION ---\n Answer the question based on the documentation, return only the JSON object with no explanations, comments, or additional text.\n"

    if ctx.verbose:
        print(f"Q{question_num}: {tw.shorten(question_text, TRUNCATE_LENGTH)}")
    
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
            print(f"A{question_num}: {tw.shorten(str(results), TRUNCATE_LENGTH)}")
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
        return mt.Metrics(provider_name=agent.provider_name()) 

async def run_model_benchmark(ctx: BenchmarkContext, model_info: bc.ModelInfo, run_index: int = 0) -> mt.Metrics:
    """Run benchmark for a single model on all questions in parallel."""
    print(f"\n==={model_info.name}===")
    # Initialize model and agent.
    try:
        handler = model_info.create_handler(
            system_ins=ctx.system_ins, 
            parse_type=bc.ListOfStrings,
            web=ctx.web, 
            verbose=False)
    except Exception as e:
        print(f"\n--- Can't get model handler: {repr(e)}")
        return mt.Metrics(
            name=f"ERROR-{model_info.name}", 
            provider_name=model_info.provider_name(),
            api_issues=len(ctx.questions), 
            api_calls=len(ctx.questions) )
    
    # Create tasks for all questions to run in parallel.
    question_tasks = [
        get_question_task(ctx, model_info, handler, qi, question) 
        for qi, question in enumerate(ctx.questions, 1)
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
                task_metrics.append(mt.Metrics(provider_name=model_info.provider_name()))
    
    # Summarize model metrics, print, and return data.
    sum_metrics = mt.summarize_metrics(model_info.name, task_metrics)
    if ctx.verbose:
        mt.print_metrics(task_metrics)
    else:
        mt.print_metrics([sum_metrics])

    # Close handler to clean up resources.
    await handler.close()

    return sum_metrics

async def benchmark_context(ctx: BenchmarkContext) -> mt.Metrics:
    print(f"\n===== Benchmarking {len(ctx.models)} model(s) on {len(ctx.questions)} question(s) {ctx.bench_n_times} times. =====")
    print(ctx)

    perf_data = {model.name: [] for model in ctx.models}
    for run_idx in range(ctx.bench_n_times):
        print(f"\n=== Run {run_idx + 1} of {ctx.bench_n_times} ===")
        for model in ctx.models:
            perf_data[model.name].append(
                await run_model_benchmark(ctx, model, run_idx))

    # Flatten list of lists.
    all_metrics = [ 
        mt.summarize_metrics(model_name, model_metrics) 
        for model_name, model_metrics in perf_data.items()]

    print(f"\n=== SUMMARY OF: {ctx.description} ===")
    mt.print_metrics(all_metrics, csv_format=True)
    
    return (mt.summarize_metrics(ctx.description, all_metrics) if len(all_metrics) > 1 
            else all_metrics[0])

# Main test.

async def main_test():
    print("\n===== benchmark.main_test() =====")
    
    # Load questions from JSONL file.
    questions = qs.load_questions_from_jsonl("../2-bench-filter/test.jsonl")[:5]

    # Load documentation.
    doc, doc_approx_tokens = load_txt_file("GemBox-Spreadsheet-examples.txt")
    assert doc and doc_approx_tokens > 70_000, "FAIL: Documentation load."

    # Testing context.
    ctx = BenchmarkContext(
            models=bc.Models().by_names(['gpt-5-nano']),
            system_ins=bc.DEFAULT_SYSTEM_INS,
            questions=questions )
    
    # Benchmark.
    perf_data = await benchmark_context(ctx)
    assert perf_data.api_calls == len(questions), "FAIL: Not all questions were processed."

if __name__ == "__main__":
    # Load environment variables from parent directory .env.
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")
    
    asyncio.run(main_test())