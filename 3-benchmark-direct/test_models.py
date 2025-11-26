# Python stdlib:
import asyncio

# Third-party:
import dotenv

# Local modules:
import base_classes as bc
import questions as qs
import metrics as mt
import benchmark
from tee_logging import logging_context

async def main_test():
    print("\n===== test_models.main_test() =====")
    
    # Load questions from JSONL file.
    questions = qs.load_questions_from_jsonl("../2-bench-filter/test.jsonl")[:10]
    print(f"Using {len(questions)} questions.")

    # Load documentation.
    context_txt, context_approx_tokens = benchmark.load_txt_file("GemBox-Spreadsheet-examples.txt")
    print(f"Documentation 1 of ~length: {context_approx_tokens} tokens, starting with: {context_txt[:100]}")

    # Filter models.
    models = (
        bc.Models()
        # .by_web_search(True)
        # .by_min_context_length(context_approx_tokens)
        # .by_tags(include={'openrouter'})
        # .by_names(['prompt-GBS-examples-GPT5mini', 'prompt-GBS-examples-GPT5']) 
    )

    # print(f"Filtered models ({len(models)}): {models}")
    
    # Create starting context.
    start_bench_ctx = benchmark.BenchmarkContext()
    
    # Create testing runs.
    runs = [
        # ('A. Plain call + low reasoning', False, 'low', 30, '', 
        #  models.by_names(['gemini-2.5-flash', 'gpt-5-nano', 'gpt-5-mini', 'gpt-5.1','gpt-5.1-codex-mini', 'gpt-5-codex'])),
        ('B. Web search + medium reasoning', True, 'medium', 60, '', 
         models.by_names(['gpt-5-mini'])),
        # ('C. Context + medium reasoning', False, 'medium', 60, context_txt, 
        #  models.by_names(['gemini-2.5-flash', 'gpt-5-mini', 'gpt-5', 'gpt-5-codex', 'gpt-5.1-codex'])),
        # ('D. RAG OpenAI + medium reasoning', False, 'medium', 60, '', 
        #  models.by_names(['prompt-web-search-GPT-5-mini'])),
        ]
    
    for run_name, _, _, _, _, run_models in runs:
        print(run_name, [m.name for m in run_models]) 

    # Benchmark models.
    perf_data = [
        await benchmark.benchmark_context(
            text,
            start_bench_ctx.replace(
                web_search=web, 
                reasoning_effort=reasoning,
                timeout_sec=timeout,
                context=context),
            run_models, 
            questions)
        for text, web, reasoning, timeout, context, run_models in runs
        ]

    # Print summary.
    print(f"\n=== SUMMARY OF ALL TESTS in test_models.py ===")
    mt.print_metrics(perf_data, True)

if __name__ == "__main__":
    # Load environment variables from parent directory .env.
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")
    
    with logging_context("test_models"):
        asyncio.run(main_test())