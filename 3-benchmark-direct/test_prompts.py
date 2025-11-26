# Python stdlib:
import asyncio
from collections import namedtuple

# Third-party:
import dotenv

# Local modules:
import base_classes as bc
import questions as qs
import metrics as mt
import benchmark
from tee_logging import logging_context

async def main_test():
    print("\n===== test_prompts.main_test() =====")
    
    # Load questions from JSONL file.
    questions = qs.load_questions_from_jsonl("../2-bench-filter/test.jsonl")[:5]
    print(f"Using {len(questions)} questions.")

    # Starting context.
    s_ctx = benchmark.BenchmarkContext().replace(
        models=bc.Models().by_names(['gpt-5-mini']), 
        system_ins=bc._DEFAULT_SYSTEM_INS,
        questions=questions )

    # Testing contexts.
    contexts = [
        s_ctx.replace(
            description='gpt-5-mini, low reasoning, no search',
            reasoning='low',
            web_search=False,
        ),
        s_ctx.replace(
            description='gpt-5-mini, medium reasoning, web search',
            reasoning='medium',
            web_search=True,
        ),
        s_ctx.replace(
            description='gpt-5-mini, medium reasoning, domain search',
            reasoning='medium',
            web_search=True,
            include_domains='www.gemboxsoftware.com',
        ),
    ]

    # Benchmark models.
    perf_data = [await benchmark.benchmark_context(ctx) for ctx in contexts]

    # Print summary.
    print(f"\n=== SUMMARY OF ALL TESTS in test_prompts.py ===")
    mt.print_metrics(perf_data, True)

if __name__ == "__main__":
    # Load environment variables from parent directory .env.
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")
    
    with logging_context("test_prompts"):
        asyncio.run(main_test())