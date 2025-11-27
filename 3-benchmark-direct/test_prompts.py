# Python stdlib.
import asyncio
import dataclasses as dc

# Third-party.
import dotenv

# Local modules.
import base_classes as bc
import metrics as mt
import questions as qs
import benchmark
from tee_logging import logging_context

# Main test.

async def main_test():
    print("\n===== test_prompts.main_test() =====")
    
    # Load questions from JSONL file.
    questions = qs.load_questions_from_jsonl("../2-bench-filter/test.jsonl")[:5]
    print(f"Using {len(questions)} questions.")

    # Starting context.
    s_ctx = benchmark.BenchmarkContext(
        models=bc.Models().by_names(['gemini-2.5-flash', 'deepseek/deepseek-chat']), #['gpt-5-mini']), 
        system_ins=bc.DEFAULT_SYSTEM_INS,
        questions=questions )

    # Testing contexts.
    contexts = [
        # dc.replace(s_ctx, description='gpt-5-mini, low reasoning, no search',
        #     reasoning='low', web=False),
        # dc.replace(s_ctx, description='gpt-5-mini, medium reasoning, web search',
        #     reasoning='medium', web=True),
        dc.replace(s_ctx, description='gpt-5-mini, medium reasoning, domain search',
            reasoning='medium', web=True, include_domains='www.gemboxsoftware.com'),
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