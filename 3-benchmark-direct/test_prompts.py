# Python stdlib.
import asyncio
import dataclasses as dc
import warnings

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
    questions = qs.load_questions_from_jsonl("../2-bench-filter/test.jsonl")
    print(f"Using {len(questions)} questions.")

    # Load documentation.
    doc_sitemap, doc_sitemap_tokens = benchmark.load_txt_file("docs/GB-Spreadsheet-sitemap-examples-LLM.txt")
    doc_examples, doc_examples_tokens = benchmark.load_txt_file("docs/GB-Spreadsheet-examples.txt")
    doc_min_tokens = max(doc_sitemap_tokens, doc_examples_tokens)

    # Filter models.
    models = (
        bc.Models()
        .by_web(True)
        .by_min_context_length(doc_min_tokens)
        # .by_max_price(0.50, 3.00)
        # .by_tags(exclude={'rag', 'prompt'})
        # .by_names(['gpt-5-mini'])
        .by_names(['gemini-2.5-flash', 'gpt-5-mini'])
    )
    print(f"Filtered models ({len(models)}): {models}")

    # Starting context.
    s_ctx = benchmark.BenchmarkContext(
        models=models,
        verbose=True, 
        system_ins=bc.DEFAULT_SYSTEM_INS,
        questions=questions )

    # Testing contexts.
    contexts = [
        dc.replace(s_ctx, description='G. Sitemap web search + medium reasoning (p2)',
            reasoning='medium', timeout_sec=60, web=True, system_doc=doc_sitemap, include_domains='gemboxsoftware.com'),
        # dc.replace(s_ctx, description='C. Context + medium reasoning',
        #     reasoning='medium', timeout_sec=60, web=False, system_doc=doc_examples),
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