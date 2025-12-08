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
    print("\n===== test_models.main_test() =====")
    
    # Load questions from JSONL file.
    questions = qs.load_questions_from_jsonl("../2-bench-filter/test.jsonl")
    print(f"Using {len(questions)} questions.")

    # Load documentation.
    doc, doc_approx_tokens = benchmark.load_txt_file("docs/GB-Spreadsheet-examples.txt")
    print(f"Documentation 1 of ~length: {doc_approx_tokens} tokens, starting with: {doc[:100]}")

    # Filter models.
    models = (
        bc.Models()
        # .by_web(True)
        # .by_min_context_length(doc_approx_tokens)
        # .by_tags(include={'openrouter'})
        # .by_max_price(0.50, 2.00)
        .by_names(['gpt-4o']) 
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
        dc.replace(s_ctx, description='A. Plain call + low reasoning', 
                   reasoning='low', timeout_sec=30),
        dc.replace(s_ctx, description='B. Web search + medium reasoning', 
                   reasoning='medium', web=True, timeout_sec=60),
        dc.replace(s_ctx, description='C. Context + medium reasoning', 
                   reasoning='medium', timeout_sec=60, system_doc=doc),
        # dc.replace(s_ctx, description='D. RAG OpenAI + medium reasoning', 
        #            reasoning='medium', timeout_sec=60),
    ]
  
    # Benchmark models.
    perf_data = [await benchmark.benchmark_context(ctx) for ctx in contexts]

    # Print summary.
    print(f"\n=== SUMMARY OF ALL TESTS ===")
    mt.print_metrics(perf_data, True)

if __name__ == "__main__":
    # Load environment variables from parent directory .env.
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")
    
    with logging_context("test_models"):
        asyncio.run(main_test())