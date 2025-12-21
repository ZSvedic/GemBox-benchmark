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
import dotnet_compile
from tee_logging import logging_context

# Main test.

_PROMPT = '''You are GemBoxGPT, a GPT chatbot that provides coding assistance for GemBox.Spreadsheet. 
Assist only with GemBox-related queries. when giving code, give the complete C# code marked with ```csharp ... ``` blocks.'''

_QUESTIONS = [
    qs.QuestionData(
        category="compilation",
        question="Write C# code that uses GemBox.Spreadsheet to create an Excel file with 'Hello!' in cell A1.",
        masked_code="",
        answers=["using GemBox.Spreadsheet;\nclass Program {\n    static void Main() {\n        SpreadsheetInfo.SetLicense(\"FREE-LIMITED-KEY\");\n        var ef = new ExcelFile();\n        var ws = ef.Worksheets.Add(\"Sheet1\");\n        ws.Cells[0, 0].Value = \"Hello!\";\n        ef.Save(\"Output.xlsx\");\n    }\n}"]
    ),
    qs.QuestionData(
        category='compilation',
        question='''Using GemBox.Spreadsheet, generate C# code to create "../Earth–HHhMMm.xlsx", where HH and MM are current hours and minutes (24-hour time). A "Breakdown" sheet has columns "Continents" and "Area (km2)". List all known continents and respective areas that you know. Use thousands separators for km2, make columns autofit, and make header bold. 
Right to the table, create a pie chart named "Landmass breakdown" that shows continent's area percentage. Each pie should have a label with continent name, area, and percentage. ''',
        masked_code="",
        answers=[],
    ),
]

async def main_test():
    print("\n===== test_compilation.main_test() =====")

    # Filter models.
    models = (
        bc.Models()
        .by_names(['gpt-5-mini', 'gpt-5', 'gpt-5.1-codex', 'x-ai/grok-4.1-fast'])
    )
    print(f"Filtered models ({len(models)}): {models}")

    # Starting context.
    s_ctx = benchmark.BenchmarkContext(
        models=models,
        verbose=True, 
        system_ins=_PROMPT,
        questions=_QUESTIONS,
        parse_type=None )
    
    # Testing contexts.
    contexts = [
        dc.replace(s_ctx, description='Plain call + low reasoning', 
                   reasoning='low', timeout_sec=40),
         dc.replace(s_ctx, description='Plain call + medium reasoning', 
                   reasoning='medium', timeout_sec=60),   
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
    
    with logging_context("test_compilation"):
        asyncio.run(main_test())