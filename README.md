# GemBox Benchmark

This repository contains three sub-projects that create a benchmark dataset for [GemBox Software](https://www.gemboxsoftware.com/) components and evaluate LLMs on that generated dataset: 

+ *Project 1: Inputs (C#)* - Contains raw question-answer pairs about common GemBox usage tasks (e.g. printing options, reading Excel files, etc.).
+ *Project 2: Bench Filter (Python)* - Filters "1-inputs" into a benchmark dataset (JSON format), see example: [GBS-benchmark at HuggingFace](https://huggingface.co/datasets/ZSvedic/GBS-benchmark)
+ *Project 3: Benchmark LLM (Python)* - Uses the dataset to evaluate LLMs on accuracy, speed and cost when answering GemBox API questions.

## Installation & Setup

Requirements:
+ Visual Studio Code - [official download](https://code.visualstudio.com/download)
+ C# 13.0 / [.NET 9.0 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/9.0) - The easiest install is via [VS Code C# extension](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.csharp).
+ GemBox.Spreadsheet and dependencies - If not installed automatically by VS Code when opening the workspace, get [v2025.9.10 via NuGet](https://www.nuget.org/packages/GemBox.Spreadsheet/):
```bash
dotnet add package GemBox.Spreadsheet --version 2025.9.107
dotnet add package HarfBuzzSharp.NativeAssets.Linux
dotnet add package SkiaSharp.NativeAssets.Linux.NoDependencies
```

Next steps:
1. Git clone:
```bash
git clone https://github.com/ZSvedic/GemBox-benchmark
```
2. For the Python project, use [uv](https://github.com/astral-sh/uv) package manager to install dependencies:
```bash
cd GemBox-benchmark/3-benchmark-llm/    # Go to the Python project.
uv venv --python 3.10                   # Env with py 3.10 or newer.
source .venv/bin/activate               # For Linux/MacOS.
uv sync                                 # Install dependencies.
cd ..                                   # Go back to the root.
```
3. Create an ".env" file in the root ("GemBox-benchmark" folder) with your API keys. If only using OpenRouter, then only OPENROUTER_API_KEY is needed. Example:
```bash
OPENROUTER_API_KEY=...
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
```
4. Open in VSCode:
```bash
code GB-benchmark.code-workspace
```
VSCode "Run and Debug" tab should now have run configurations for each of the subprojects below.

## Project "1-inputs" (optional)

This C# project contains Q&A data for the dataset. It uses the GemBox.Spreadsheet library to enumerate typical tasks. Comments before task contain a single "Question:" and one or more "Mask:" for each code answer. Each mask specifies a regex that will mask certain part of the following code line before asking LLM to fill it. 

### Example input C# code
```csharp
...
    // Question: How do you enable printing of row and column headings?
    // Mask: \bPrintOptions\.PrintHeadings\b
    // Mask: \btrue\b
    worksheet.PrintOptions.PrintHeadings = true;

    // Question: How do you set the worksheet to print in landscape orientation?
    // Mask: \bPrintOptions\.Portrait\b
    // Mask: \bfalse\b
    worksheet.PrintOptions.Portrait = false;
...
```

## Project "2-bench-filter" (optional)

This Python project filters .cs files from "1-inputs" to extract Q&A into a benchmark dataset. Each dataset row contains:  
+ *category* (from the .cs file name),  
+ *question* (EN language query),  
+ *masked_code* (code snippet with `???` placeholders),  
+ *answers* (correct text to fill `???` placeholders).  

### Example [JSONL dataset](https://huggingface.co/datasets/ZSvedic/GBS-benchmark)
```json
...
{"category": "PrintView", "question": "How do you enable printing of row and column headings?", "masked_code": "worksheet.??? = ???;", "answers": ["PrintOptions.PrintHeadings", "true"]}
{"category": "PrintView", "question": "How do you set the worksheet to print in landscape orientation?", "masked_code": "worksheet.??? = ???;", "answers": ["PrintOptions.Portrait", "false"]}
...
```

## Project "3-benchmark-llm"

This Python project uses the dataset to run LLM evaluations. It supports OpenAI, Google, Anthropic, and many other providers via [OpenRouter](https://openrouter.ai/). Each model is asked to fill in `???` placeholders, and the outputs are validated. The evaluation measures *accuracy*, *speed*, and *cost*.

### Example output and results

```console
...
Context(timeout_seconds=30,
        delay_ms=50,
        verbose=True,
        truncate_length=150,
        max_parallel_questions=30,
        retry_failures=True,
        use_caching=True,
        use_open_router=True,
        benchmark_n_times=3)

Benchmarking 4 model(s) on 28 question(s) 3 times.

=== Run 1 of 3 ===

...
Q3: How do you enable printing of row and column headings?
worksheet.??? = ???;
Q4: How do you set the worksheet to print in landscape orientation?
worksheet.??? = ???;
...
A3: ['PrintOptions.PrintHeadings', 'true']
✓ CORRECT
A4: ['PrintOptions.Orientation', 'Orientation.Landscape']
✗ INCORRECT, expected: ['PrintOptions.Portrait', 'false']
...
=== OVERALL BENCHMARK SUMMARY ===

  Model Summary: openai/gpt-5-mini
  Total Tokens: 48671
  Total Cost: $0.067764
  Total Time: 86.70s
  Overall Accuracy: 65.4%
  Errors: 0 out of 84 API calls

...

Total Cost: $0.219338
Total Time: 341.45s
Total Errors: 3 out of 336 API calls = 0.9% error rate

Accuracy Ranking (highest accuracy first):
1. openai/gpt-5-mini: 65.4%
2. google/gemini-2.5-flash: 60.5%
3. mistralai/codestral-2508: 39.3%
4. google/gemini-2.5-flash-lite: 38.7%

Error Rate Ranking (lowest error rate first):
1. openai/gpt-5-mini: 0.0%
2. google/gemini-2.5-flash: 0.0%
3. mistralai/codestral-2508: 0.0%
4. google/gemini-2.5-flash-lite: 10.7%
```


