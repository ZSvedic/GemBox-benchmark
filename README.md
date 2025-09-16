# GemBox Benchmark

This repository contains three *sub-projects* that create a benchmark dataset for [GemBox Software](https://www.gemboxsoftware.com/) components and evaluate LLMs on that generated dataset: 

+ *Project 1: Inputs* - Gathers raw question-answer pairs about common GemBox usage tasks (e.g. printing options, reading Excel files, formatting cells).
+ *Project 2: Bench Filter* - Cleans and structures the data into a proper benchmark dataset (JSON format). The final dataset is published on Hugging Face: [GBS-benchmark at HuggingFace](https://huggingface.co/datasets/ZSvedic/GBS-benchmark)
+ *Project 3: Benchmark LLM* - Uses the dataset to evaluate various LLMs, measuring their accuracy and performance in answering the GemBox API questions.

The benchmark goal is to test LLM accuracy in generating the correct code for common tasks using GemBox APIs.

## Installation & Setup

Requirements:
+ Visual Studio Code - [official download](https://code.visualstudio.com/download)
+ C# 13.0 / [.NET 9.0 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/9.0) - The easiest install is via [VSCode C# extension](https://marketplace.visualstudio.com/items?itemName=ms-dotnettools.csharp).
+ GemBox.Spreadsheet and dependencies - [v2025.9.10 via NuGet](https://www.nuget.org/packages/GemBox.Spreadsheet/):
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
cd GemBox-benchmark/3-benchmark-llm/    # Go to Python project.
uv venv --python 3.10                   # Or newer.
source .venv/bin/activate               # Linux/macOS.
uv sync                                 # Install dependencies.
cd ..                                   # Go back to root.
```
3. Create an ".env" file in the project root with your API keys (if only using OpenRouter, then only OPENROUTER_API_KEY is needed):
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

This C# project generates the initial Q&A *prompt data* for the benchmark. It uses the GemBox.Spreadsheet library to enumerate typical tasks a developer can perform and creates *question* and *answer* pairs for each. The output is a raw collection of GemBox-related Q&A items with code snippets containing `???` placeholders, plus the correct answers.

## Project "2-bench-filter" (optional)

This Python project filters and structures the raw Q&A data into a benchmark dataset. Each item includes:  
+ *category* (feature area),  
+ *question* (natural language query),  
+ *masked_code* (snippet with `???`),  
+ *answers* (correct tokens).  

The dataset is stored in JSONL format (see [GBS-benchmark at HuggingFace](https://huggingface.co/datasets/ZSvedic/GBS-benchmark)).

## Project "3-benchmark-llm"

This Python project runs evaluations of different LLMs using the dataset. It supports OpenAI, Google, Anthropic, and many other providers via [OpenRouter](https://openrouter.ai/). Each model is asked to fill in the `???` tokens, and the outputs are validated. The evaluation measures *accuracy*, *speed*, and *cost*.

### Example results

```bash
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


