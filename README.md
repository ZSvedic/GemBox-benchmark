# GemBox Benchmark

This repository contains three *sub-projects* that create a benchmark dataset for [GemBox Software](https://www.gemboxsoftware.com/) components and evaluate LLMs on that generated dataset. The goal of this benchmark is to test LLM accuracy in generating the correct code for common tasks using these GemBox APIs.

- *Project 1: Inputs* – Gathers raw question-answer pairs about common GemBox usage tasks (e.g. printing options, reading Excel files, formatting cells).
- *Project 2: Bench Filter* – Cleans and structures the data into a proper benchmark dataset (JSON format). The final dataset is published on Hugging Face: [GBS-benchmark at HuggingFace](https://huggingface.co/datasets/ZSvedic/GBS-benchmark)
- *Project 3: Benchmark LLM* – Uses the dataset to evaluate various LLMs, measuring their accuracy and performance in answering the GemBox API questions.

## 1. Inputs: Data Collection

This C# project generates the initial Q&A *prompt data* for the benchmark. It uses the GemBox.Spreadsheet library to enumerate typical tasks a developer can perform and creates *question* and *answer* pairs for each. The output is a raw collection of GemBox-related Q&A items with code snippets containing `???` placeholders, plus the correct answers.

## 2. Bench Filter: Dataset Preparation

This Python project filters and structures the raw Q&A data into a benchmark dataset. Each item includes:  
- *category* (feature area),  
- *question* (natural language query),  
- *masked_code* (snippet with `???`),  
- *answers* (correct tokens).  

The dataset is stored in JSONL format (see [GBS-benchmark at HuggingFace](https://huggingface.co/datasets/ZSvedic/GBS-benchmark)).

## 3. Benchmark LLM: Model Evaluation

This Python project runs evaluations of different LLMs using the dataset. It supports OpenAI, Google, Anthropic, and many other providers via OpenRouter. Each model is asked to fill in the `???` tokens, and the outputs are validated. The evaluation measures *accuracy*, *latency*, and *cost*.

## Installation & Setup

Git clone and move to folder:
```bash
git clone https://github.com/ZSvedic/GemBox-benchmark
cd GemBox-benchmark
```

Install [uv](https://github.com/astral-sh/uv) and then:

```bash
uv venv --python 3.10       # Or newer.
source .venv/bin/activate   # Linux/macOS.
uv sync                     # Install dependencies.
```

Create a .env file in the project root with your API keys (if you will only use OpenRouter, then only OPENROUTER_API_KEY is needed):
```bash
OPENROUTER_API_KEY=...
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
```

Then run '3-benchmark-llm' to execute the benchmark and see the results.

## Example results

```bash
...

Total Wall Time: 111.40s
Total Errors: 18 out of 224 API calls = 8.0% error rate

Accuracy Ranking (highest accuracy first):
1. openai/gpt-5-mini: 58.9%
2. google/gemini-2.5-flash: 52.7%
3. deepseek/deepseek-chat-v3-0324: 41.1%
4. mistralai/codestral-2508: 35.7%
5. openai/gpt-5-nano: 25.6%
6. google/gemini-2.5-flash-lite: 25.3%
7. openai/gpt-4o-mini: 22.6%
8. anthropic/claude-3-haiku: 16.7%

Error Rate Ranking (lowest error rate first):
1. openai/gpt-4o-mini: 0.0%
2. anthropic/claude-3-haiku: 0.0%
3. mistralai/codestral-2508: 0.0%
4. deepseek/deepseek-chat-v3-0324: 0.0%
5. google/gemini-2.5-flash-lite: 3.6%
6. openai/gpt-5-mini: 10.7%
7. google/gemini-2.5-flash: 10.7%
8. openai/gpt-5-nano: 39.3%
```


