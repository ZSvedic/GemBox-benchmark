from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

@dataclass(frozen=True)
class ModelInfo:
    openrouter_name: str
    direct_name: str
    input_cost: float
    output_cost: float
    tags: set[str]

    def __str__(self) -> str:
        return self.openrouter_name.split('/')[-1]

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * self.input_cost + (output_tokens / 1_000_000) * self.output_cost

_ALL_MODELS = [
    # ModelInfo('provider/model', 'direct_name', input_cost, output_cost, tags),
    # TEMPLATE:
    # ModelInfo('', '', 0.0, 0.0, {''}),
    # OpenAI models: https://openrouter.ai/provider/openai
    ModelInfo('openai/gpt-3.5-turbo', 'openai:gpt-3.5-turbo', 0.50, 1.50, {'openai', 'fast'}),
    ModelInfo('openai/gpt-4.1', 'openai:gpt-4.1', 2.0, 8.0, {'openai', 'fast'}),
    ModelInfo('openai/gpt-4o-2024-11-20', 'openai:gpt-4o-2024-11-20', 2.5, 10.0, {'openai'}),
    ModelInfo('openai/gpt-4o-mini', 'openai:gpt-4o-mini', 0.15, 0.60, {'openai', 'fast'}), # Low accuracy.
    ModelInfo('openai/gpt-5-nano', 'openai:gpt-5-nano', 0.05, 0.40, {'openai', 'fast'}), # Low accuracy.            
    ModelInfo('openai/gpt-5-mini', 'openai:gpt-5-mini', 0.25, 2.00, {'openai', 'fast', 'accurate'}), 
    ModelInfo('openai/gpt-5', 'openai:gpt-5', 1.25, 10.00, {'openai', 'accurate'}),  
    ModelInfo('openai/gpt-5-codex', 'openai:gpt-5-codex', 1.25, 10.0, {'openai', 'accurate'}),       
    # OpenAIPrompt models (Zel's private account): 
    # https://platform.openai.com/chat/edit?prompt=pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2&version=4
    # "openaiprompt" models are handled directly.
    # Prompt version 4 uses gpt-5-mini.
    ModelInfo('openaiprompt/GemBoxGPT-GBS-examples', 'pmpt_68d2af2e837c81939eeaf15bba79e95e0d72a7a17d0ec9e2', 0.25, 2.00, {'openai', 'prompt', 'accurate'}),
    # Google models: https://openrouter.ai/provider/google-ai-studio
    ModelInfo('google/gemma-3-12b-it:free', 'google-gla:gemma-3-12b-it:free', 0.0, 0.0, {'google', 'fast'}),
    ModelInfo('google/gemini-2.0-flash-001', 'google-gla:gemini-2.0-flash-001', 0.10, 0.40, {'google', 'fast'}),
    ModelInfo('google/gemini-2.5-flash-lite', 'google-gla:gemini-2.5-flash-lite', 0.10, 0.40, {'google', 'fast', 'accurate'}), 
    ModelInfo('google/gemini-2.5-flash', 'google-gla:gemini-2.5-flash', 0.30, 2.50, {'google', 'fast', 'accurate'}),
    ModelInfo('google/gemini-2.5-pro', 'google-gla:gemini-2.5-pro',10.0, 5.16, {'google', 'accurate'}),
    # Mistral models: https://openrouter.ai/provider/mistral
    ModelInfo('mistralai/codestral-2508', 'mistral:codestral-latest', 0.30, 0.90, {'mistral', 'fast', 'accurate'}),
    ModelInfo('mistralai/devstral-medium', 'mistral:devstral-medium-latest', 0.40, 2.00, {'mistral'}),
    ModelInfo('mistralai/mistral-large', 'mistralai/mistral-large-2407', 2.0, 6.0, {'mistral'}),
    # Anthropic models: https://openrouter.ai/provider/anthropic
    ModelInfo('anthropic/claude-3-haiku', 'anthropic:claude-3-5-haiku-latest', 0.25, 1.35, {'anthropic'}), # Low accuracy.
    ModelInfo('anthropic/claude-sonnet-4.5', 'anthropic:claude-sonnet-4.5-latest', 3.0, 15.00, {'anthropic'}),
    ModelInfo('anthropic/claude-opus-4.1', 'anthropic:claude-sonnet-4.5-haiku-latest', 15.00, 75.00, {'anthropic'}),
]

class Models:
    '''Fluent query object for model filtering.'''

    def __init__(self, models: list[ModelInfo] = None):
        self.models = models or _ALL_MODELS

    def filter(self, condition: callable) -> Models:
        """Filter models using a condition function."""
        self.models = [m for m in self.models if condition(m)]
        return self

    def by_tags(self, include: set[str] = set(), exclude: set[str] = set()) -> Models:
        return self.filter(lambda m: include.issubset(m.tags) and not (exclude & m.tags))

    def by_max_price(self, input_cost: float, output_cost: float) -> Models:
        return self.filter(lambda m: m.input_cost <= input_cost and m.output_cost <= output_cost)

    def by_names(self, names: list[str]) -> Models:
        return self.filter(lambda m: str(m) in names)

    def __str__(self):
        return ", ".join([str(m) for m in self.models])

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index):
        return self.models[index]

    def __len__(self):
        return len(self.models)

    def print_by_tags(self): 
        # Group models by tags.
        by_tag = defaultdict(list)
        for m in self.models:
            for t in m.tags:
                by_tag[t].append(m)
        # Print models by tags.
        for tag, models in by_tag.items(): 
            print(f"{tag} ({len(models)}): {Models(models)}")       

def main():
    # Print all models.
    print("\n=== Models().print_by_tags() ===")
    Models().print_by_tags()
    
    # Get models by names.
    names = ['gpt-5-mini', 'gemini-2.5-flash', 'codestral-2508']
    print(f"\n=== Models().by_names({names}) ===")
    models = Models().by_names(names)
    print(*models, sep="\n")
    
    # Filter models by tags.
    tags = {'openai', 'accurate'}
    print(f"\n=== Models().by_tags(include={tags}) ===")
    models = Models().by_tags(include=tags)
    print(*models, sep="\n")

    # Filter models by max price.
    print(f"\n=== Models().by_max_price(input_cost=0.5, output_cost=1.0) ===")
    models = Models().by_max_price(input_cost=0.5, output_cost=1.0)
    print(*models, sep="\n")

    # Chain filters.
    print(f"\n=== Models().by_tags(exclude={tags}).by_max_price(input_cost=0.5, output_cost=1.0) ===")
    models = Models().by_tags(exclude=tags).by_max_price(input_cost=0.5, output_cost=1.0)
    print(*models, sep="\n")

    # Demonstrate __str__().
    print("\n=== str(models[0]) ===")
    print(str(models[0]))

    # Demonstrate __repr__().
    print("\n=== repr(models[0]) ===")
    print(repr(models[0]))

if __name__ == "__main__":
    main()