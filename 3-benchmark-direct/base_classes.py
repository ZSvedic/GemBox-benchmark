from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import override

from pydantic import BaseModel

class LLMHandler(ABC):
    @abstractmethod
    def __init__(
        self, 
        model_info: ModelInfo, 
        *,
        system_prompt: str | None, 
        parse_type: type[BaseModel] | None,
        web_search: bool,
        verbose: bool): 
        ...
    @abstractmethod
    async def call(self, input: str) -> tuple[list[str], int, int]: 
        ...

@dataclass(frozen=True)
class ListOfStrings(BaseModel):
    completions: list[str]

@dataclass(frozen=True)
class ModelInfo:
    name: str
    prompt_id: str | None
    input_cost: float
    output_cost: float
    context_length: int
    direct_class: type[LLMHandler]
    tags: frozenset[str]

    def __str__(self) -> str:
        return self.name

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * self.input_cost + (output_tokens / 1_000_000) * self.output_cost

    def create_handler(self, *args, **kwargs) -> LLMHandler:
        return self.direct_class(self, *args, **kwargs)

class Models:
    '''Fluent query object for model filtering.'''

    _MODEL_REGISTRY = []

    def __init__(self, models: list[ModelInfo] = None):
        self.models = models or Models._MODEL_REGISTRY

    def filter(self, condition: callable) -> Models:
        """Filter models using a condition function."""
        filtered = [m for m in self.models if condition(m)]
        return Models(filtered)

    def by_tags(self, include: frozenset[str] = frozenset[str](), exclude: frozenset[str] = frozenset[str]()) -> Models:
        return self.filter(lambda m: include.issubset(m.tags) and not (exclude & m.tags))

    def by_max_price(self, input_cost: float, output_cost: float) -> Models:
        return self.filter(lambda m: m.input_cost <= input_cost and m.output_cost <= output_cost)

    def by_min_context_length(self, min_context_length: int) -> Models:
        return self.filter(lambda m: m.context_length >= min_context_length)

    def by_names(self, names: list[str]) -> Models:
        filtered = self.filter(lambda m: str(m) in names)
        assert len(filtered) == len(names), f"Requested {len(names)} models ({names}), but got {len(filtered)} models ({filtered})."
        return filtered

    def by_name(self, name: str) -> ModelInfo:
        for m in self.models:
            if str(m) == name:
                return m
        raise ValueError(f"Model {name} not found")

    def __str__(self):
        return ", ".join([str(m) for m in self.models])

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index):
        return self.models[index]

    def __len__(self):
        return len(self.models)

    def __add__(self, other: Models) -> Models:
        return Models(self.models + other.models)

    def print_by_tags(self): 
        # Group models by tags.
        by_tag = defaultdict(list)
        for m in self.models:
            for t in m.tags:
                by_tag[t].append(m)
        # Print models by tags.
        for tag, models in by_tag.items(): 
            print(f"{tag} ({len(models)}): {Models(models)}")     

class _AcmeLLMHandler(LLMHandler):
    @override
    def __init__(self, *args, **kwargs): 
        ...
    
    @override
    def call(self, input: str) -> tuple[list[str], int, int]:
        return (['AcmeLLM response for input'], 5, 10)

_DEFAULT_SYSTEM_PROMPT = """Answer a coding question related to GemBox Software .NET components.
Return a JSON object with a 'completions' array containing only the code strings that should replace the ??? marks, in order. 
Completions array should not contain any extra whitespace as results will be used for string comparison.

Example question: 
How do you set the value of cell A1 to "Hello"?
worksheet.Cells[???].??? = ???;
Your response:
{'completions': ['"A1"', 'Value', '"Hello"']}

Below '--- QUESTION AND MASKED CODE:' line is the question and masked code. Return only the JSON object with no explanations, comments, or additional text.

--- QUESTION AND MASKED CODE: """

_TEST_QUESTIONS = [
    "How to set value of A1 to 'Abracadabra'?",
    "How to format B2 to bold?",
]

_TEST_MODEL_REGISTRY = [
    # ModelInfo('name', prompt_id, input_cost, output_cost, context_length, direct_class, tags),
    # TEMPLATE:
    # ModelInfo('', None, 0.0, 0.0, 0, None, {''}),
    ModelInfo('AcmeLLM-3', None, 0.03, 0.05, 1000, _AcmeLLMHandler, {'acme', 'fast'}),
    ModelInfo('AcmeLLM-4', None, 0.05, 0.10, 2000, _AcmeLLMHandler, {'acme'}),
    ModelInfo('AcmeLLM-5', None, 0.10, 0.20, 3000, _AcmeLLMHandler, {'acme'}),
    ModelInfo('FooLLM-2-mini', None, 0.02, 0.10, 2000, _AcmeLLMHandler, {'foo', 'fast'}),
    ModelInfo('BarLLM-1', None, 0.03, 0.15, 3000, _AcmeLLMHandler, {'bar', 'fast'}),
    ModelInfo('BarLLM-2', None, 0.05, 0.20, 4000, _AcmeLLMHandler, {'bar'}),
]

def main_test():
    # Create test models.
    test_models = Models(_TEST_MODEL_REGISTRY)

    # Print all models.
    print("\n=== test_models.print_by_tags() ===")
    test_models.print_by_tags()
    
    # Get models by names.
    names = ['AcmeLLM-5', 'FooLLM-2-mini', 'BarLLM-1']
    print(f"\n=== test_models.by_names({names}) ===")
    models = test_models.by_names(names)
    print(*models, sep="\n")
    
    # Filter models by tags.
    tags = {'acme'}
    print(f"\n=== test_models.by_tags(include={tags}) ===")
    models = test_models.by_tags(include=tags)
    print(*models, sep="\n")

    # Filter models by max price.
    print(f"\n=== test_models.by_max_price(input_cost=0.05, output_cost=0.15) ===")
    models = test_models.by_max_price(input_cost=0.05, output_cost=0.15)
    print(*models, sep="\n")

    # Chain filters.
    print(f"\n=== test_models.by_tags(exclude={tags}).by_max_price(input_cost=0.05, output_cost=0.15) ===")
    models = test_models.by_tags(exclude=tags).by_max_price(input_cost=0.05, output_cost=0.15)
    print(*models, sep="\n")

    # Demonstrate __str__().
    print("\n=== str(models[0]) ===")
    print(str(models[0]))

    # Demonstrate __repr__().
    print("\n=== repr(models[0]) ===")
    print(repr(models[0]))

    # Demonstrate create_handler().
    print("\n=== test model_info.create_handler(...) ===")
    model_info = test_models.by_name('AcmeLLM-5')
    handler = model_info.create_handler()
    results, input_tokens, output_tokens = handler.call("What is the capital of France?")
    print(f"results: {results}, input_tokens: {input_tokens}, output_tokens: {output_tokens}")

if __name__ == "__main__":
    main_test()