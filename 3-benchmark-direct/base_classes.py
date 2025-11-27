# Python stdlib.
from __future__ import annotations
from abc import ABC, abstractmethod
import asyncio
from collections import defaultdict
from collections.abc import Iterator, Callable, Collection
import dataclasses as dc
from typing import Any, override, Optional

# Third-party.
from pydantic import BaseModel          

# LLMHandler class.

CallDetailsType = dict[str, list[str]]
UsageType = tuple[int, int]

@dc.dataclass(frozen=True)
class LLMHandler(ABC):
    model_info: ModelInfo                       # Model information.
    web: bool = False                           # Use web search?
    include_domains: str = ''                   # Domains to include in web search.
    system_ins: str | None = None               # System instructions. 
    parse_type: type[BaseModel] | None = None   # Pydantic model type for parsing results.
    verbose: bool = False                       # Verbose output?
    
    @classmethod
    @abstractmethod
    def get_client(cls):
        '''Returns the LLM client instance.'''
        ...

    @classmethod
    @abstractmethod
    async def close(cls):
        '''Releases LLM client instance and resources.'''
        ...

    @classmethod
    @abstractmethod
    def provider_name(cls) -> str:
        ...

    @abstractmethod
    async def call(self, input: str) -> tuple[Any, CallDetailsType, UsageType]: 
        ...

    @staticmethod
    def strip_code_fences(text: str) -> str:
        '''Removes triple backtick code fences that some models that output.'''
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```") and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1])
        return text

# ListOfStrings class for parsing.

@dc.dataclass(frozen=True)
class ListOfStrings(BaseModel):
    completions: list[str]

# ModelInfo and Models classes.

@dc.dataclass(frozen=True)
class ModelInfo:
    name: str = '?'                             # Model name.
    prompt_id: str = ''                         # Prompt ID (if using a prompt).
    web: bool = False                           # Supports web search?
    in_usd: float = 0.0                         # Input cost per 1M tokens.
    out_usd: float = 0.0                        # Output cost per 1M tokens.
    context_len: int = 0                        # Maximum context length in tokens.
    handler: type[LLMHandler] | None = None     # Responsible LLMHandler class.
    tags: frozenset[str] = frozenset()          # Tags associated with the model.

    def __str__(self) -> str:
        return self.name

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * self.in_usd + (output_tokens / 1_000_000) * self.out_usd

    def create_handler(self, *args, **kwargs) -> LLMHandler:
        return self.handler(self, *args, **kwargs)
    
    def provider_name(self) -> str:
        return self.handler.provider_name()

class Models(Collection[ModelInfo]):
    '''Fluent query object for model filtering.'''

    _MODEL_REGISTRY = []

    def __init__(self, models: Optional[list[ModelInfo]] = None):
        self.models = (
            models if models is not None 
            else Models._MODEL_REGISTRY)

    def filter(self, condition: Callable) -> Models:
        """Filter models using a condition function."""
        filtered = [m for m in self.models if condition(m)]
        return Models(filtered)

    def by_tags(self, include: set[str] = set[str](), exclude: set[str] = set[str]()) -> Models:
        return self.filter(lambda m: include.issubset(m.tags) and not (exclude & m.tags))

    def by_max_price(self, in_usd: float, out_usd: float) -> Models:
        return self.filter(lambda m: m.in_usd <= in_usd and m.out_usd <= out_usd)

    def by_min_context_length(self, min_context_len: int) -> Models:
        return self.filter(lambda m: m.context_len >= min_context_len)
    
    def by_web(self, web: bool) -> Models:
        return self.filter(lambda m: m.web == web)

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

    def __iter__(self) -> Iterator[ModelInfo]:
        return iter(self.models)

    def __getitem__(self, index: int) -> ModelInfo:
        return self.models[index]

    def __len__(self) -> int:
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

    def __contains__(self, x: object) -> bool:
        return x in self.models

# Constants.

DEFAULT_SYSTEM_INS = """Answer a coding question related to GemBox Software .NET components.
Return a JSON object with a "completions" array containing only the code strings that should replace the ??? marks, in order. 
Completions array should not contain any extra whitespace as results will be used for string comparison.
If needed and available, use web search to find the most recent version of the GemBox API help pages.

Example question: 
How do you set the value of cell A1 to "Hello"?
worksheet.Cells[???].??? = ???;
Your response:
{"completions": ["\\"A1\\"", "Value", "\\"Hello\\""]}

Return only the JSON object with no explanations, comments, or additional text.
"""

TEST_QUESTIONS = [
    "How to set value of A1 to 'Abracadabra'?\nworksheet.Cells[???].??? = ???;",
    "How to format B2 to bold?\nworksheet.Cells[???].??? = ???;",
]

# Example LLMHandler implementation for testing.

class _AcmeLLMHandler(LLMHandler):

    @override
    @classmethod
    def get_client(cls):
        return None
    
    @override
    @classmethod
    def provider_name(cls) -> str:
        return "Acme"
    
    @override
    async def call(self, input: str) -> tuple[Any, CallDetailsType, UsageType]:
        return (['AcmeLLM response for input'], {'web': ['https://www.acme.com']}, (5, 10))
    
    @override
    @classmethod
    async def close(cls):
        pass

_base = ModelInfo(handler=_AcmeLLMHandler)

_TEST_MODEL_REGISTRY = [
    dc.replace(_base, name='AcmeLLM-3',     in_usd=0.03, out_usd=0.05, context_len=1000, web=False, tags={'acme'}),
    dc.replace(_base, name='AcmeLLM-4',     in_usd=0.05, out_usd=0.10, context_len=2000, web=False, tags={'acme'}),
    dc.replace(_base, name='AcmeLLM-5',     in_usd=0.10, out_usd=0.20, context_len=3000, web=True,  tags={'acme'}),
    dc.replace(_base, name='FooLLM-2-mini', in_usd=0.02, out_usd=0.10, context_len=2000, web=False, tags={'foo'}),
    dc.replace(_base, name='BarLLM-1',      in_usd=0.03, out_usd=0.15, context_len=3000, web=False, tags={'bar'}),
    dc.replace(_base, name='BarLLM-2',      in_usd=0.05, out_usd=0.20, context_len=4000, web=True,  tags={'bar'}),
]

# Test functions.

async def _test_call_handler(handler: LLMHandler, questions: list[str]):
    print(f'Calling LLM for {len(questions)} questions...')

    async_responses = [handler.call(q) for q in questions]
    responses = await asyncio.gather(*async_responses)
    
    for question, response in zip(questions, responses):
        result, links, (input_tokens, output_tokens) = response
        print(f"\nQuestion: {question}\nResults: {result}\nLinks: {links}\nInput tokens: {input_tokens}\nOutput tokens: {output_tokens}\n")

async def main_test():
    print("\n===== base_classes.main_test() =====")

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
    assert len(models) == len(names), "FAIL: wrong number of models."
    
    # Filter models by tags.
    tags = {'acme'}
    print(f"\n=== test_models.by_tags(include={tags}) ===")
    models = test_models.by_tags(include=tags)
    print(*models, sep="\n")
    assert all(tags.issubset(m.tags) for m in models), "FAIL: some models missing required tags."

    # Filter models by max price.
    print(f"\n=== test_models.by_max_price(input_cost=0.05, output_cost=0.15) ===")
    models = test_models.by_max_price(in_usd=0.05, out_usd=0.15)
    print(*models, sep="\n")
    assert all(m.in_usd <= 0.05 and m.out_usd <= 0.15 
               for m in models), "FAIL: some models exceed max price."

    # Chain filters.
    print(f"\n=== test_models.by_tags(exclude={tags}).by_max_price(input_cost=0.05, output_cost=0.15) ===")
    models = test_models.by_tags(exclude=tags).by_max_price(in_usd=0.05, out_usd=0.15)
    print(*models, sep="\n")
    assert len(models)==2, "FAIL: wrong chained filter result."

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
    results, links, (input_tokens, output_tokens) = await handler.call("What is the capital of France?")
    print(f"results: {results}\nlinks: {links}\ninput_tokens: {input_tokens}\noutput_tokens: {output_tokens}")

    # Demonstrate _test_call_handler().
    print("\n=== _test_call_handler(handler, ['What is the capital of France?']) ===")
    await _test_call_handler(handler, ['What is the capital of France?'])

if __name__ == "__main__":
    asyncio.run(main_test())