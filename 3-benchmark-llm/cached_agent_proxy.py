import os
import json
import httpx

from typing import Type
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model

class CachedAgentProxy(Agent):
    ''' A proxy agent that caches responses from a real OpenRouter Agent to a specified file. 
    Once cached, the agent will return the cached response from dict instead of making a real API call. 
    This proxy only implements the methods it can accurately simulate. '''
    
    def __init__(self, model: Model | str, output_type: Type[BaseModel], cache_file: str, verbose: bool = False):
        super().__init__(model, output_type=output_type)
        self.cache_file = cache_file
        self.cache = {}
        self.verbose = verbose
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from file if it exists."""
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached responses from: {self.cache_file}")
        except (FileNotFoundError, json.JSONDecodeError):
            self.cache = {}
            print(f"No existing cache found, starting with empty cache.")
        
    async def run(self, *args, **kwargs):
        if args[0] in self.cache:
            if self.verbose:
                print("CACHE HIT")
            cached_result = self._reconstruct_result(self.cache[args[0]])
            cached_result._was_cached = True
            return cached_result
        else:
            result = await super().run(*args, **kwargs)
            # Cache only the serializable parts of the result
            cacheable_result = self._make_cacheable(result)
            self.cache[args[0]] = cacheable_result
            result._was_cached = False
            return result

    def response_2_usage_results(self, response: httpx.Response) -> tuple[dict, list[str]]:
        return response.usage(), response.output.completions
    
    def _make_cacheable(self, result):
        """Extract serializable data from the agent result."""
        try:
            # Handle Pydantic models by converting to dict
            output = getattr(result, 'output', None)
            if hasattr(output, 'model_dump'):
                output = output.model_dump()
            elif hasattr(output, 'dict'):
                output = output.dict()
            
            # Create a simple object with just the essential data
            cacheable = {
                'output': output,
                'usage_data': self._extract_usage(result),
                'metadata': {
                    'cached_at': str(type(result)),
                    'has_output': hasattr(result, 'output'),
                    'has_usage': hasattr(result, 'usage')
                }
            }
            return cacheable
        except Exception as e:
            print(f"Warning: Could not make result cacheable: {e}")
            return {'error': f'Could not cache result: {e}'}
    
    def _extract_usage(self, result):
        """Extract usage information if available."""
        try:
            usage = result.usage() if hasattr(result, 'usage') and callable(result.usage) else None
            if usage:
                return {
                    'input_tokens': getattr(usage, 'input_tokens', 0),
                    'output_tokens': getattr(usage, 'output_tokens', 0)
                }
            return None
        except:
            return None
    
    def _reconstruct_result(self, cached_data):
        """Reconstruct a result-like object from cached data."""
        class CachedResult:
            def __init__(self, data):
                # Reconstruct the output - if it was a dict, try to recreate the original structure
                output_data = data.get('output')
                if isinstance(output_data, dict) and 'completions' in output_data:
                    # Try to recreate a CodeCompletion-like object
                    class CachedCodeCompletion:
                        def __init__(self, completions):
                            self.completions = completions
                    self.output = CachedCodeCompletion(output_data['completions'])
                else:
                    self.output = output_data
                
                self._cached_usage = data.get('usage_data')
            
            def usage(self):
                if self._cached_usage:
                    class CachedUsage:
                        def __init__(self, usage_data):
                            self.input_tokens = usage_data.get('input_tokens', 0)
                            self.output_tokens = usage_data.get('output_tokens', 0)
                    return CachedUsage(self._cached_usage)
                return None
        
        return CachedResult(cached_data)
    
    def __enter__(self):
        """Context manager entry."""
        return self

    def close(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            print(f"Cache saved to: {self.cache_file}")
        except Exception as e:
            print(f"Warning: Could not save cache to: {self.cache_file}: {e}")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
