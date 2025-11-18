import asyncio
import threading
from typing import Any, override

import dotenv
from google import genai
from google.genai import types

import base_classes as bc
    
# Constants.

_LOCATION = "europe-west4"
_PROJECT_ID = "gen-lang-client-0658217610"
_RAG_CORPUS_PATH = f"projects/{_PROJECT_ID}/locations/{_LOCATION}/ragCorpora/"

# GoogleHandler.

class GoogleHandler(bc.LLMHandler):
    # Client is a singleton that requires close()...
    _client = None
    # ...and is protected by a lock.
    _lock = threading.Lock()

    @classmethod
    def get_client(cls):
        with cls._lock:
            if cls._client is None:
                cls._client = genai.Client(vertexai=True, project=_PROJECT_ID, location=_LOCATION)
            return cls._client

    @override
    @classmethod
    async def close(cls):
        with cls._lock:
            if cls._client:
                cls._client.close()
                cls._client = None

    @override
    async def call(self, input_text: str) -> tuple[Any, bc.CallDetailsType, bc.UsageType]:
        model_name, tools = self.model_info.name, []
        
        if self.model_info.prompt_id:
            model_name, rag_id = self.model_info.prompt_id.split(":")
            tools.append(types.Tool(retrieval=types.Retrieval(vertex_rag_store=types.VertexRagStore(
                rag_resources=[types.VertexRagStoreRagResource(rag_corpus=_RAG_CORPUS_PATH + rag_id)],
                similarity_top_k=20))))
        
        if self.web_search:
            tools.append(types.Tool(google_search=types.GoogleSearchRetrieval()))
        
        content = types.Content(role="user", parts=[types.Part.from_text(text=input_text)])
        config = types.GenerateContentConfig(
            tools=tools,
            system_instruction=[types.Part.from_text(text=self.system_prompt)] if self.system_prompt is not None else None )

        if self.verbose:
            print(content)

        response = await asyncio.to_thread(
            GoogleHandler.get_client().models.generate_content,
            model=model_name,
            contents=[content],
            config=config,
        )

        candidate = response.candidates[0]
        text = bc.LLMHandler.strip_code_fences(candidate.content.parts[0].text.strip())
        result = self.parse_type.model_validate_json(text) if self.parse_type else text
        links = self.get_web_search_links(candidate)
        usage = response.usage_metadata
        out_tokens = usage.candidates_token_count + (usage.thoughts_token_count if usage.thoughts_token_count else 0)
        usage_tuple = (usage.prompt_token_count, out_tokens)
        
        if self.verbose:
            print(f"result: {result}\nlinks: {links}")
        
        return result, links, usage_tuple

    def get_web_search_links(self, candidate: types.Candidate) -> bc.CallDetailsType:
        if not self.web_search:
            return None

        metadata = candidate.grounding_metadata
        links = {
            f'web_search_calls': [chunk.web.uri for chunk in metadata.grounding_chunks],
            f'web_search_queries': metadata.web_search_queries,
        } if metadata and metadata.grounding_chunks else None

        if links and 'web_search_calls' in links:
            print(f"DEBUG: Found {len(links['web_search_calls'])} web search links.")
        elif self.web_search:
            print(f"WARNING: web_search is True but no links were returned.")

        return links

# Google models registry.

_GOOGLE_MODELS = [
    # ModelInfo('name', prompt_id, input_cost, output_cost, context_length, direct_class, tags),
    # TEMPLATE:
    # ModelInfo('', None, 0.0, 0.0, 0, None, {''}),
    # Google models: https://openrouter.ai/provider/google-ai-studio
    bc.ModelInfo('gemini-2.0-flash-001', None, 0.10, 0.40, 1_050_000, GoogleHandler, False, {'google', 'fast', 'old'}),
    bc.ModelInfo('gemini-2.5-flash-lite', None, 0.10, 0.40, 1_050_000, GoogleHandler, False, {'google', 'fast'}), 
    bc.ModelInfo('gemini-2.5-flash', None, 0.30, 2.50, 1_050_000, GoogleHandler, True, {'google', 'fast'}),
    bc.ModelInfo('gemini-2.5-pro', None,1.25, 10.00, 1_050_000, GoogleHandler, True, {'google'}),
    bc.ModelInfo('gemini-3-flash-preview', None, 0.50, 4.00, 1_050_000, GoogleHandler, True, {'google', 'fast'}), # ?
    bc.ModelInfo('gemini-3-pro-preview', None, 2.50, 12.00, 1_050_000, GoogleHandler, True, {'google'}), # ?
    # Google Vertex AI models: 
    # "googlevertexai" models are handled directly.
    bc.ModelInfo('rag-default-gemini-2.5-flash', 'gemini-2.5-flash:7991637538768945152', 
                 0.30, 2.50, 1_050_000, GoogleHandler, False, {'google', 'prompt'}),
    bc.ModelInfo('rag-default-gemini-2.5-pro', 'gemini-2.5-pro:7991637538768945152', 
                 1.25, 10.00, 1_050_000, GoogleHandler, False, {'google', 'prompt'}),
    bc.ModelInfo('rag-llmparser-gemini-2.5-flash', 'gemini-2.5-flash:4532873024948404224', 
                 0.30, 2.50, 1_050_000, GoogleHandler, False, {'google', 'prompt'}),
    bc.ModelInfo('rag-llmparser-gemini-2.5-pro', 'gemini-2.5-pro:4532873024948404224', 
                 1.25, 10.00, 1_050_000, GoogleHandler, False, {'google', 'prompt'}),
]

bc.Models._MODEL_REGISTRY += _GOOGLE_MODELS

# Main test functions.

async def main_test():
    print("===== async_google_prompt.main_test() =====")

    # Test web search.
    for model in _GOOGLE_MODELS:
        if model.has_web_search:
            print(f"\n--- Testing model: {model.name} (web_search=True) ---")
            handler = bc.Models().by_name(model.name).create_handler(web_search=True)
            await bc._test_call_handler(handler, ["Give me the second news item from news.ycombinator.com right now?"])

    # Test with model default system prompt and web search.
    handler = bc.Models().by_name('gemini-2.5-flash').create_handler(
        system_prompt=bc._DEFAULT_SYSTEM_PROMPT, web_search=True, parse_type=bc.ListOfStrings)
    await bc._test_call_handler(handler, bc._TEST_QUESTIONS)

    # Test prompts.
    handler = bc.Models().by_name('rag-default-gemini-2.5-flash').create_handler(
        system_prompt=bc._DEFAULT_SYSTEM_PROMPT, web_search=False, parse_type=bc.ListOfStrings)
    await bc._test_call_handler(handler, bc._TEST_QUESTIONS)

if __name__ == "__main__":
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")
        
    asyncio.run(main_test())
