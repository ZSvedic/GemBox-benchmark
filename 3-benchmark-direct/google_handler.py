# Python stdlib.
import asyncio
import dataclasses as dc
import threading
from typing import Any, override

# Third-party.
import dotenv
from google import genai
from google.genai import types

# Local modules.
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

    @override
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
    @classmethod
    def provider_name(cls) -> str:
        return "Google"

    @override
    async def call(self, input_text: str) -> tuple[Any, bc.CallDetailsType, bc.UsageType]:
        tools = []
        model_name= self.model_info.name
        
        if self.model_info.prompt_id:
            model_name, rag_id = self.model_info.prompt_id.split(":")
            tools.append(types.Tool(retrieval=types.Retrieval(vertex_rag_store=types.VertexRagStore(
                rag_resources=[types.VertexRagStoreRagResource(rag_corpus=_RAG_CORPUS_PATH + rag_id)],
                similarity_top_k=20))))
        
        if self.web:
            if self.model_info.web is False:
                raise ValueError(f"Model {model_name} does not support web search.")
            elif self.include_domains:
                print("WARNING: Domain search is not supported in GoogleHandler.")
            else:
                tools.append(types.Tool(google_search=types.GoogleSearchRetrieval()))
        
        content = types.Content(role="user", parts=[types.Part.from_text(text=input_text)])
        config = types.GenerateContentConfig(
            tools=tools,
            system_instruction=[types.Part.from_text(text=self.system_ins)] if self.system_ins is not None else None )

        if self.verbose:
            print(f"=== CONFIG:\n{config}\n\n=== CONTENT:\n{content}")

        response = await asyncio.to_thread(
            GoogleHandler.get_client().models.generate_content,
            model=model_name,
            contents=[content],
            config=config,
        )

        candidate = response.candidates[0]
        text = GoogleHandler.strip_code_fences(candidate.content.parts[0].text.strip())
        result = self.parse_type.model_validate_json(text) if self.parse_type else text
        links = self.get_web_search_links(candidate)
        usage = response.usage_metadata
        out_tokens = usage.candidates_token_count + (usage.thoughts_token_count if usage.thoughts_token_count else 0)
        usage_tuple = (usage.prompt_token_count, out_tokens)
        
        if self.verbose:
            print(f"result: {result}\nlinks: {links}")
        
        return result, links, usage_tuple

    def get_web_search_links(self, candidate: types.Candidate) -> bc.CallDetailsType:
        if not self.web:
            return None

        metadata = candidate.grounding_metadata
        links = {
            f'web_search_calls': [chunk.web.uri for chunk in metadata.grounding_chunks],
            f'web_search_queries': metadata.web_search_queries,
        } if metadata and metadata.grounding_chunks else None

        if links and 'web_search_calls' in links:
            print(f"DEBUG: Found {len(links['web_search_calls'])} web search links.")
        elif self.web:
            print(f"WARNING: web_search is True but no links were returned.")

        return links

# Google models.

_base = bc.ModelInfo(handler=GoogleHandler, context_len=1_050_000, tags={'google'})
_base_rag = bc.ModelInfo(handler=GoogleHandler, tags={'google', 'rag'})

_GOOGLE_MODELS = [
    # Google models: https://openrouter.ai/provider/google-ai-studio
    dc.replace(_base, name='gemini-2.0-flash-001',      in_usd=0.10, out_usd= 0.40, web=False),
    dc.replace(_base, name='gemini-2.5-flash-lite',     in_usd=0.10, out_usd= 0.40, web=False), 
    dc.replace(_base, name='gemini-2.5-flash',          in_usd=0.30, out_usd= 2.50, web=True),
    dc.replace(_base, name='gemini-2.5-pro',            in_usd=1.25, out_usd=10.00, web=True),
    dc.replace(_base, name='gemini-3-flash-preview',    in_usd=0.50, out_usd= 3.00, web=True), # ?
    dc.replace(_base, name='gemini-3-pro-preview',      in_usd=2.50, out_usd=12.00, web=True), # ?
    # Google Vertex AI models.
    dc.replace(_base_rag, name='rag-default-gemini-2.5-flash', prompt_id='gemini-2.5-flash:7991637538768945152',
               in_usd=0.30, out_usd= 2.50, web=False),
    dc.replace(_base_rag, name='rag-default-gemini-2.5-pro', prompt_id='gemini-2.5-pro:7991637538768945152',
               in_usd=1.25, out_usd=10.00, web=False),
    dc.replace(_base_rag, name='rag-llmparser-gemini-2.5-flash', prompt_id='gemini-2.5-flash:4532873024948404224',
               in_usd=0.30, out_usd= 2.50, web=False),
    dc.replace(_base_rag, name='rag-llmparser-gemini-2.5-pro', prompt_id='gemini-2.5-pro:4532873024948404224',
               in_usd=1.25, out_usd=10.00, web=False),
]

bc.Models._MODEL_REGISTRY += _GOOGLE_MODELS

# Main test functions.

async def main_test():
    print("\n===== google_handler.main_test() =====")

    # Test with model default system instructions and web search.
    handler = bc.Models().by_name('gemini-2.5-flash').create_handler(
        system_ins=bc.DEFAULT_SYSTEM_INS, web=True, parse_type=bc.ListOfStrings, verbose=True)
    await bc._test_call_handler(handler, bc.TEST_QUESTIONS)

    # # Test RAG.
    # handler = bc.Models().by_name('rag-default-gemini-2.5-flash').create_handler(
    #     system_prompt=bc._DEFAULT_SYSTEM_PROMPT, web=False, parse_type=bc.ListOfStrings)
    # await bc._test_call_handler(handler, bc._TEST_QUESTIONS)

if __name__ == "__main__":
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")
        
    asyncio.run(main_test())
