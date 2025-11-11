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
    # Client is a singleton that requires close() is protected by a lock:
    _client = None
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
        tools = []
        if self.model_info.prompt_id:
            tools.append(types.Tool(retrieval=types.Retrieval(vertex_rag_store=types.VertexRagStore(
                rag_resources=[types.VertexRagStoreRagResource(rag_corpus=_RAG_CORPUS_PATH + self.model_info.prompt_id)],
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
            model=self.model_info.name,
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
        metadata = candidate.grounding_metadata
        links_dict = {
            f'web_search_calls': [chunk.web.uri for chunk in metadata.grounding_chunks],
            f'web_search_queries': metadata.web_search_queries,
        } if metadata and metadata.grounding_chunks else None
        if self.web_search and not links_dict and self.verbose:
            print(f"WARNING: Google({self.model_info.name}) web_search is True but no links were returned.")
        return links_dict

    @staticmethod
    def strip_code_fences(text: str) -> str:
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```") and lines[-1].strip() == "```":
                return "\n".join(lines[1:-1])
        return text

# Google models registry.

_GOOGLE_MODELS = [
    # ModelInfo('name', prompt_id, input_cost, output_cost, context_length, direct_class, tags),
    # TEMPLATE:
    # ModelInfo('', None, 0.0, 0.0, 0, None, {''}),
    # Google models: https://openrouter.ai/provider/google-ai-studio
    bc.ModelInfo('gemini-2.0-flash-001', None, 0.10, 0.40, 1_050_000, GoogleHandler, True, {'google', 'fast'}),
    bc.ModelInfo('gemini-2.5-flash-lite', None, 0.10, 0.40, 1_050_000, GoogleHandler, True, {'google', 'fast'}), 
    bc.ModelInfo('gemini-2.5-flash', None, 0.30, 2.50, 1_050_000, GoogleHandler, True, {'google', 'fast'}),
    bc.ModelInfo('gemini-2.5-pro', None,1.25, 10.00, 1_050_000, GoogleHandler, True, {'google', 'accurate'}),
    # Google Vertex AI models: 
    # "googlevertexai" models are handled directly.
    bc.ModelInfo('rag-gemini-2.5-flash', '6917529027641081856', 
                 0.30, 2.50, 1_050_000, GoogleHandler, False, {'google', 'prompt'}),
    bc.ModelInfo('rag-gemini-2.5-pro', '6917529027641081856', 
                 1.25, 10.00, 1_050_000, GoogleHandler, False, {'google', 'prompt'}),
]

bc.Models._MODEL_REGISTRY += _GOOGLE_MODELS

# Main test functions.

async def main_test():
    print("===== async_google_prompt.main_test() =====")
    
    # Test plain text response for question about today's news.
    handler = bc.Models().by_name('gemini-2.5-flash').create_handler(web_search=True)
    await bc._test_call_handler(handler, ["What are the latest tech news today, be concise?"])

    # Test with model default system prompt and web search.
    handler = bc.Models().by_name('gemini-2.5-flash').create_handler(system_prompt=bc._DEFAULT_SYSTEM_PROMPT, web_search=True, parse_type=bc.ListOfStrings)
    await bc._test_call_handler(handler, bc._TEST_QUESTIONS)

    # TODO: Test with prompt_id.

if __name__ == "__main__":
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")
        
    asyncio.run(main_test())