import asyncio
from typing import override

import dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

import base_classes as bc

# Google Vertex AI settings:
_LOCATION = "europe-west4"
_PROJECT_ID = "gen-lang-client-0658217610"
_RAG_CORPUS_PATH = f"projects/{_PROJECT_ID}/locations/{_LOCATION}/ragCorpora/"

class GoogleHandler(bc.LLMHandler):
    @override
    def __init__(
        self, 
        model_info: bc.ModelInfo, 
        *,
        system_prompt: str | None = None, 
        parse_type: type[BaseModel] | None = None,
        web_search: bool = False,
        verbose: bool = True): 

        self.model_info = model_info
        self.model_name = model_info.name
        if model_info.prompt_id:
            self.tools = [types.Tool(retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                rag_resources=[types.VertexRagStoreRagResource(rag_corpus=_RAG_CORPUS_PATH + model_info.prompt_id)],
                similarity_top_k=20,
            )
            ))]
        else:
            self.tools = []

        self.system_prompt = system_prompt
        self.parse_type = parse_type
        self.web_search = web_search
        self.verbose = verbose
        self.client = genai.Client(vertexai=True, project=_PROJECT_ID, location=_LOCATION)

    def _config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            tools=self.tools,
            system_instruction=[types.Part.from_text(text=self.system_prompt)])

    async def call(self, input: str) -> tuple[list[str], int, int]:
        content = types.Content(
            role="user", 
            parts=[types.Part.from_text(text=input)])
        config = types.GenerateContentConfig(
            tools=self.tools,
            system_instruction=[types.Part.from_text(text=self.system_prompt)])
        
        if self.verbose:
            print(content)

        response = await asyncio.to_thread(
            self.client.models.generate_content,
            model=self.model_info.name,
            contents=[content],
            config=config,
        )

        results = response.candidates[0].content.parts[0].text.strip()
        if self.parse_type:
            try:
                if results.startswith("```json") and results.endswith("```"):
                    results = results[len("```json"):-len("```")]
                if results[0] == "{" and "'" in results and '"' not in results:
                    results = results.replace("'", '"')
                results = self.parse_type.model_validate_json(results)
            except Exception as e:
                print(f"Error: {e}")

        usage = response.usage_metadata
        return results, usage.prompt_token_count, usage.candidates_token_count+usage.thoughts_token_count

_GOOGLE_MODELS = [
    # ModelInfo('name', prompt_id, input_cost, output_cost, context_length, direct_class, tags),
    # TEMPLATE:
    # ModelInfo('', None, 0.0, 0.0, 0, None, {''}),
    # Google models: https://openrouter.ai/provider/google-ai-studio
    bc.ModelInfo('gemini-2.0-flash-001', None, 0.10, 0.40, 1_050_000, GoogleHandler, {'google', 'fast'}),
    bc.ModelInfo('gemini-2.5-flash-lite', None, 0.10, 0.40, 1_050_000, GoogleHandler, {'google', 'fast'}), 
    bc.ModelInfo('gemini-2.5-flash', None, 0.30, 2.50, 1_050_000, GoogleHandler, {'google', 'fast'}),
    bc.ModelInfo('gemini-2.5-pro', None,1.25, 10.00, 1_050_000, GoogleHandler, {'google', 'accurate'}),
    # Google Vertex AI models: 
    # "googlevertexai" models are handled directly.
    bc.ModelInfo('rag-gemini-2.5-flash', '6917529027641081856', 0.30, 2.50, 1_050_000, GoogleHandler, {'google', 'prompt'}),
    bc.ModelInfo('rag-gemini-2.5-pro', '6917529027641081856', 1.25, 10.00, 1_050_000, GoogleHandler, {'google', 'prompt'}),
]

bc.Models._MODEL_REGISTRY += _GOOGLE_MODELS

async def call_handler(handler: GoogleHandler):
    async_responses = [handler.call(q) for q in bc._TEST_QUESTIONS]
    responses = await asyncio.gather(*async_responses)
    for results, input_tokens, output_tokens in responses:
        print(f"\nResults: {results}\nInput tokens: {input_tokens}\nOutput tokens: {output_tokens}")

async def test_main():
    if not dotenv.load_dotenv():
        raise FileNotFoundError(".env file not found or empty")

    handler = bc.Models().by_name('gemini-2.5-flash').create_handler(
        system_prompt=bc._DEFAULT_SYSTEM_PROMPT, parse_type=bc.ListOfStrings)

    await call_handler(handler)

    # TODO: Test with prompt_id.

if __name__ == "__main__":
    asyncio.run(test_main())