import asyncio
import json
import dotenv

from typing import Tuple
from pydantic import BaseModel
from google import genai
from google.genai import types

# Google Vertex AI settings:
LOCATION = "europe-west4"
PROJECT_ID = "gen-lang-client-0658217610"
RAG_CORPUS_PATH = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/"

PROMPT = """Answer a coding question related to GemBox Software .NET components.
Return a JSON object with a 'completions' array containing only the code strings that should replace the ??? marks, in order. 
Completions array should not contain any extra whitespace as results will be used for string comparison.

Example question: 
How do you set the value of cell A1 to Hello?
worksheet.Cells[???].??? = ???;
Your response:
{'completions': ['A1', 'Value', 'Hello']}

Below is the question and masked code. Return only the JSON object with no explanations, comments, or additional text.
"""


class GeminiPromptAgent:
  def __init__(self, model_name: str, rag_id: str):
    self.client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    self.model_name = model_name
    if rag_id=="none":
      self.tools = []
    else:
      self.tools = [types.Tool(retrieval=types.Retrieval(
        vertex_rag_store=types.VertexRagStore(
          rag_resources=[types.VertexRagStoreRagResource(rag_corpus=RAG_CORPUS_PATH + rag_id)],
          similarity_top_k=20,
        )
      ))]

  def _config(self) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
      tools=self.tools,
      system_instruction=[types.Part.from_text(text=PROMPT)],
    )

  async def run(self, input: str):
    def _call():
      return self.client.models.generate_content(
        model=self.model_name,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=input)])],
        config=self._config(),
      )

    return await asyncio.to_thread(_call)

  def response_2_results_tokens(self, response) -> Tuple[list[str], int, int]:
    text = ""
    results: list[str] = []
    try:
      text = response.candidates[0].content.parts[0].text
      candidate = text.strip()
      if text.startswith("```json") and text.endswith("```"):
        candidate = text[len("```json"):-len("```")]
      # Accept both single-quoted and standard JSON
      if candidate and candidate[0] == "{" and "'" in candidate and '"' not in candidate:
        candidate = candidate.replace("'", '"')
      obj = json.loads(candidate)
      results = obj.get("completions", [])
    except Exception:
      results = []

    usage = getattr(response, "usage_metadata", {}) or {}
    return results, usage.prompt_token_count, usage.candidates_token_count+usage.thoughts_token_count


questions = [
  "How to set value of A1 to 'Abracadabra'?",
  # "How to format B2 to bold?",
  # "How to print sheet?"
]

# For local testing:
MODEL_NAME = "gemini-2.5-flash"
RAG_ID = "4611686018427387904"

async def main():
  dotenv.load_dotenv()
  assert dotenv.dotenv_values().values(), ".env file not found or empty"

  agent = GeminiPromptAgent(MODEL_NAME, RAG_ID)
  async_responses = [agent.run(q) for q in questions]
  responses = await asyncio.gather(*async_responses)

  for res in responses:
    results, input_tokens, output_tokens = agent.response_2_results_tokens(res)
    print(f"\nResults: {results}\nInput tokens: {input_tokens}\nOutput tokens: {output_tokens}")


if __name__ == "__main__":
  asyncio.run(main())


