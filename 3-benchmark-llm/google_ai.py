from google import genai
from google.genai import types

MODEL_NAME = "gemini-2.5-flash"
PROJECT_ID = "gen-lang-client-0658217610"
LOCATION = "europe-west3"
RAG_ID = "4611686018427387904"
RAG_CORPUS_PATH = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAG_ID}"

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

QUESTION = """How do you set the value of cell B1 to Hi?\nworksheet.Cells[???].??? = ???;"""

def generate():
  client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
  resp = client.models.generate_content(
    model=MODEL_NAME,
    contents=[types.Content(role="user", parts=[types.Part.from_text(text=QUESTION)])],
    config=types.GenerateContentConfig(
      tools=[types.Tool(retrieval=types.Retrieval(
        vertex_rag_store=types.VertexRagStore(
          rag_resources=[types.VertexRagStoreRagResource(rag_corpus=RAG_CORPUS_PATH)],
          similarity_top_k=20,
        )
      ))],
      system_instruction=[types.Part.from_text(text=PROMPT)],
    ),
  )
  print(resp.candidates[0].content.parts[0].text)

# def generate_old():
#   # Client.
#   client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
#   # Contents.
#   contents = [ types.Content(role="user", parts=[types.Part.from_text(text=QUESTION)]) ]
#   # Tools.
#   tools = [
#     types.Tool(
#       retrieval=types.Retrieval(
#         vertex_rag_store=types.VertexRagStore(
#           rag_resources=[types.VertexRagStoreRagResource(rag_corpus=RAG_CORPUS_PATH)],
#           similarity_top_k=20,
#         )
#       )
#     )
#   ]
#   # Generate content config.
#   content_config = types.GenerateContentConfig(
#     temperature = 1,
#     top_p = 0.95,
#     max_output_tokens = 65535,
#     tools = tools,
#     system_instruction=[types.Part.from_text(text=PROMPT)],
#     thinking_config=types.ThinkingConfig(thinking_budget=-1),
#   )
#   # Get a generator over content stream.
#   stream = client.models.generate_content_stream( 
#     model = MODEL_NAME,
#     contents = contents,
#     config = content_config
#   )
#   # Iterate over content stream.
#   for chunk in stream:
#     if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
#       print(chunk.candidates[0].content.parts[0].text, end="")

if __name__ == "__main__":
  generate()