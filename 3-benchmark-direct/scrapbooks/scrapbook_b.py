import dotenv
if not dotenv.load_dotenv():
    raise FileExistsError(".env file not found or empty")

from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="gen-lang-client-0658217610", location="europe-west4")

grounding_tool = types.Tool(google_search=types.GoogleSearch())
config = types.GenerateContentConfig(tools=[grounding_tool])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What are the latest news?",
    config=config
)

candidate = response.candidates[0]
print(f"Text answer: {candidate.content.parts[0].text}")
print(f"Web search queries: {candidate.grounding_metadata.web_search_queries}")
for chunk in candidate.grounding_metadata.grounding_chunks:
    print(f"Chunk URI: {chunk.web.uri}")