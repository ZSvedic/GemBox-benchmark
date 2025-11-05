import dotenv
if not dotenv.load_dotenv():
    raise FileExistsError(".env file not found or empty")

from openai import OpenAI
client = OpenAI()

def get_web_search_links(response) -> set[str]:
    return {
        source.url 
        for item in response.output if item.type == "web_search_call" 
        for source in item.action.sources }

def get_web_search_links_print(response) -> set[str]:
    links = set()
    for index, item in enumerate(response.output):
        if item.type == "web_search_call":
            sources = item.action.sources
            print(f"web_search_call {index} has query '{item.action.query}' and has {len(sources)} sources:")
            for source in sources:
                print(f"    appending source: {source.url}")
                links.add(source.url)
    print(f"returning {len(links)} unique links")
    return links

response = client.responses.create(
  model="gpt-5-mini",
  reasoning={"effort": "low"},
  tools=[
      {
          "type": "web_search",
        #   "filters": {
        #       "allowed_domains": [
        #           "pubmed.ncbi.nlm.nih.gov",
        #           "clinicaltrials.gov",
        #           "www.who.int",
        #           "www.cdc.gov",
        #           "www.fda.gov",
        #       ]
        #   },
      }
  ],
  tool_choice="auto",
  include=["web_search_call.action.sources"],
  input="Please perform a web search on how semaglutide is used in the treatment of diabetes.",
)

print(response.output_text)
links = get_web_search_links(response)
print(links)
print('end of links') 