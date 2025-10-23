# Calling the Pydantic AI with "openai-responses" OpenRouter prefix works with a web search tool.

import dotenv
if not dotenv.load_dotenv():
    raise FileExistsError(".env file not found or empty")

from pydantic_ai import Agent, WebSearchTool

agent = Agent('openai-responses:gpt-4.1', builtin_tools=[WebSearchTool()])

result = agent.run_sync('Give me a sentence with the biggest news in AI this week.')

print(result.output)
