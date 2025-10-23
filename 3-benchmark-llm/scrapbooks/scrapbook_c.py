# Bug reproduction: web search tool is not supported in @benchmark_llm.py. 

import dotenv
import re
from pydantic import BaseModel
from pydantic_ai import Agent, WebSearchTool

PYDANTIC_AI_MODEL = 'openai-responses:gpt-5-mini' # 'openai-responses:gpt-4.1'
QUESTION = 'Which link from https://svedic.org/about page points to a "space simulation game"?'

class CodeCompletion(BaseModel):
    completions: list[str]

from pydantic_ai.providers.openrouter import OpenRouterProvider

# Works if using builtin_tools=[WebSearchTool()].
def get_agent_works(model: str) -> Agent:
    return Agent(model, output_type=CodeCompletion, builtin_tools=[WebSearchTool()])

from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

# Fails if using OpenAIResponsesModelSettings(openai_builtin_tools=[{"type": "web_search","search_context_size": "medium"}]).
# pydantic_ai.exceptions.ModelHTTPError: status_code: 400, model_name: openai-responses:gpt-5-nano, body: {'message': "Hosted tool 'web_search_preview' is not supported with openai-responses:gpt-5-nano.", 'type': 'invalid_request_error', 'param': 'tools', 'code': None}
def get_agent_fails_1(model_name: str) -> Agent:
    model = OpenAIResponsesModel(
        model_name, 
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort='medium',
            openai_builtin_tools=[{"type": "web_search", "search_context_size": "medium"}] 
            )
    )
    agent = Agent(model, output_type=CodeCompletion)
    return agent

# Actually it fails if using OpenAIResponsesModelSettings.
# pydantic_ai.exceptions.ModelHTTPError: status_code: 400, model_name: openai-responses:gpt-5-nano, body: {'message': "Hosted tool 'web_search_preview' is not supported with openai-responses:gpt-5-nano.", 'type': 'invalid_request_error', 'param': 'tools', 'code': None}
def get_agent_fails_2(model_name: str) -> Agent:
    tools = [WebSearchTool()]
    model = OpenAIResponsesModel(
        model_name, 
        settings=OpenAIResponsesModelSettings(
            openai_reasoning_effort='medium' 
            )
    )
    agent = Agent(model, output_type=CodeCompletion, builtin_tools=tools)
    return agent

from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings

# pydantic_ai.exceptions.UserError: WebSearchTool is not supported with `OpenAIChatModel` and model 'openai-responses:gpt-5-nano'. Please use `OpenAIResponsesModel` instead.
def get_agent_fails_3(model_name: str) -> Agent:
    tools = [WebSearchTool()]
    model = OpenAIChatModel(
        model_name, 
        settings=OpenAIChatModelSettings(
            openai_reasoning_effort='medium' 
            )
    )
    agent = Agent(model, output_type=CodeCompletion, builtin_tools=tools)
    return agent

def main():
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")

    print(f"\n========== Running {PYDANTIC_AI_MODEL} with web search tool... ==========")
    print(f"Question: {QUESTION}")
    agent = get_agent_works(PYDANTIC_AI_MODEL)
    result = agent.run_sync(QUESTION)
    result_text = result.output.completions[0]
    links = re.findall(r'https?://[^\s)>\]]+', result_text)
    print(f"Output: {result_text} \nLinks: {links}")

if __name__ == "__main__":
    main()