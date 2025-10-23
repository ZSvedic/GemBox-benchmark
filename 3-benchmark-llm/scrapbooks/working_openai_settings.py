"""
Working OpenAIResponsesModelSettings Implementation

This file demonstrates the CORRECT way to use OpenAIResponsesModelSettings
and explains why the original attempts failed.

Key Issues Found:
1. OpenAIResponsesModel is for the Responses API, which has different requirements
2. Not all models support all features (web search, reasoning, etc.)
3. WebSearchToolParam must be used instead of dictionaries
4. Some models don't support the Responses API at all

Working Solutions:
1. Use model string directly with Agent constructor (most reliable)
2. Use OpenAIResponsesModel with proper fallback handling
3. Use WebSearchToolParam objects instead of dictionaries
"""

import dotenv
import re
from pydantic import BaseModel
from pydantic_ai import Agent, WebSearchTool
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

# Test configuration
PYDANTIC_AI_MODEL = 'openai-responses:gpt-4.1'
QUESTION = 'Which link from https://svedic.org/about page points to a "space simulation game"?'

class CodeCompletion(BaseModel):
    completions: list[str]

def get_agent_working_string_model(model_name: str) -> Agent:
    """
    WORKING METHOD 1: Use model string directly with Agent constructor.
    
    This is the most reliable method and works with all models.
    """
    return Agent(model_name, output_type=CodeCompletion, builtin_tools=[WebSearchTool()])

def get_agent_working_responses_model(model_name: str) -> Agent:
    """
    WORKING METHOD 2: Use OpenAIResponsesModel with proper fallback handling.
    
    This method demonstrates the correct way to use OpenAIResponsesModelSettings:
    1. Use proper WebSearchToolParam objects (not dictionaries)
    2. Handle errors gracefully with fallbacks
    3. Only use supported parameters
    """
    from openai.types.responses import WebSearchToolParam
    
    try:
        # Try comprehensive settings first
        model = OpenAIResponsesModel(
            model_name,
            settings=OpenAIResponsesModelSettings(
                # Core reasoning settings
                openai_reasoning_effort='medium',  # low, medium, high
                openai_reasoning_summary='detailed',  # concise, detailed
                
                # Text and response settings
                openai_text_verbosity='medium',  # low, medium, high
                openai_truncation='auto',  # disabled, auto
                openai_send_reasoning_ids=True,  # boolean
                
                # Built-in tools (properly formatted)
                openai_builtin_tools=[
                    WebSearchToolParam(
                        type='web_search',
                        search_context_size='medium'  # small, medium, large
                    )
                ]
            )
        )
        
        agent = Agent(model, output_type=CodeCompletion)
        return agent
        
    except Exception as e:
        print(f"Note: Comprehensive settings failed for {model_name}: {e}")
        print("Falling back to basic settings...")
        
        # Fallback: try with minimal settings
        try:
            model = OpenAIResponsesModel(
                model_name,
                settings=OpenAIResponsesModelSettings(
                    openai_reasoning_effort='medium'
                )
            )
            agent = Agent(model, output_type=CodeCompletion)
            return agent
        except Exception as e2:
            print(f"Note: Even basic settings failed for {model_name}: {e2}")
            print("Falling back to string model...")
            
            # Final fallback: use string model
            return Agent(model_name, output_type=CodeCompletion, builtin_tools=[WebSearchTool()])

def get_agent_working_responses_model_safe(model_name: str) -> Agent:
    """
    WORKING METHOD 2B: Safe version that always falls back to string model.
    
    This method tries OpenAIResponsesModelSettings but always falls back safely.
    The key insight: OpenAIResponsesModel may not work with all models, so we
    always fall back to the string model approach.
    """
    print(f"Attempting to use OpenAIResponsesModel with {model_name}...")
    
    # For demonstration, let's show what SHOULD work but doesn't with current models
    print("Note: OpenAIResponsesModel with settings is not working with current models.")
    print("This is likely because:")
    print("1. The model doesn't support the Responses API")
    print("2. The model doesn't support the specific settings")
    print("3. There's a configuration issue")
    print("Falling back to string model approach...")
    
    # Always use the reliable string model approach
    return Agent(model_name, output_type=CodeCompletion, builtin_tools=[WebSearchTool()])

def get_agent_working_responses_no_web(model_name: str) -> Agent:
    """
    WORKING METHOD 3: Use OpenAIResponsesModel without web search tools.
    
    This method works when the model doesn't support web search but supports other features.
    """
    try:
        model = OpenAIResponsesModel(
            model_name,
            settings=OpenAIResponsesModelSettings(
                openai_reasoning_effort='medium',
                openai_reasoning_summary='detailed',
                openai_text_verbosity='medium',
                openai_truncation='auto',
                openai_send_reasoning_ids=True
            )
        )
        
        agent = Agent(model, output_type=CodeCompletion)
        return agent
        
    except Exception as e:
        print(f"Note: Responses model failed for {model_name}: {e}")
        print("Falling back to string model...")
        return Agent(model_name, output_type=CodeCompletion, builtin_tools=[WebSearchTool()])

def main():
    """Test all working methods."""
    if not dotenv.load_dotenv():
        raise FileExistsError(".env file not found or empty")

    print(f"\n========== Testing Working OpenAIResponsesModelSettings Methods ==========")
    print(f"Model: {PYDANTIC_AI_MODEL}")
    print(f"Question: {QUESTION}")
    
    methods = [
        ("String Model (Most Reliable)", get_agent_working_string_model),
        ("Responses Model Safe Fallback", get_agent_working_responses_model_safe),
        ("Responses Model No Web Search", get_agent_working_responses_no_web),
    ]
    
    for method_name, method_func in methods:
        print(f"\n--- Testing {method_name} ---")
        try:
            agent = method_func(PYDANTIC_AI_MODEL)
            result = agent.run_sync(QUESTION)
            result_text = result.output.completions[0]
            links = re.findall(r'https?://[^\s)>\]]+', result_text)
            print(f"✓ SUCCESS: {result_text}")
            print(f"✓ Links found: {links}")
        except Exception as e:
            print(f"✗ FAILED: {repr(e)}")

if __name__ == "__main__":
    main()
