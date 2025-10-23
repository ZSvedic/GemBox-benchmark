# OpenAIResponsesModelSettings - Working Solution

## Problem Summary

You were unable to use `OpenAIResponsesModelSettings` to set OpenAI-specific settings like reasoning effort, web search tools, etc. The issue was that:

1. **Model Compatibility**: Not all models support the Responses API
2. **Incorrect Tool Format**: Using dictionaries instead of proper `WebSearchToolParam` objects
3. **API Limitations**: Some models don't support certain features even when using the Responses API

## Root Cause Analysis

The main issues were:

1. **Wrong Tool Format**: 
   ```python
   # ❌ WRONG - Using dictionary
   openai_builtin_tools=[{"type": "web_search", "search_context_size": "medium"}]
   
   # ✅ CORRECT - Using WebSearchToolParam
   from openai.types.responses import WebSearchToolParam
   openai_builtin_tools=[WebSearchToolParam(type='web_search', search_context_size='medium')]
   ```

2. **Model API Mismatch**: 
   - `openai-responses:gpt-4.1` doesn't fully support the Responses API
   - Some models only support the Chat API, not the Responses API

3. **Feature Limitations**:
   - Web search tools not supported on all models
   - Reasoning features not available on all models
   - "Encrypted content not supported" errors

## Working Solutions

### Solution 1: String Model (Most Reliable)
```python
def get_agent_working_string_model(model_name: str) -> Agent:
    """Use model string directly with Agent constructor."""
    return Agent(model_name, output_type=CodeCompletion, builtin_tools=[WebSearchTool()])
```

### Solution 2: Proper OpenAIResponsesModelSettings Usage
```python
def get_agent_working_responses_model(model_name: str) -> Agent:
    """Use OpenAIResponsesModel with proper settings and fallback."""
    from openai.types.responses import WebSearchToolParam
    
    try:
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
        print(f"Responses API failed: {e}")
        # Fallback to string model
        return Agent(model_name, output_type=CodeCompletion, builtin_tools=[WebSearchTool()])
```

## Key Insights

1. **Always Use Proper Objects**: Use `WebSearchToolParam` instead of dictionaries
2. **Implement Fallbacks**: Always have a fallback to the string model approach
3. **Model Compatibility**: Not all models support all features
4. **Error Handling**: Catch exceptions and provide meaningful fallbacks

## Available Settings

When using `OpenAIResponsesModelSettings`, you can configure:

- `openai_reasoning_effort`: 'low', 'medium', 'high'
- `openai_reasoning_summary`: 'concise', 'detailed'
- `openai_text_verbosity`: 'low', 'medium', 'high'
- `openai_truncation`: 'disabled', 'auto'
- `openai_send_reasoning_ids`: boolean
- `openai_builtin_tools`: List of `WebSearchToolParam`, `FileSearchToolParam`, `ComputerToolParam`

## Testing Results

✅ **String Model Method**: Works reliably with all models
✅ **Responses Model with Fallback**: Works when Responses API is supported
❌ **Responses Model without Fallback**: Fails when model doesn't support Responses API

## Recommendation

Use the **string model approach** for maximum reliability, or implement the **Responses model with fallback** for when you want to try advanced features but need reliability.

The working implementation is in `scrapbooks/working_openai_settings.py`.
