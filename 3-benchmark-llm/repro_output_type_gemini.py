import asyncio
import dotenv
import time
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

class CodeCompletion(BaseModel):
    completions: list[str]

async def test_with_timeout(agent, prompt, timeout=15):
    """Test agent with timeout to fail faster"""
    try:
        start_time = time.time()
        result = await asyncio.wait_for(agent.run(prompt), timeout=timeout)
        duration = time.time() - start_time
        
        print(f'✓ SUCCESS in {duration:.2f}s')
        print('  type(output):', type(result.output))
        print('  output:', result.output)
        try:
            print('  usage:', result.usage())
        except Exception as e:
            print('  usage error:', e)
        return True
        
    except asyncio.TimeoutError:
        print(f'✗ TIMEOUT after {timeout}s')
        return False
    except Exception as e:
        duration = time.time() - start_time if 'start_time' in locals() else 0
        print(f'✗ ERROR after {duration:.2f}s: {type(e).__name__}: {e}')
        return False

async def main():
    # Load environment variables from parent directory .env
    dotenv.load_dotenv('../.env')
    
    if not dotenv.get_key('../.env', 'OPENROUTER_API_KEY'):
        print("ERROR: OPENROUTER_API_KEY not found in ../.env file")
        return

    prompt = (
        "Instructions: Return a CodeCompletion for the C# snippet below by filling just ??? placeholders in order. Return JSON object, not text. \n"
        "Question: How do you set the value of cell A1 to 'Hello'?\n"
        "C# snippet: \n"
        "worksheet.Cells[???].??? = ???;"
    )

    # Test different approaches
    for model_name in ['openai/gpt-4o-mini', 'google/gemini-2.5-flash']:
    # for model_name in ['openai:gpt-4o-mini', 'google-gla:gemini-2.5-flash']:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        model = OpenAIModel(model_name, provider=OpenRouterProvider())
        agent1 = Agent(model, output_type=CodeCompletion)
        # agent1 = Agent(model_name, output_type=CodeCompletion)
        success1 = await test_with_timeout(agent1, prompt)

        # model = OpenAIModel(model_name, provider=OpenRouterProvider())
        
        # # Approach 1: Try with output_type (current failing approach)
        # print(f"\n1. Testing with output_type=CodeCompletion:")
        # agent1 = Agent(model, output_type=CodeCompletion)
        # success1 = await test_with_timeout(agent1, prompt)
        
        # # Approach 2: Try without output_type (fallback)
        # print(f"\n2. Testing without output_type (text fallback):")
        # agent2 = Agent(model)
        # success2 = await test_with_timeout(agent2, prompt)
        
        # # Approach 3: Try with different output_type syntax
        # print(f"\n3. Testing with list[str] output_type:")
        # try:
        #     agent3 = Agent(model, output_type=list[str])
        #     success3 = await test_with_timeout(agent3, prompt)
        # except Exception as e:
        #     print(f"✗ Failed to create agent with list[str]: {e}")
        #     success3 = False
        
        # # Approach 4: Try with dict output_type
        # print(f"\n4. Testing with dict output_type:")
        # try:
        #     agent4 = Agent(model, output_type=dict)
        #     success4 = await test_with_timeout(agent4, prompt)
        # except Exception as e:
        #     print(f"✗ Failed to create agent with dict: {e}")
        #     success4 = False
        
        print(f"\nSummary for {model_name}:")
        print(f"  output_type=CodeCompletion: {'✓' if success1 else '✗'}")
        # print(f"  no output_type: {'✓' if success2 else '✗'}")
        # print(f"  output_type=list[str]: {'✓' if success3 else '✗'}")
        # print(f"  output_type=dict: {'✓' if success4 else '✗'}")

if __name__ == '__main__':
    asyncio.run(main())