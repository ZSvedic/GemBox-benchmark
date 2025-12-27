def extract_code_block(text: str) -> str:
    try:
        block = text.split("```", 2)[1]
        code = block.split("\n", 1)[1]
        return code.strip()
    except Exception:
        return text.strip() 

tests = [
    "```csharp\nConsole.WriteLine(\"Hi\");\nConsole.WriteLine(\"again\");```",
    "no code here",
    "before ```python\nprint('hi')\n``` after",
    "```csharp\nConsole.WriteLine(\"Hi\");\n```",
    "```text\nhello\nworld\n```",
    "```onlyonefence\nunfinished",
    "```csharp\nConsole.WriteLine(\"Hi\");\n```\nSome text."
]

for t in tests:
    print("INPUT :", repr(t))
    print("OUTPUT:", repr(extract_code_block(t)))
    print()


 # Inside this project create a new 4-test-github-agent-cs project, what will be a C# project using .NET 10.0. That project should implement a custom extract_code_block method that gets and returns a string. Input is LLM text response, and output is a text inside 1st code block starting with "```csharp\n" and ending with "```". Create 4-8 test cases and run the program to test if it works.