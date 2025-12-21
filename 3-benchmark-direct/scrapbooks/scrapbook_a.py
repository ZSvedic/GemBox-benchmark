def extract_code_block(text: str) -> str:
    try:
        block = text.split("```", 2)[1]
        return block.split("\n", 1)[1].strip()
    except IndexError:
        return text.strip()


tests = [
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
