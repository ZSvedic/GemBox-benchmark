public class Program
{
    public static string ExtractCodeBlock(string llmResponse)
    {
        if (string.IsNullOrEmpty(llmResponse)) return "";

        const string start = "```csharp\n";
        const string end = "```";

        int startIdx = llmResponse.IndexOf(start);
        if (startIdx == -1) return "";

        int codeStart = startIdx + start.Length;
        int endIdx = llmResponse.IndexOf(end, codeStart);
        
        return endIdx == -1 ? "" : llmResponse.Substring(codeStart, endIdx - codeStart);
    }

    static bool RunTest(string name, string input, string expected)
    {
        var result = ExtractCodeBlock(input);
        bool pass = result == expected;
        Console.WriteLine($"{(pass ? "✓" : "✗")} {name}");
        if (!pass) Console.WriteLine($"  Expected: {expected}\n  Got: {result}");
        return pass;
    }

    static void Main()
    {
        Console.WriteLine($".NET {Environment.Version}\n");

        int passed = 0;
        passed += RunTest("Valid code block", "Code:\n```csharp\nvar x = 1;\n```\nDone.", "var x = 1;\n") ? 1 : 0;
        passed += RunTest("Extra content", "Text\n```csharp\nreturn 42;\n```\nMore text", "return 42;\n") ? 1 : 0;
        passed += RunTest("Multiple blocks", "```csharp\nint a;\n```\n```csharp\nint b;\n```", "int a;\n") ? 1 : 0;
        passed += RunTest("No code block", "Just plain text.", "") ? 1 : 0;
        passed += RunTest("Empty block", "```csharp\n```", "") ? 1 : 0;
        passed += RunTest("Wrong language", "```python\nprint()\n```", "") ? 1 : 0;
        passed += RunTest("Empty input", "", "") ? 1 : 0;
        passed += RunTest("Backticks in code", "```csharp\nvar s = \"`code`\";\n```", "var s = \"`code`\";\n") ? 1 : 0;

        Console.WriteLine($"\n{passed}/8 tests passed");
        Environment.Exit(passed == 8 ? 0 : 1);
    }
}
