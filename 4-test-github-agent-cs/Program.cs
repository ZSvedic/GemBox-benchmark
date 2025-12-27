using System;

public class Program
{
    /// <summary>
    /// Extracts the first C# code block from an LLM text response.
    /// </summary>
    /// <param name="llmResponse">The LLM text response containing code blocks</param>
    /// <returns>The code inside the first C# code block, or empty string if not found</returns>
    public static string ExtractCodeBlock(string llmResponse)
    {
        if (string.IsNullOrEmpty(llmResponse))
        {
            return string.Empty;
        }

        const string startMarker = "```csharp\n";
        const string endMarker = "```";

        int startIndex = llmResponse.IndexOf(startMarker);
        if (startIndex == -1)
        {
            return string.Empty;
        }

        // Move past the start marker
        int codeStartIndex = startIndex + startMarker.Length;

        // Find the closing ```
        int endIndex = llmResponse.IndexOf(endMarker, codeStartIndex);
        if (endIndex == -1)
        {
            return string.Empty;
        }

        // Extract the code between markers
        string code = llmResponse.Substring(codeStartIndex, endIndex - codeStartIndex);
        return code;
    }

    static void Main(string[] args)
    {
        Console.WriteLine($"Running extract_code_block tests on .NET {System.Environment.Version}");
        Console.WriteLine();

        int testsPassed = 0;
        int totalTests = 0;

        // Test 1: Valid code block with csharp tag
        totalTests++;
        Console.WriteLine("Test 1: Valid code block with csharp tag");
        string test1Input = "Here is some code:\n```csharp\nConsole.WriteLine(\"Hello\");\n```\nThat's the code.";
        string test1Expected = "Console.WriteLine(\"Hello\");\n";
        string test1Result = ExtractCodeBlock(test1Input);
        bool test1Pass = test1Result == test1Expected;
        Console.WriteLine($"  Input: {test1Input.Replace("\n", "\\n")}");
        Console.WriteLine($"  Expected: {test1Expected.Replace("\n", "\\n")}");
        Console.WriteLine($"  Got: {test1Result.Replace("\n", "\\n")}");
        Console.WriteLine($"  Result: {(test1Pass ? "PASS" : "FAIL")}");
        if (test1Pass) testsPassed++;
        Console.WriteLine();

        // Test 2: Code block with extra content before and after
        totalTests++;
        Console.WriteLine("Test 2: Code block with extra content before and after");
        string test2Input = "Let me help you with that.\n\nHere's the solution:\n```csharp\nvar x = 42;\nreturn x;\n```\n\nThis should work!";
        string test2Expected = "var x = 42;\nreturn x;\n";
        string test2Result = ExtractCodeBlock(test2Input);
        bool test2Pass = test2Result == test2Expected;
        Console.WriteLine($"  Input: {test2Input.Replace("\n", "\\n")}");
        Console.WriteLine($"  Expected: {test2Expected.Replace("\n", "\\n")}");
        Console.WriteLine($"  Got: {test2Result.Replace("\n", "\\n")}");
        Console.WriteLine($"  Result: {(test2Pass ? "PASS" : "FAIL")}");
        if (test2Pass) testsPassed++;
        Console.WriteLine();

        // Test 3: Multiple code blocks (should extract first one)
        totalTests++;
        Console.WriteLine("Test 3: Multiple code blocks (should extract first one)");
        string test3Input = "First block:\n```csharp\nint a = 1;\n```\nSecond block:\n```csharp\nint b = 2;\n```";
        string test3Expected = "int a = 1;\n";
        string test3Result = ExtractCodeBlock(test3Input);
        bool test3Pass = test3Result == test3Expected;
        Console.WriteLine($"  Input: {test3Input.Replace("\n", "\\n")}");
        Console.WriteLine($"  Expected: {test3Expected.Replace("\n", "\\n")}");
        Console.WriteLine($"  Got: {test3Result.Replace("\n", "\\n")}");
        Console.WriteLine($"  Result: {(test3Pass ? "PASS" : "FAIL")}");
        if (test3Pass) testsPassed++;
        Console.WriteLine();

        // Test 4: No code block present
        totalTests++;
        Console.WriteLine("Test 4: No code block present");
        string test4Input = "This is just plain text without any code blocks.";
        string test4Expected = "";
        string test4Result = ExtractCodeBlock(test4Input);
        bool test4Pass = test4Result == test4Expected;
        Console.WriteLine($"  Input: {test4Input}");
        Console.WriteLine($"  Expected: (empty string)");
        Console.WriteLine($"  Got: {(test4Result == "" ? "(empty string)" : test4Result)}");
        Console.WriteLine($"  Result: {(test4Pass ? "PASS" : "FAIL")}");
        if (test4Pass) testsPassed++;
        Console.WriteLine();

        // Test 5: Empty code block
        totalTests++;
        Console.WriteLine("Test 5: Empty code block");
        string test5Input = "Empty code:\n```csharp\n```\nNothing there.";
        string test5Expected = "";
        string test5Result = ExtractCodeBlock(test5Input);
        bool test5Pass = test5Result == test5Expected;
        Console.WriteLine($"  Input: {test5Input.Replace("\n", "\\n")}");
        Console.WriteLine($"  Expected: (empty string)");
        Console.WriteLine($"  Got: {(test5Result == "" ? "(empty string)" : test5Result)}");
        Console.WriteLine($"  Result: {(test5Pass ? "PASS" : "FAIL")}");
        if (test5Pass) testsPassed++;
        Console.WriteLine();

        // Test 6: Code block with different language tag (should not match)
        totalTests++;
        Console.WriteLine("Test 6: Code block with different language tag (should not match)");
        string test6Input = "Python code:\n```python\nprint('hello')\n```\nNot C#.";
        string test6Expected = "";
        string test6Result = ExtractCodeBlock(test6Input);
        bool test6Pass = test6Result == test6Expected;
        Console.WriteLine($"  Input: {test6Input.Replace("\n", "\\n")}");
        Console.WriteLine($"  Expected: (empty string)");
        Console.WriteLine($"  Got: {(test6Result == "" ? "(empty string)" : test6Result)}");
        Console.WriteLine($"  Result: {(test6Pass ? "PASS" : "FAIL")}");
        if (test6Pass) testsPassed++;
        Console.WriteLine();

        // Test 7: Null or empty input
        totalTests++;
        Console.WriteLine("Test 7: Null or empty input");
        string test7Input = "";
        string test7Expected = "";
        string test7Result = ExtractCodeBlock(test7Input);
        bool test7Pass = test7Result == test7Expected;
        Console.WriteLine($"  Input: (empty string)");
        Console.WriteLine($"  Expected: (empty string)");
        Console.WriteLine($"  Got: {(test7Result == "" ? "(empty string)" : test7Result)}");
        Console.WriteLine($"  Result: {(test7Pass ? "PASS" : "FAIL")}");
        if (test7Pass) testsPassed++;
        Console.WriteLine();

        // Test 8: Code block with backticks inside the code
        totalTests++;
        Console.WriteLine("Test 8: Code block with backticks inside the code (using markdown)");
        string test8Input = "Code with markdown:\n```csharp\nstring markdown = \"Use `backticks` for code\";\n```\nDone.";
        string test8Expected = "string markdown = \"Use `backticks` for code\";\n";
        string test8Result = ExtractCodeBlock(test8Input);
        bool test8Pass = test8Result == test8Expected;
        Console.WriteLine($"  Input: {test8Input.Replace("\n", "\\n")}");
        Console.WriteLine($"  Expected: {test8Expected.Replace("\n", "\\n")}");
        Console.WriteLine($"  Got: {test8Result.Replace("\n", "\\n")}");
        Console.WriteLine($"  Result: {(test8Pass ? "PASS" : "FAIL")}");
        if (test8Pass) testsPassed++;
        Console.WriteLine();

        // Summary
        Console.WriteLine("=".PadRight(50, '='));
        Console.WriteLine($"Tests passed: {testsPassed}/{totalTests}");
        Console.WriteLine($"Success rate: {(testsPassed * 100.0 / totalTests):F1}%");
        
        if (testsPassed == totalTests)
        {
            Console.WriteLine("All tests PASSED! ✓");
            Environment.Exit(0);
        }
        else
        {
            Console.WriteLine($"Some tests FAILED! ({totalTests - testsPassed} failures)");
            Environment.Exit(1);
        }
    }
}
