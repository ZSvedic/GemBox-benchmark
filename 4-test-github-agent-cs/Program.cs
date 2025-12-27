using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

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

    public static List<string> ExtractLinks(string llmResponse)
    {
        var links = new List<string>();
        if (string.IsNullOrWhiteSpace(llmResponse)) return links;

        static string Clean(string link) => link.TrimEnd('.', ',', ';', ':', ')', ']', '"', '\'');

        foreach (Match match in Regex.Matches(llmResponse, @"\[[^\]]*\]\((https?://[^)]+)\)"))
        {
            var link = Clean(match.Groups[1].Value);
            if (!links.Contains(link)) links.Add(link);
        }

        foreach (Match match in Regex.Matches(llmResponse, @"https?://\S+"))
        {
            var link = Clean(match.Value);
            if (!links.Contains(link)) links.Add(link);
        }

        return links;
    }

    public static string Shorten(string input, int charLimit)
    {
        if (string.IsNullOrEmpty(input)) return "";
        if (charLimit <= 0) return "";
        if (input.Length <= charLimit) return input;

        int halfLimit = charLimit / 2;
        string firstPart = input.Substring(0, halfLimit);
        string lastPart = input.Substring(input.Length - halfLimit);

        return $"{firstPart}\n...\n{lastPart}";
    }

    static bool RunTest(string name, string input, string expected)
    {
        var result = ExtractCodeBlock(input);
        bool pass = result == expected;
        Console.WriteLine($"{(pass ? "✓" : "✗")} {name}");
        if (!pass) Console.WriteLine($"  Expected: {expected}\n  Got: {result}");
        return pass;
    }

    static bool RunLinkTest(string name, string input, List<string> expected)
    {
        var result = ExtractLinks(input);
        bool pass = result.SequenceEqual(expected);
        Console.WriteLine($"{(pass ? "✓" : "✗")} {name}");
        if (!pass)
        {
            Console.WriteLine($"  Expected: [{string.Join(", ", expected)}]\n  Got: [{string.Join(", ", result)}]");
        }
        return pass;
    }

    static bool RunShortenTest(string name, string input, int charLimit, string expected)
    {
        var result = Shorten(input, charLimit);
        bool pass = result == expected;
        Console.WriteLine($"{(pass ? "✓" : "✗")} {name}");
        if (!pass)
        {
            Console.WriteLine($"  Expected: \"{expected.Replace("\n", "\\n")}\"\n  Got: \"{result.Replace("\n", "\\n")}\"");
        }
        return pass;
    }

    static void Main()
    {
        Console.WriteLine($".NET {Environment.Version}\n");

        int passed = 0;
        int total = 0;

        total += 8;
        passed += RunTest("Valid code block", "Code:\n```csharp\nvar x = 1;\n```\nDone.", "var x = 1;\n") ? 1 : 0;
        passed += RunTest("Extra content", "Text\n```csharp\nreturn 42;\n```\nMore text", "return 42;\n") ? 1 : 0;
        passed += RunTest("Multiple blocks", "```csharp\nint a;\n```\n```csharp\nint b;\n```", "int a;\n") ? 1 : 0;
        passed += RunTest("No code block", "Just plain text.", "") ? 1 : 0;
        passed += RunTest("Empty block", "```csharp\n```", "") ? 1 : 0;
        passed += RunTest("Wrong language", "```python\nprint()\n```", "") ? 1 : 0;
        passed += RunTest("Empty input", "", "") ? 1 : 0;
        passed += RunTest("Backticks in code", "```csharp\nvar s = \"`code`\";\n```", "var s = \"`code`\";\n") ? 1 : 0;

        total += 5;
        passed += RunLinkTest("Single link", "Visit https://example.com for more info.", new List<string> { "https://example.com" }) ? 1 : 0;
        passed += RunLinkTest("Multiple links", "Docs: https://a.com/docs and https://b.com/guide.", new List<string> { "https://a.com/docs", "https://b.com/guide" }) ? 1 : 0;
        passed += RunLinkTest("Markdown link", "Check [the guide](https://docs.example.com/page) please.", new List<string> { "https://docs.example.com/page" }) ? 1 : 0;
        passed += RunLinkTest("Trailing punctuation", "See https://site.com/test, then https://next.com/end.", new List<string> { "https://site.com/test", "https://next.com/end" }) ? 1 : 0;
        passed += RunLinkTest("Mixed formats", "Link: https://one.com [alt](https://two.com/path).", new List<string> { "https://two.com/path", "https://one.com" }) ? 1 : 0;

        total += 5;
        passed += RunShortenTest("Long text shortening", "This is a very long LLM response that contains a lot of text and needs to be shortened for display purposes.", 40, "This is a very long LLM respo\n...\nened for display purposes.") ? 1 : 0;
        passed += RunShortenTest("Short text no shortening", "Short text", 20, "Short text") ? 1 : 0;
        passed += RunShortenTest("Empty string", "", 10, "") ? 1 : 0;
        passed += RunShortenTest("Exact limit", "12345678901234567890", 20, "12345678901234567890") ? 1 : 0;
        passed += RunShortenTest("Zero char limit", "Some text here", 0, "") ? 1 : 0;

        Console.WriteLine($"\n{passed}/{total} tests passed");
        Environment.Exit(passed == total ? 0 : 1);
    }
}
