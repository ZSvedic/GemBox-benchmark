import subprocess
import pathlib
from pprint import pprint

def _clean_lines(dotnet_output: str, dir:str) -> list[str]:
    '''Internal helper to clean dotnet build output lines.'''
    lines = (dotnet_output
            .partition('\nBuild ')[0] # Keep only the text before "\nBuild succeeded/failed".
            .replace(dir, '...') # Hide temp dir paths.
            .replace('  Determining projects to restore...\n', '') # Remove.
            .replace('  All projects are up-to-date for restore.\n', '') # Remove.
            .splitlines() # Split into lines.
            )
    
    # Now remove trailing [/private.../....csproj] and return.
    return [line.rsplit(' [', 1)[0] for line in lines]

def compile_csharp(code: str, dir: str = "CSConsoleApp") -> dict[str, int|str]:
    '''Compiles the provided C# code in a predefined CS Project dir.'''
    # Always use the same CS Project dir.
    d = pathlib.Path(dir)

    # Write the provided code
    (d / "Program.cs").write_text(code)

    # Build the project
    p = subprocess.run(['dotnet', 'build'], cwd=d, capture_output=True, text=True)

    # Clean the output lines
    lines = _clean_lines(p.stdout, str(d.resolve()))

    # Return number of warnings, errors, and the cleaned stdout.
    return {
        "n_warnings": sum(1 for line in lines if ": warning " in line),
        "n_errors": sum(1 for line in lines if ": error " in line),
        "stdout": '\n'.join(lines),
    }

# Main test.

_TEST_1_WARNING = """using GemBox.Spreadsheet;
class Program {
    static void Main() {
        SpreadsheetInfo.SetLicense("FREE-LIMITED-KEY");
        var ef = new ExcelFile();
        var i = 8;
        var ws = ef.Worksheets.Add("Sheet1").Cells[0, 0].Value = "Hello, GemBox!";
        ef.Save("Output.xlsx");
        Console.WriteLine("Hello");
    }
}"""

_TEST_2_ERRORS = """
class Program {
    static void Main() {
        int i = "string instead of int"; // Type error here
        undeclaredVariable = 5; // Use of undeclared variable
    }
}"""

def main_test():
    print("\n===== dotnet_compile.main_test() =====")

    compile_result = compile_csharp(_TEST_1_WARNING)
    pprint(compile_result)
    assert compile_result["n_warnings"] == 1, "FAIL: Expected 1 warning."

    compile_result = compile_csharp(_TEST_2_ERRORS)
    pprint(compile_result)
    assert compile_result["n_errors"] == 2, "FAIL: Expected 2 errors."

if __name__ == "__main__":
    main_test()