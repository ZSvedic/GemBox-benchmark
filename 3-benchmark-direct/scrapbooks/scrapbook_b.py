import subprocess
import tempfile
import pathlib
import itertools
from typing import Dict, Any
from pprint import pprint

def clean_lines(s: str, dir:str) -> list[str]:
    lines = (s
            .partition('\nBuild ')[0] # Keep only lines before "Build succeeded/failed"
            .replace(dir, '...') # Hide temp dir paths.
            .replace('  Determining projects to restore...\n', '') # Remove.
            .replace('  All projects are up-to-date for restore.\n', '') # Remove.
            .splitlines() # Split into lines.
            )
    
    # Now remove trailing [/private.../....csproj]
    return [line.rsplit(' [', 1)[0] for line in lines]

def compile_csharp(code: str,framework: str = "net10.0") -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as dir:
        d = pathlib.Path(dir)

        # Create new console project
        subprocess.run(["dotnet", "new", "console", "--framework", framework], 
                       cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Write the provided code
        (d / "Program.cs").write_text(code)

        # Build the project
        p = subprocess.run(['dotnet', 'build'], 
                           cwd=d, capture_output=True, text=True)

        lines = clean_lines(p.stdout, str(d))

        return {
            "n_warnings": sum(1 for line in lines if ": warning " in line),
            "n_errors": sum(1 for line in lines if ": error " in line),
            "stdout": '\n'.join(lines),
        }


# Example usage
if __name__ == "__main__":
    pprint(compile_csharp("""
using System;
using GemBox.Spreadsheet;
class Program {
    static void Main() {
        SpreadsheetInfo.SetLicense("FREE-LIMITED-KEY");
        var ef = new ExcelFile();
        var ws = ef.Worksheets.Add("Sheet1").Cells[0, 0].Value = "Hello, GemBox!";
        ef.Save("Output.xlsx");
        Console.WriteLine("Hello");
    }
}
"""))
