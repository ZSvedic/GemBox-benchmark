import subprocess
import tempfile
import pathlib
from typing import Dict, Any
from pprint import pprint

def compile_csharp(code: str, framework: str = "net10.0") -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as d:
        d = pathlib.Path(d)

        # Create new console project
        params = ["dotnet", "new", "console", "--framework", framework]
        subprocess.run(params, cwd=d, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Write the provided code
        (d / "Program.cs").write_text(code)

        # Build the project
        params = ["dotnet", "build", "/p:TreatWarningsAsErrors=false"]
        p = subprocess.run(params, cwd=d, capture_output=True, text=True)

        # Parse warnings and errors from output
        warnings = []
        errors = []

        for line in (p.stdout + p.stderr).splitlines():
            if ": warning " in line:
                warnings.append(line.strip())
            elif ": error " in line:
                errors.append(line.strip())

        return {
            "success": p.returncode == 0,
            "warnings": warnings,
            "errors": errors,
            "stdout": p.stdout,
            "stderr": p.stderr,
        }


# Example usage
if __name__ == "__main__":
    code = """
using System;
class Program {
    static void Main() {
        Console.WriteLine("Hello")
    }
}
"""
    result = compile_csharp(code)

    for key in ['success', 'warnings', 'errors']:
        print(key.upper() + ":")
        r = result[key]
        pprint(r)