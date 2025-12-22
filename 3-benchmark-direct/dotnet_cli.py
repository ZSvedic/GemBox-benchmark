import shutil
import subprocess
import pathlib
import shlex
import dataclasses as dc
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

def _run_cli(cmd: str, d: pathlib.Path, capture: bool):
    params = shlex.split(cmd)
    if capture:
        return subprocess.run(params, cwd=d, check=False, 
                              capture_output=True, text=True)
    else:
        return subprocess.run(params, cwd=d, check=True, 
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
@dc.dataclass(frozen=True)
class CSCompileRunResult:
    compiler_errors: int
    compiler_warnings: int
    compiler_output: str
    run_returncode: int
    run_output: str
    run_files: str

def cs_compile_execute(code: str, dir_name: str) -> CSCompileRunResult:
    '''Compiles the provided C# code in a predefined CS Project dir.'''
    
    # Create a new directory.
    d = pathlib.Path("scrapbooks") / dir_name
    d.mkdir(parents=False, exist_ok=False)

    # Create a new console project and add GemBox.Spreadsheet package.
    _run_cli("dotnet new console --framework net10.0", d, False)
    _run_cli("dotnet add package GemBox.Spreadsheet --version 2025.12.105", d, False)

    # Write the provided code into that project.
    (d / "Program.cs").write_text(code)

    # Get starting list of files.
    files_before = {p.name for p in d.iterdir() if p.is_file()}

    # Publish the project (forces compilation).
    compiler_proc = _run_cli("dotnet publish -c Release", d, True)
    compiler_lines = _clean_lines(compiler_proc.stdout, str(d.resolve()))

    if compiler_proc.returncode == 0:
        # Run the published executable.
        exe_rel = next((d / "bin").rglob(f"publish/{d.name}")).relative_to(d)
        run_proc = _run_cli(str(exe_rel), d, True)
    else:
        run_proc = subprocess.CompletedProcess(args=[], returncode=-1, stdout='', stderr='')

    # Clean up temp directories.
    shutil.rmtree(d / "bin", ignore_errors=True)
    shutil.rmtree(d / "obj", ignore_errors=True)

    # Ending list of files and diff.
    files_after = {p.name for p in d.iterdir() if p.is_file()}
    diff_files = sorted(str(f) for f in files_after - files_before)

    # Return number of warnings, errors, and the cleaned stdout.
    return CSCompileRunResult(
        compiler_errors = sum(1 for line in compiler_lines if ": error " in line),
        compiler_warnings = sum(1 for line in compiler_lines if ": warning " in line),
        compiler_output = '\n'.join(compiler_lines),
        run_returncode = run_proc.returncode,
        run_output = run_proc.stdout + run_proc.stderr,
        run_files = ', '.join(diff_files),
    )

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

_TEST_RUNTIME_FAIL = """
class Program {
    static void Main() {
        Console.WriteLine("This will fail at runtime.");
        int x = 0;
        int y = 5 / x;
    }
}"""

def main_test():
    print("\n===== dotnet_compile.main_test() =====")

    # Test code with 1 warning.
    run_res = cs_compile_execute(_TEST_1_WARNING, "test_1_warning")
    pprint(run_res)
    assert run_res.compiler_warnings == 1, "FAIL: Expected 1 warning."

    # Test code with 2 errors.
    run_res = cs_compile_execute(_TEST_2_ERRORS, "test_2_errors")
    pprint(run_res)
    assert run_res.compiler_errors == 2, "FAIL: Expected 2 errors."

    # Test code that compiles but executable fails.
    run_res = cs_compile_execute(_TEST_RUNTIME_FAIL, "test_runtime_fail")
    pprint(run_res)
    assert run_res.compiler_errors == 0, "FAIL: Expected 0 errors."
    assert run_res.run_returncode != 0, "FAIL: Expected non-zero return code."

if __name__ == "__main__":
    main_test()