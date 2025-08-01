# uv: python>=3.9
# No external dependencies; uses only Python standard library (re, sys, collections.abc)

import sys
import re
import collections.abc as abc
import os
import json
import dataclasses as dc

USAGE = """
GemBox benchmark filter. Usage:

bench_filter.py <path-to-csharp-folder>

This script reads all C# (.cs) files in the specified folder and prints out filtered data from lines like:

// Question: How to set GemBox.Spreadsheet to use the free license?
// Mask: \\bSpreadsheetInfo\\b
// Mask: \\bSetLicense\\b
SpreadsheetInfo.SetLicense("FREE-LIMITED-KEY");

The script prints for each file:
****** <filename-without-extension> ******
Q: How to set GemBox.Spreadsheet to use the free license?
M: ???.???("FREE-LIMITED-KEY");
A: SpreadsheetInfo SetLicense

Subtotal for <filename-without-extension>: N rows

In other words, the script:
- Finds comment lines starting with "// Question:" and prints the rest of the line as the question.
- Reads one or more following comment lines starting with "// Mask:" and extracts the regex after "Mask:".
- Finds the first code line after that comment block.
- Applies all regex masks to the code line, replacing the matched part with ???.
- Prints the code line with found items replaced and prints "A: " followed by strings captured by each regex.
- Prints the category (filename without extension) once per file, demarcated with ******, and a subtotal of generated rows after each category.
- Prints all output to the console and writes a test.jsonl file with the same data in JSONL format (one object per line).
"""

QUESTION_PREFIX = '// Question:'
MASK_PREFIX = '// Mask:'

@dc.dataclass
class BenchBlock:
    category: str
    question: str
    masked_code: str
    answers: list[str]

    def __str__(self) -> str:
        return (f"Category: {self.category}\n"
                f"Q: {self.question}\n"
                f"M: {self.masked_code}\n"
                f"A: {' '.join(self.answers)}\n")

def parse_blocks(lines: list[str], category: str) -> abc.Iterator[BenchBlock]:
    """Yield BenchBlock instances with fields: category, question, masked_code, answers."""
    slines = [line.strip() for line in lines]
    ri = 0
    while ri < len(slines):
        if slines[ri].startswith(QUESTION_PREFIX):
            question = slines[ri][len(QUESTION_PREFIX):].lstrip()
            masks, mask_patterns = [], []
            ri += 1
            while ri < len(slines) and slines[ri].startswith(MASK_PREFIX):
                mask_pattern = slines[ri][len(MASK_PREFIX):].strip()
                masks.append(mask_pattern)
                try:
                    mask_patterns.append(re.compile(mask_pattern))
                except re.error as e:
                    print(f"Warning: Invalid regex '{mask_pattern}' in category '{category}' (line {ri+1}): {e}")
                    mask_patterns.append(None)
                ri += 1
            valid_patterns = [p for p in mask_patterns if p is not None]
            if valid_patterns:
                masked_code, answers = apply_masks(slines[ri], valid_patterns)
                yield BenchBlock(category, question, masked_code, answers)
        ri += 1

def apply_masks(code_line: str, mask_patterns: list[re.Pattern]) -> tuple[str, list[str]]:
    """Apply each regex mask to the code line, replacing the first match with ??? and collecting answers."""
    answers = []
    masked_code = code_line
    for pattern in mask_patterns:
        match = pattern.search(masked_code)
        if match:
            answers.append(match.group(0))
            masked_code = pattern.sub('???', masked_code, count=1)
        else:
            raise ValueError(f"No match found for {pattern} in {masked_code}")
    return masked_code, answers

def print_block(block: dict) -> None:
    """Print a block in the nice formatted way."""
    print(f"Q: {block['question']}")
    print(f"M: {block['masked_code']}")
    print(f"A: {' '.join(block['answers'])}")
    print()

def main(dir_path: str) -> None:
    jsonl_lines = []
    for entry in os.listdir(dir_path):
        if entry.endswith('.cs'):
            csharp_path = os.path.join(dir_path, entry)
            category = os.path.splitext(entry)[0]
            with open(csharp_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            blocks = list(parse_blocks(lines, category))
            print(f"****** {category} ******")
            for block in blocks:
                print(block, end='')
                jsonl_lines.append(dc.asdict(block))
            print(f"Subtotal for {category}: {len(blocks)} rows\n")
    output_path = os.path.join(os.path.dirname(__file__), 'test.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in jsonl_lines:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Wrote {len(jsonl_lines)} lines to {output_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(USAGE)
        sys.exit(1)
    main(sys.argv[1])