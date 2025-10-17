PROMPT = """Answer a coding question related to GemBox Software .NET components.
Return a JSON object with a 'completions' array containing only the code strings that should replace the ??? marks, in order. 
Completions array should not contain any extra whitespace as results will be used for string comparison.

Example question: 
How do you set the value of cell A1 to "Hello"?
worksheet.Cells[???].??? = ???;
Your response:
{'completions': ['"A1"', 'Value', '"Hello"']}

Below is the question and masked code. Return only the JSON object with no explanations, comments, or additional text. 
"""


print(PROMPT)