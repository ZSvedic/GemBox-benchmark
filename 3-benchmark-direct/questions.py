import json
import os
import tempfile

from pydantic import BaseModel

# Data structure for JSONL questions.
class QuestionData(BaseModel):
    category: str
    question: str
    masked_code: str
    answers: list[str]

def load_questions_from_jsonl(file_path: str) -> list[QuestionData]:
    """Load questions from a JSONL file using Pydantic for automatic parsing and validation."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return [
            QuestionData.model_validate_json(line.strip())
            for line in file
            if line.strip() 
        ]

def main_test():
    print("\n===== questions.main_test() =====")

    q1 = {"category": "calc", "question": "2+3=5", "masked_code": "3", "answers": ["3"]}
    q2 = {"category": "code", "question": "if __name__ ==", "masked_code": "__name__", "answers": ["__name__"]}
    
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(q1) + "\n\n" + json.dumps(q2))
        items = load_questions_from_jsonl(path)
        assert len(items) == 2, "FAIL: Wrong length."
        assert items[0].category == "calc" and items[1].masked_code == "__name__", "FAIL: Wrong data."
        print(f'PASS: Loaded {len(items)} questions from JSONL file.')
    finally:
        try: os.unlink(path)
        except Exception: pass

if __name__ == "__main__":
    main_test()