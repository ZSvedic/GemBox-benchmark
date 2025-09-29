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
        questions = [
            QuestionData.model_validate_json(line.strip())
            for line in file
            if line.strip()
        ]
        print(f"Loaded {len(questions)} questions from {file_path}")
        return questions