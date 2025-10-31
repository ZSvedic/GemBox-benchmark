from dataclasses import dataclass

@dataclass
class Metrics:
    name: str
    cost: float
    tokens: int
    time: float
    accuracy: float # None for complete failures.
    error_count: int = 0
    api_calls: int = 0

def get_accuracy(name: str, results: list[str], expected_answers: list[str] = None) -> float:
    """Validate model response against expected answers and return accuracy."""
    max_length = max(len(results), len(expected_answers))
    try:
        correct_completions = sum(
            str(res).strip() == exp.strip() 
            for res, exp in zip(results, expected_answers))
        return correct_completions / max_length
    except Exception as parse_ex:
        print(f"  Warning: {name} failed with exception: {repr(parse_ex)}")
        return 0.0

def summarize_metrics(name: str, metrics: list[Metrics]) -> Metrics:
    """Calculate a single summary metric from a list of Metrics."""
    assert len(metrics) > 0, "No metrics to summarize."

    accuracy_scores = [m.accuracy for m in metrics if m.accuracy is not None] # Exclude errors.
    accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else None
    
    return Metrics(
        name=name,
        cost=sum(m.cost for m in metrics),
        tokens=sum(m.tokens for m in metrics),
        time=sum(m.time for m in metrics),
        accuracy=accuracy,
        error_count=sum(m.error_count for m in metrics),
        api_calls=sum(m.api_calls for m in metrics),
    )

def print_metrics(name: str, metrics: list[Metrics]) -> None:
    """Print a list of Metrics entries."""
    print(f"\n==={name}===")
    for m in metrics:
        acc_str = f"{m.accuracy:.0%}" if m.accuracy is not None else "N/A"
        print(f"\t{m.name:32.32}\ttokens={m.tokens},\tcost=${m.cost:.6f},\ttime={m.time:.2f}s,\taccuracy={acc_str},\terrors={m.error_count}/{m.api_calls}")