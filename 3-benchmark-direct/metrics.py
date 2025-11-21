from dataclasses import dataclass
from statistics import median

@dataclass
class Metrics:
    name: str               # Model or run name.
    cost_mdn: float         # Median cost per API call.
    tokens_mdn: int         # Median token count per API call.
    time_mdn: float         # Median time per API call.
    error_rate_mdn: float   # Median of error rates per API call.
    api_issues: int = 0     # Total count of API issues (e.g., timeouts, quota errors).
    api_calls: int = 1      # Total count of API calls.

def get_error_rate(name: str, results: list[str], expected_answers: list[str]) -> float:
    """Validates model response against expected answers and return error rate."""
    max_length = max(len(results), len(expected_answers))
    try:
        incorrect_completions = sum(
            str(res).strip() != exp.strip() 
            for res, exp in zip(results, expected_answers))
        return incorrect_completions / max_length
    except Exception as parse_ex:
        print(f"  Warning: {name} failed with exception: {repr(parse_ex)}")
        # If model returns unparsable results, count as 100% error rate.
        return 1.0

def summarize_metrics(name: str, metrics: list[Metrics]) -> Metrics:
    """Calculate a single summary metric from a list of Metrics."""
    assert (len_all:=len(metrics)) > 0, "No metrics to summarize."

    # 2025-11-20 Timeouts and quota errors are now ignored in calculations.
    valid_metrics = [m for m in metrics if m.api_issues < m.api_calls]

    if (len_valid:=len(valid_metrics)) > 0:
        cost_per_call = median(m.cost_mdn for m in valid_metrics)
        tokens_per_call = int(median(float(m.tokens_mdn) for m in valid_metrics))
        time_per_call = median(m.time_mdn for m in valid_metrics)
        error_rate = median(m.error_rate_mdn for m in valid_metrics)
    else:
        cost_per_call = time_per_call = error_rate = 0.0
        tokens_per_call = 0

    api_issues = sum(m.api_issues for m in metrics)
    api_calls = sum(m.api_calls for m in metrics)

    return Metrics(name=name, cost_mdn=cost_per_call, tokens_mdn=tokens_per_call, time_mdn=time_per_call,
                   error_rate_mdn=error_rate, api_issues=api_issues, api_calls=api_calls)

def print_metrics(metrics: list[Metrics], csv_format: bool = False) -> None:
    """Print a list of Metrics entries."""
    for m in metrics:
        error_rate_str = f"{m.error_rate_mdn:.0%}"
        name = f"{m.name},"
        if csv_format:
            print(f"   {name} {m.tokens_mdn}, ${m.cost_mdn:.6f}, {m.time_mdn:.2f}, {error_rate_str}, {m.api_issues}")
        else:
            print(f"\t{name:32.32}\ttokens_mdn={m.tokens_mdn},\tcost_mdn=${m.cost_mdn:.6f},\ttime_mdn={m.time_mdn:.2f}s,\terror_rate_mdn={error_rate_str},\tapi_issues={m.api_issues}/{m.api_calls}")

def main_test():
    print("\n===== metrics.main_test() =====")

    # Create 3 valid metrics:
    v1 = Metrics(name="v1", cost_mdn=0.010, tokens_mdn=100, time_mdn=0.20, error_rate_mdn=0.20)
    v2 = Metrics(name="v2", cost_mdn=0.015, tokens_mdn=120, time_mdn=0.30, error_rate_mdn=0.10)
    v3 = Metrics(name="v3", cost_mdn=0.020, tokens_mdn=150, time_mdn=0.50, error_rate_mdn=0.00)
    expected = Metrics(name="valid-only", cost_mdn=0.015, tokens_mdn=120, time_mdn=0.30, 
                       error_rate_mdn=0.10, api_issues=0, api_calls=3)

    # Create 2 invalid metrics:
    i1 = Metrics(name="i1", cost_mdn=9.999, tokens_mdn=9999, time_mdn=9.99, error_rate_mdn=1.00, api_issues=1)
    i2 = Metrics(name="i2", cost_mdn=8.888, tokens_mdn=8888, time_mdn=8.88, error_rate_mdn=0.90, api_issues=1)

    # Test 3 valid metrics:
    s_valid = summarize_metrics(expected.name, [v1, v2, v3])
    print_metrics([s_valid])
    assert s_valid == expected, "FAIL: Valid metrics summary mismatch."

    # Test empy metrics:
    try:
        summarize_metrics("empty", [])
        raise ValueError("FAIL: [] should raise AssertionError.")
    except AssertionError as ex:
        print(f'PASS: Raised AssertionError("{str(ex)}").')

    # Test 3 valid + 2 invalid metrics:
    expected.name, expected.api_issues, expected.api_calls = "mixed", 2, 5
    s_mixed = summarize_metrics(expected.name, [v1, i1, v2, i2, v3])
    print_metrics([s_mixed], True)
    assert s_mixed == expected, "FAIL: Mixed metrics summary mismatch."

    # Test if summarization of calculated metrics sums api_issues_count and api_calls:
    expected.name, expected.api_calls = "calculated", 8
    s_calc = summarize_metrics(expected.name, [s_valid, s_mixed])
    print_metrics([s_calc])
    assert s_calc == expected, "FAIL: Calculated metrics summary mismatch."

if __name__ == "__main__":
    main_test()