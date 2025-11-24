from dataclasses import dataclass
from statistics import median, mean
from datetime import datetime

@dataclass
class Metrics:
    name: str               # Model or run name.
    provider_name: str      # Provider name.
    cost_mdn: float         # Median cost per API call, because of outliers.
    tokens_mdn: int         # Median token count per API call, because of outliers.
    time_avg: float         # Average of time.
    error_rate_avg: float   # Average of error rate.
    api_issues: int = 0     # Total count of API issues (e.g., timeouts, quota errors).
    api_calls: int = 1      # Total count of API calls.

    @classmethod
    def get_error(cls, provider_name: str) -> 'Metrics':
        return Metrics(
            name="Error",
            provider_name=provider_name,
            cost_mdn=0.0,
            tokens_mdn=0,
            time_avg=0.0,
            error_rate_avg=1.0,
            api_issues=1,
            api_calls=1
        )

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
        cost = median(m.cost_mdn for m in valid_metrics)
        tokens = int(median(float(m.tokens_mdn) for m in valid_metrics))
        time = mean(m.time_avg for m in valid_metrics)
        error_rate = mean(m.error_rate_avg for m in valid_metrics)
    else:
        cost = time = error_rate = 0.0
        tokens = 0

    providers = {m.provider_name for m in metrics}
    provider_name = providers.pop() if len(providers) == 1 else "mixed" 
    api_issues = sum(m.api_issues for m in metrics)
    api_calls = sum(m.api_calls for m in metrics)

    return Metrics(name=name, provider_name=provider_name, 
                   cost_mdn=cost, tokens_mdn=tokens, time_avg=time,
                   error_rate_avg=error_rate, api_issues=api_issues, api_calls=api_calls)

def print_metrics(metrics: list[Metrics], csv_format: bool = False) -> None:
    """Print a list of Metrics entries."""
    for m in metrics:
        name = f"{m.name},"
        if csv_format:
            id = datetime.now().strftime("%y%m%d%H%M%f")
            api_issue_rate = m.api_issues / m.api_calls if m.api_calls > 0 else 0.0
            print(f"{id}, {name} {m.provider_name}, {m.tokens_mdn}, ${m.cost_mdn:.6f}, {m.time_avg:.2f}, {m.error_rate_avg:.0%}, {api_issue_rate:.0%}")
        else:
            print(f"\t{name:12.12}\ttokens_mdn={m.tokens_mdn:>6,d},\tcost_mdn=${m.cost_mdn:.6f},\ttime_avg={m.time_avg:>5.2f}s,\terror_rate_avg={m.error_rate_avg:>5.0%},\tapi_issues={m.api_issues}/{m.api_calls}")

def main_test():
    print("\n===== metrics.main_test() =====")

    # Create 3 valid metrics:
    v1 = Metrics(name="v1", provider_name="ACME", cost_mdn=0.010, tokens_mdn=100, time_avg=0.20, error_rate_avg=0.20)
    v2 = Metrics(name="v2", provider_name="ACME", cost_mdn=0.015, tokens_mdn=120, time_avg=0.30, error_rate_avg=0.10)
    v3 = Metrics(name="v3", provider_name="ACME", cost_mdn=0.020, tokens_mdn=150, time_avg=0.40, error_rate_avg=0.00)
    expected = Metrics(name="valid-only", provider_name="ACME", cost_mdn=0.015, tokens_mdn=120, time_avg=0.30, 
                       error_rate_avg=0.10, api_issues=0, api_calls=3)

    # Create 2 invalid metrics:
    i1 = Metrics(name="i1", provider_name="BAD_ACME", cost_mdn=9.999, tokens_mdn=9999, time_avg=9.99, error_rate_avg=1.00, api_issues=1)
    i2 = Metrics(name="i2", provider_name="BAD_ACME", cost_mdn=8.888, tokens_mdn=8888, time_avg=8.88, error_rate_avg=0.90, api_issues=1)

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
    expected.name = expected.provider_name = "mixed"
    expected.api_issues, expected.api_calls = 2, 5
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