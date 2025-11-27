# Python stdlib.
import dataclasses as dc
from statistics import median, mean
from datetime import datetime

# Data class to hold metrics results.

@dc.dataclass(frozen=True)
class Metrics:
    name: str = 'ERROR'             # Model or run name.
    provider_name: str = 'ERROR'    # Provider name.
    cost_mdn: float = 0.0           # Median cost per API call, because of outliers.
    tokens_mdn: int = 0             # Median token count per API call, because of outliers.
    time_avg: float = 0.0           # Average of time.
    error_rate_avg: float = 1.0     # Average of error rate.
    api_issues: int = 1             # Count of API issues (e.g., timeouts, quota errors).
    api_calls: int = 1              # Count of API calls.
    
# Helper functions.

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
    assert len(metrics) > 0, "No metrics to summarize."

    # 2025-11-20 Timeouts and quota errors are now ignored in calculations.
    valid_metrics = [m for m in metrics if m.api_issues < m.api_calls]

    if len(valid_metrics) > 0:
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
                   cost_mdn=cost, tokens_mdn=tokens, 
                   time_avg=time, error_rate_avg=error_rate, 
                   api_issues=api_issues, api_calls=api_calls)

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

# Main test.

def main_test():
    print("\n===== metrics.main_test() =====")

    # Create 3 valid, 2 invalid, and summary metrics.
    v1 = Metrics(name="v1", provider_name="ACME", cost_mdn=0.010, tokens_mdn=100, time_avg=0.20, error_rate_avg=0.20, api_issues=0)
    v2 = Metrics(name="v2", provider_name="ACME", cost_mdn=0.015, tokens_mdn=120, time_avg=0.30, error_rate_avg=0.10, api_issues=0)
    v3 = Metrics(name="v3", provider_name="ACME", cost_mdn=0.020, tokens_mdn=150, time_avg=0.40, error_rate_avg=0.00, api_issues=0)
    summary_3v = Metrics(name="valid-only", \
                       provider_name="ACME", cost_mdn=0.015, tokens_mdn=120, time_avg=0.30, error_rate_avg=0.10, api_issues=0, api_calls=3)
    i1 = Metrics(name="i1", provider_name="BAD_ACME", cost_mdn=9.999, tokens_mdn=9999, time_avg=9.99, error_rate_avg=1.00, api_issues=1)
    i2 = Metrics(name="i2", provider_name="BAD_ACME", cost_mdn=8.888, tokens_mdn=8888, time_avg=8.88, error_rate_avg=0.90, api_issues=1)
    summary_5m = dc.replace(summary_3v, name="mixed", provider_name="mixed", api_issues=2, api_calls=5)
    summary_3v5m = dc.replace(summary_3v, name="calculated", provider_name="mixed", api_issues=2, api_calls=8)

    # Test 3 valid metrics:
    s_valid = summarize_metrics(summary_3v.name, [v1, v2, v3])
    print_metrics([s_valid])
    assert s_valid == summary_3v, "FAIL: Valid metrics summary mismatch."

    # Test empty metrics:
    try:
        summarize_metrics("empty", [])
        raise ValueError("FAIL: [] should raise AssertionError.")
    except AssertionError as ex:
        print(f'PASS: Raised AssertionError("{str(ex)}").')

    # Test 3 valid + 2 invalid metrics:
    s_mixed = summarize_metrics(summary_5m.name, [v1, i1, v2, i2, v3])
    print_metrics([s_mixed], True)
    assert s_mixed == summary_5m, "FAIL: Mixed metrics summary mismatch."

    # Test if summarization of calculated metrics sums api_issues_count and api_calls:
    s_calc = summarize_metrics(summary_3v5m.name, [s_valid, s_mixed])
    print_metrics([s_calc])
    assert s_calc == summary_3v5m, "FAIL: Calculated metrics summary mismatch."

if __name__ == "__main__":
    main_test()