"""Offline deterministic checks: no API credentials or model calls required."""
import math
from data_tools import load_dataset, inspect_dataframe, describe_column, fit_ols
from diagnostic_tools import check_ols_diagnostics


def run_tests():
    df = load_dataset("biomarker")
    assert inspect_dataframe(df)["rows"] == 200
    assert math.isfinite(describe_column(df, "age")["mean"])
    model = fit_ols(df, "biomarker", ["age", "sex"])
    assert model["n_observations"] == 200
    assert 0.5 < model["coefficients"]["age"] < 1.1
    assert 5 < model["coefficients"]["sex"] < 20
    diag = check_ols_diagnostics(df, "biomarker", ["age", "sex"])
    assert diag["n_observations"] == 200
    assert 0 <= diag["breusch_pagan_lm_p"] <= 1
    assert 0 <= diag["shapiro_wilk_p"] <= 1
    try:
        check_ols_diagnostics(df, "age", ["sex"])
        raise AssertionError("Unsupported outcome should raise ValueError")
    except ValueError:
        pass
    print("PASS: inspect, summaries, regression, diagnostics, invalid-input handling")
    print("Regression coefficients:", model["coefficients"])
    print("Diagnostics:", {k: diag[k] for k in ("shapiro_wilk_p", "breusch_pagan_lm_p", "max_cooks_distance")})

if __name__ == "__main__":
    run_tests()
