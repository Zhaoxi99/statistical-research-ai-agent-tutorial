"""Offline checks: run without an API key to verify numerical correctness."""
import math
from data_tools import load_dataset, inspect_dataframe, describe_column, fit_ols


def main() -> None:
    df = load_dataset("biomarker")
    structure = inspect_dataframe(df)
    assert structure["rows"] == 200
    assert structure["columns"] == ["age", "sex", "biomarker"]
    assert all(count == 0 for count in structure["missing_values"].values())

    description = describe_column(df, "age")
    assert description["count"] == 200
    assert math.isclose(description["mean"], df["age"].mean(), rel_tol=1e-12)

    result = fit_ols(df, "biomarker", ["age", "sex"])
    assert result["n_observations"] == 200
    assert set(result["coefficients"]) == {"Intercept", "age", "sex"}
    assert result["r_squared"] >= 0

    try:
        fit_ols(df, "biomarker", ["__import__('os')"])
        raise AssertionError("Unsafe predictor was accepted")
    except ValueError:
        pass

    print("PASS: dataset inspection, summaries, regression and invalid-input checks")
    print("Age mean:", round(description["mean"], 3))
    print("Regression coefficients:", result["coefficients"])


if __name__ == "__main__":
    main()
