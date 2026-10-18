"""Ordinary, independently testable Python statistical functions."""
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
# Explicit registry: the agent cannot choose arbitrary filesystem paths.
DATASET_FILES = {
    "biomarker": HERE / "biomarker.csv",
    "biomarker_replication": HERE / "biomarker_replication.csv",
}
ALLOWED_COLUMNS = {"age", "sex", "biomarker"}


def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a named local CSV; error if name is not registered."""
    if dataset_name not in DATASET_FILES:
        raise ValueError(f"Unknown dataset {dataset_name!r}. Available: {list(DATASET_FILES)}")
    path = DATASET_FILES[dataset_name]
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is missing. Run python generate_data.py first.")
    return pd.read_csv(path)


def inspect_dataframe(df: pd.DataFrame) -> dict:
    """Summarise columns, rows, data types and missingness."""
    return {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "dtypes": {key: str(value) for key, value in df.dtypes.items()},
        "missing_values": {key: int(value) for key, value in df.isna().sum().items()},
    }


def describe_column(df: pd.DataFrame, column: str) -> dict:
    """Return numeric summaries of a named DataFrame column."""
    if column not in df.columns:
        raise ValueError(f"Unknown column {column!r}; available: {df.columns.tolist()}")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column!r} is not numeric")
    values = df[column].dropna()
    if values.empty:
        raise ValueError(f"Column {column!r} contains no valid observations")
    return {
        "column": column,
        "count": int(values.count()),
        "missing": int(df[column].isna().sum()),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def fit_ols(df: pd.DataFrame, outcome: str, predictors: list[str]) -> dict:
    """Fit an additive OLS model with a constrained set of variable names."""
    if outcome != "biomarker":
        raise ValueError("This lesson only permits outcome='biomarker'")
    if not predictors or len(set(predictors)) != len(predictors):
        raise ValueError("Provide at least one distinct predictor")
    if any(p not in {"age", "sex"} for p in predictors):
        raise ValueError("Only 'age' and 'sex' are supported as predictors")
    needed = [outcome] + predictors
    if any(col not in df.columns for col in needed):
        raise ValueError(f"Missing one of the required columns: {needed}")
    if any(not pd.api.types.is_numeric_dtype(df[col]) for col in needed):
        raise ValueError("All model columns must be numeric")
    complete = df[needed].dropna()
    if len(complete) <= len(predictors) + 1:
        raise ValueError("Not enough complete observations to fit this model")
    formula = outcome + " ~ " + " + ".join(predictors)
    model = smf.ols(formula, data=complete).fit()
    ci = model.conf_int()
    return {
        "formula": formula,
        "n_observations": int(model.nobs),
        "rows_dropped_for_missingness": int(len(df) - len(complete)),
        "coefficients": {k: float(v) for k, v in model.params.items()},
        "standard_errors": {k: float(v) for k, v in model.bse.items()},
        "p_values": {k: float(v) for k, v in model.pvalues.items()},
        "confidence_intervals_95": {
            k: [float(ci.loc[k, 0]), float(ci.loc[k, 1])]
            for k in model.params.index
        },
        "r_squared": float(model.rsquared),
    }
