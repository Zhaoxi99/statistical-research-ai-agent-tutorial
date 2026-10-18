"""Agent-facing wrappers: simple JSON arguments, no DataFrame parameters."""
from dotenv import load_dotenv
from agents import Agent, function_tool
from data_tools import (
    DATASET_FILES, load_dataset, inspect_dataframe, describe_column, fit_ols
)

load_dotenv()


@function_tool
def list_datasets() -> dict:
    """List the dataset names that are available for statistical analysis."""
    print("TOOL CALLED: list_datasets")
    return {"datasets": list(DATASET_FILES)}


@function_tool
def inspect_data(dataset_name: str) -> dict:
    """Inspect a named dataset's columns, data types, rows and missing values.

    Args:
        dataset_name: Registered dataset name, e.g. 'biomarker'.
    """
    print(f"TOOL CALLED: inspect_data({dataset_name!r})")
    try:
        return inspect_dataframe(load_dataset(dataset_name))
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}


@function_tool
def describe_variable(dataset_name: str, column: str) -> dict:
    """Compute count, mean, sample SD, min and max for one numeric column.

    Args:
        dataset_name: Registered dataset name, e.g. 'biomarker'.
        column: Numeric column to summarise, e.g. 'age'.
    """
    print(f"TOOL CALLED: describe_variable({dataset_name!r}, {column!r})")
    try:
        return describe_column(load_dataset(dataset_name), column)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}


@function_tool
def fit_regression(dataset_name: str, outcome: str, predictors: list[str]) -> dict:
    """Fit an ordinary least squares regression and return numeric model results.

    Use when the user asks for adjusted associations, OLS coefficients,
    confidence intervals, or p-values. This tutorial supports outcome
    'biomarker' and additive predictors 'age' and/or 'sex' only.

    Args:
        dataset_name: Registered dataset name, e.g. 'biomarker'.
        outcome: Response variable; must be 'biomarker' in this lesson.
        predictors: List of predictors, e.g. ['age', 'sex'].
    """
    print(f"TOOL CALLED: fit_regression({dataset_name!r}, {outcome!r}, {predictors!r})")
    try:
        return fit_ols(load_dataset(dataset_name), outcome, predictors)
    except (ValueError, FileNotFoundError) as exc:
        return {"error": str(exc)}


agent = Agent(
    name="Statistical Research Assistant",
    instructions=(
        "You analyse only the datasets exposed through tools. "
        "Use tools for information about data and for numerical calculations. "
        "If a dataset name is unclear, use list_datasets. "
        "Only report numerical results returned by tools; never invent values. "
        "When a tool returns an error, explain it. "
        "Sex is coded 0/1; interpret coefficients as associations, not causation. "
        "OLS results alone do not verify regression assumptions."
    ),
    tools=[list_datasets, inspect_data, describe_variable, fit_regression],
)
