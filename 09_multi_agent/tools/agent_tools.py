"""Narrow model-facing tool wrappers. Errors returned as data, not invented answers."""
from agents import function_tool
from tools.statistical_tools import inspect_dataset, describe_numeric, regression, regression_diagnostics

def safe(call, *args):
    try: return call(*args)
    except (ValueError, KeyError, TypeError) as exc: return {'error':str(exc)}

@function_tool
def inspect_data(dataset: str) -> dict:
    """Inspect dataset shape, columns and missingness. Dataset: biomarker or biomarker_replication.

    Args:
        dataset: Exact allowed dataset name.
    """
    return safe(inspect_dataset, dataset)

@function_tool
def describe_variable(dataset: str, column: str) -> dict:
    """Compute descriptive statistics for a numeric column in one allowed dataset.

    Args:
        dataset: Exact dataset name.
        column: Column to summarise.
    """
    return safe(describe_numeric, dataset, column)

@function_tool
def fit_regression(dataset: str, outcome: str, predictors: list[str]) -> dict:
    """Fit OLS regression and return coefficients, p-values and R squared.

    Args:
        dataset: Exact dataset name.
        outcome: Numeric dependent variable.
        predictors: List of numeric predictors.
    """
    return safe(regression, dataset, outcome, predictors)

@function_tool
def check_regression_assumptions(dataset: str, outcome: str, predictors: list[str]) -> dict:
    """Compute Shapiro-Wilk, Breusch-Pagan and maximum Cook's distance for an OLS specification.

    Args:
        dataset: Exact dataset name.
        outcome: Numeric dependent variable.
        predictors: List of numeric predictors.
    """
    return safe(regression_diagnostics, dataset, outcome, predictors)
