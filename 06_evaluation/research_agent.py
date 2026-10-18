"""LLM-facing tools for multi-step biomedical research analysis."""
from dotenv import load_dotenv
from agents import Agent, function_tool
from data_tools import DATASET_FILES, load_dataset, inspect_dataframe, describe_column, fit_ols
from diagnostic_tools import check_ols_diagnostics

load_dotenv()

@function_tool
def list_datasets() -> dict:
    """List registered research datasets when the available names are unknown."""
    print("TOOL EXECUTED: list_datasets")
    return {"datasets": list(DATASET_FILES)}

@function_tool
def inspect_data(dataset_name: str) -> dict:
    """Inspect row count, columns, data types, and missingness before analysis.

    Args:
        dataset_name: Registered dataset identifier, for example 'biomarker'.
    """
    print(f"TOOL EXECUTED: inspect_data({dataset_name})")
    try:
        return inspect_dataframe(load_dataset(dataset_name))
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}

@function_tool
def describe_variable(dataset_name: str, column: str) -> dict:
    """Calculate sample size, mean, sample SD and range for a numeric variable.

    Args:
        dataset_name: Registered dataset identifier.
        column: Column name, such as age, sex, or biomarker.
    """
    print(f"TOOL EXECUTED: describe_variable({dataset_name}, {column})")
    try:
        return describe_column(load_dataset(dataset_name), column)
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}

@function_tool
def fit_regression(dataset_name: str, outcome: str, predictors: list[str]) -> dict:
    """Fit additive OLS regression, providing estimates, SEs, 95% CIs and p-values.

    Use for biomarker association questions, not as proof of causation.
    This lesson only supports outcome 'biomarker' and predictors 'age' and/or 'sex'.

    Args:
        dataset_name: Registered dataset identifier.
        outcome: Response variable, currently 'biomarker'.
        predictors: Predictor names; for example ['age', 'sex'].
    """
    print(f"TOOL EXECUTED: fit_regression({dataset_name}, {outcome}, {predictors})")
    try:
        return fit_ols(load_dataset(dataset_name), outcome, predictors)
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}

@function_tool
def check_regression_assumptions(dataset_name: str, outcome: str, predictors: list[str]) -> dict:
    """Screen residual normality, heteroscedasticity and influential observations for an OLS model.

    Use after fitting a regression when the user asks about assumptions or diagnostics.
    Results are limited screening evidence, not proof assumptions hold.

    Args:
        dataset_name: Registered dataset identifier.
        outcome: Outcome variable, currently 'biomarker'.
        predictors: Predictor variables, 'age' and/or 'sex'.
    """
    print(f"TOOL EXECUTED: check_regression_assumptions({dataset_name}, {outcome}, {predictors})")
    try:
        return check_ols_diagnostics(load_dataset(dataset_name), outcome, predictors)
    except (ValueError, FileNotFoundError) as e:
        return {"error": str(e)}

agent = Agent(
    name="Multi-step Statistical Research Agent",
    instructions=(
        "You assist with statistical research using ONLY registered datasets. "
        "Use tools rather than inventing data, p-values, coefficients, or diagnostic results. "
        "For a comprehensive regression investigation, inspect columns/missingness; "
        "get relevant descriptive summaries; fit the requested regression; "
        "then use the diagnostic tool to screen its assumptions. "
        "After reviewing tool outputs, decide whether additional available tools are needed. "
        "Interpret sex as coded 0/1. Distinguish associations from causation. "
        "Report model uncertainty and limitations accurately. "
        "If a tool returns an error, explain it; do not invent results. "
        "For simple questions, use only necessary tools."
    ),
    tools=[list_datasets, inspect_data, describe_variable, fit_regression, check_regression_assumptions],
)
