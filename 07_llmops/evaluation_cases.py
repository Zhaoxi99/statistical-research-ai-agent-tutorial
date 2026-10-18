"""Small, labelled evaluation set for the Lesson 05 research agent."""
CASES = [
    {"id": "T1", "question": "What columns and missing values are in the biomarker dataset?", "required": ["inspect_data"], "allowed": ["inspect_data", "list_datasets"]},
    {"id": "T2", "question": "What is the mean age in the biomarker dataset?", "required": ["describe_variable"], "allowed": ["describe_variable", "inspect_data", "list_datasets"]},
    {"id": "T3", "question": "Fit biomarker on age and sex in the biomarker dataset; report the coefficients.", "required": ["fit_regression"], "allowed": ["fit_regression", "inspect_data", "describe_variable", "list_datasets"]},
    {"id": "T4", "question": "For biomarker ~ age + sex in the biomarker dataset, check regression assumptions.", "required": ["check_regression_assumptions"], "allowed": ["check_regression_assumptions", "fit_regression", "inspect_data", "list_datasets"]},
    {"id": "T5", "question": "Explain the distinction between a p-value and a Bayes factor. Do not use any dataset.", "required": [], "allowed": []},
    {"id": "T6", "question": "Perform a comprehensive analysis of biomarker ~ age + sex in the biomarker dataset: inspect, summarise, regress, and check diagnostics.", "required": ["inspect_data", "describe_variable", "fit_regression", "check_regression_assumptions"], "allowed": ["list_datasets", "inspect_data", "describe_variable", "fit_regression", "check_regression_assumptions"]},
    {"id": "T7", "question": "Fit biomarker on age in the nonexistent dataset named phantom; report what happened.", "required": ["fit_regression"], "allowed": ["fit_regression", "list_datasets", "inspect_data"], "expect_error": True},
]
