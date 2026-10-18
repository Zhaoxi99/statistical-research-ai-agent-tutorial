# Lesson 04: Data Analysis — Connecting AI Agents to Research Datasets

## Learning Objectives

In Lesson 03, we gave an agent a NumPy function that accepted numbers typed into a question. Now we will give the agent access to **CSV datasets** so it can perform common statistical analyses on real tabular structures.

By the end of this lesson, you should be able to:

- Generate and inspect a reproducible synthetic biomedical dataset.
- Read CSV files with pandas and distinguish a DataFrame from a tool argument.
- Write reusable functions for inspecting datasets, describing variables, and fitting linear regression.
- Design LLM tools with simple, validated parameters such as `dataset_name`, `column`, and `predictors`.
- Ask an interactive agent to select and execute appropriate statistical tools.
- Verify results independently instead of trusting the LLM's narrative.

**Prerequisite:** Lessons 01–03, including running a basic agent and using `@function_tool`.

## 1. Why Do We Need Data Analysis Tools?

In Lesson 03, you could ask:

> Calculate the mean of 12, 15, 18, 20, and 25.

But researchers generally work with datasets containing hundreds of observations and multiple variables. Typing all observations into a prompt is inconvenient and unreliable.

Instead, we want to ask:

> What is the average age in the biomarker dataset?

> Is age associated with biomarker levels after adjusting for sex?

The agent should select tools that read the relevant CSV file, perform a numerical calculation, and return the results. **The LLM chooses the analysis, while pandas and statsmodels perform it.**

## 2. Files in This Lesson

```text
04_data_analysis/
├── Lesson4.md
├── generate_data.py
├── biomarker.csv
├── biomarker_replication.csv
├── data_tools.py
├── data_agent.py
├── interactive_agent.py
└── test_tools.py
```

Run commands from inside this directory. The Python scripts use paths relative to their own locations, so they also work when launched from the repository root.

## 3. Environment Setup

Install the required packages in your existing virtual environment:

```bash
python -m pip install openai-agents python-dotenv numpy pandas statsmodels
```

As in previous lessons, configure `OPENAI_API_KEY` through an environment variable or a private `.env` file. **Do not commit your API key to GitHub.** The local dataset-generation and test scripts do **not** require an API key.

## 4. Create a Synthetic Biomedical Dataset

Start with `generate_data.py`. It uses NumPy's reproducible random-number generator to create variables:

| Variable | Description |
|---|---|
| `age` | Continuous variable, approximately normally distributed |
| `sex` | Binary variable coded as 0 or 1 |
| `biomarker` | Continuous outcome influenced by age, sex, and random noise |

The data-generating model is

\[
Y_i = 100 + 0.8\,\mathrm{Age}_i + 12\,\mathrm{Sex}_i + \varepsilon_i,
\qquad \varepsilon_i \sim N(0,15^2).
\]

Run:

```bash
python generate_data.py
```

This creates two CSV files: `biomarker.csv` (200 observations) and `biomarker_replication.csv` (150 observations), using different random seeds and generating parameters. The CSV files are included in the lesson for convenience; rerunning the generator overwrites them with the same reproducible values.

Why use synthetic data? We know the underlying generating model, so we can check whether estimated coefficients are plausible. We can also share the data freely in a public tutorial.

## 5. Read and Inspect a CSV Using pandas

Open `data_tools.py`. The first important function is:

```python
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATASET_FILES = {
    "biomarker": HERE / "biomarker.csv",
    "biomarker_replication": HERE / "biomarker_replication.csv",
}

def load_dataset(dataset_name: str) -> pd.DataFrame:
    if dataset_name not in DATASET_FILES:
        raise ValueError("Unknown dataset")
    return pd.read_csv(DATASET_FILES[dataset_name])
```

The actual function also checks whether the file exists and provides helpful error messages.

**What is a DataFrame?** A pandas `DataFrame` is a table with named columns and rows. You can access a column using `df["age"]`, count rows using `len(df)`, and check missing values using `df.isna().sum()`.

For example:

```python
from data_tools import load_dataset

df = load_dataset("biomarker")
print(df.head())
print(df.columns.tolist())
```

Rather than allowing the agent to select an arbitrary filename, we provide a **registry of approved datasets**. The agent can choose from these names but cannot access arbitrary files on your computer.

## 6. Write Reusable Statistical Functions

The module `data_tools.py` includes three key functions:

| Function | Input | Output |
|---|---|---|
| `inspect_dataframe(df)` | pandas DataFrame | Rows, columns, types, missing-value counts |
| `describe_column(df, column)` | DataFrame and column name | Count, mean, sample SD, minimum, maximum |
| `fit_ols(df, outcome, predictors)` | DataFrame, outcome, predictor list | OLS estimates, SEs, p-values, CIs, R² |

These are **ordinary Python functions**, not agent tools. This is intentional: you can test them without sending requests to an LLM.

For descriptive statistics, we calculate quantities like:

```python
values = df["age"].dropna()
mean = values.mean()
sample_sd = values.std(ddof=1)
```

For linear regression, `statsmodels` can fit the model

\[
\mathrm{Biomarker}_i = \beta_0 + \beta_1\mathrm{Age}_i
    + \beta_2\mathrm{Sex}_i + \varepsilon_i.
\]

For instance:

```python
import statsmodels.formula.api as smf

model = smf.ols("biomarker ~ age + sex", data=df).fit()
print(model.params)
```

The tutorial implementation deliberately restricts the outcome to `biomarker` and predictors to `age` and `sex`. It constructs the formula only after validating the variable names, rather than executing arbitrary model expressions supplied by an LLM.

The result includes estimates, standard errors, p-values and 95% confidence intervals. The model uses complete cases for the requested variables and reports how many rows were dropped.

## 7. A Design Question: Why Not Use `def inspect_data(df)`?

In an ordinary Python function, passing a DataFrame is a good design:

```python
def inspect_dataframe(df: pd.DataFrame) -> dict:
    return {"rows": len(df), "columns": df.columns.tolist()}
```

But when using `@function_tool`, the SDK needs to create a **JSON Schema for the arguments the LLM should supply**. A DataFrame is a complex Python object and is not an appropriate simple LLM-generated argument. Unannotated parameters can also cause schema errors.

Therefore, **we separate the data-processing function from the agent-facing tool**:

```python
@function_tool
def inspect_data(dataset_name: str) -> dict:
    """Inspect a registered dataset.

    Args:
        dataset_name: Name of the dataset to inspect.
    """
    df = load_dataset(dataset_name)
    return inspect_dataframe(df)
```

Now the model only needs to supply a string such as `"biomarker"`. Python is responsible for retrieving the DataFrame.

This approach gives us both **reusable statistical functions** and **simple tool interfaces**.

## 8. Create the Data Analysis Agent

Open `data_agent.py`. We expose four tools:

1. `list_datasets()` — find the approved dataset names.
2. `inspect_data(dataset_name)` — inspect columns, size and missingness.
3. `describe_variable(dataset_name, column)` — calculate numerical summaries.
4. `fit_regression(dataset_name, outcome, predictors)` — fit an OLS regression.

Pay attention to the docstrings. For example, `fit_regression` explains that it should be used for OLS coefficients, associations, confidence intervals and p-values. Its arguments have explicit types:

```python
@function_tool
def fit_regression(
    dataset_name: str,
    outcome: str,
    predictors: list[str],
) -> dict:
    """Fit an ordinary least squares regression.

    Args:
        dataset_name: Registered dataset name.
        outcome: Response variable, 'biomarker' in this lesson.
        predictors: List such as ['age', 'sex'].
    """
    ...
```

The SDK converts these types and descriptions into a tool definition. The **LLM selects the tool and supplies its simple arguments**. The Python tool then calls the underlying statistical function.

The agent's instructions also tell it not to invent numerical results, to treat the regression as associational rather than causal, and to explain tool errors.

## 9. Verify the Statistical Calculations First

Run:

```bash
python test_tools.py
```

This script does not use the OpenAI API. It checks dataset dimensions, column names, numerical summaries, regression outputs, and rejection of an invalid predictor.

Expected output includes:

```text
PASS: dataset inspection, summaries, regression and invalid-input checks
Age mean: 64.695
Regression coefficients: {'Intercept': 101.414..., 'age': 0.7715..., 'sex': 11.2814...}
```

Small differences in the printed number of digits are not important, but the estimates should match when using the supplied CSV.

The fitted coefficient of age is close to its data-generating value of 0.8; the sex coefficient is close to 12. They are not exactly equal because the dataset includes random noise.

## 10. Run the Interactive Data Analysis Agent

Now run:

```bash
python interactive_agent.py
```

The program will display:

```text
Data Analysis Agent — type 'exit' to quit.
You:
```

Try these questions in sequence:

**Question 1:**
> What datasets can you analyse?

Expected: the agent should use `list_datasets` and mention `biomarker` and `biomarker_replication`.

**Question 2:**
> What variables are in the biomarker dataset? Are there missing values?

Expected: `inspect_data`, returning 200 rows and columns `age`, `sex`, `biomarker` with zero missing values.

**Question 3:**
> What is the mean age in the biomarker dataset?

Expected: `describe_variable` with `dataset_name="biomarker"` and `column="age"`. The mean should be approximately **64.695**.

**Question 4:**
> Fit an OLS model of biomarker on age and sex in the biomarker dataset. Interpret the coefficients.

Expected: `fit_regression`, with coefficients approximately `Intercept=101.415`, `age=0.772`, `sex=11.281`. The agent should explain these as **conditional associations**, not causal effects.

**Question 5:**
> Now do the same analysis in the biomarker_replication dataset.

Expected: the agent should use conversation context to select the second dataset and call `fit_regression` again. The numerical estimates will differ.

After each answer, the program prints requested tool names:

```text
TOOL CALLED: fit_regression('biomarker', 'biomarker', ['age', 'sex'])
Tool calls: ['fit_regression']
Agent: ...
```

The exact text and the number/order of tool calls may differ. A correct-looking answer **without any relevant tool invocation** is not evidence that the agent analysed the data.

## 11. Exercises

### Exercise 1: Invalid dataset

Ask:

> What is the mean age in a dataset called patients_2026?

The agent should not invent data. It may ask you to select an existing dataset or report the error from `describe_variable`.

### Exercise 2: Invalid column

Ask:

> What is the mean weight in the biomarker dataset?

The tool should report that the column is unavailable.

### Exercise 3: Compare datasets

Ask:

> Compare the mean age of the biomarker and biomarker_replication datasets.

The agent may call `describe_variable` twice and summarise the two results.

### Exercise 4: Inspect the agent's tools

Print each tool's `name`, `description`, and `params_json_schema`, as in Lesson 03. Identify how the LLM knows which dataset and predictors to request.

### Exercise 5: Reproducibility

Run `generate_data.py` and `test_tools.py` again. Check that the same results are obtained because the random-number generator uses fixed seeds.

## 12. Limitations

This is an educational prototype, not an unrestricted statistical analysis system.

- Only registered synthetic CSV datasets are available.
- Regression is limited to an additive OLS model with a specified outcome and predictors.
- OLS standard errors and p-values depend on statistical assumptions that **have not been checked** in this lesson.
- Missing observations are handled using complete-case analysis, which may be inappropriate for some real datasets.
- The agent may make incorrect tool-selection or interpretation decisions.
- Tool outputs can still be misreported by the LLM; always compare with direct computations.
- Real clinical datasets may contain sensitive information and should not be uploaded to an external service without appropriate governance.

We'll add regression diagnostics and multi-step reasoning in Lesson 05.

## 13. Key Takeaways

- A research agent can work with tabular data without sending an entire DataFrame as a tool argument.
- Registered **dataset names** are a simple interface between an LLM and local files.
- `@function_tool` exposes statistical capabilities using JSON-compatible inputs.
- pandas and statsmodels perform deterministic computations; the LLM selects tools and interprets the outputs.
- Keeping reusable calculations separate from agent wrappers makes debugging and testing easier.
- Numerical correctness, tool-selection accuracy, and scientific interpretation are **different aspects of reliability**.

**Next: Lesson 05 — Multi-Step Statistical Analysis.** We will add model diagnostics and let the agent decide what analyses to perform based on earlier results.

## References

- [OpenAI Agents SDK — Tools](https://openai.github.io/openai-agents-python/tools/)
- [OpenAI Agents SDK — Running Agents](https://openai.github.io/openai-agents-python/running_agents/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [statsmodels — OLS](https://www.statsmodels.org/stable/regression.html)
- [NumPy Documentation](https://numpy.org/doc/stable/)
