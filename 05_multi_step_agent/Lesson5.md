# Lesson 05: Multi-step Statistical Research Agents

## Learning Objectives

In Lesson 04, our agent could inspect a CSV file, describe variables, and fit an ordinary least squares (OLS) regression. **Now we will let the agent combine these tools into a complete research workflow.**

By the end of the lesson, you should be able to:

- Explain the **Reason–Act–Observe** loop in a real statistical example.
- Distinguish a fixed Python pipeline from a tool-using LLM agent.
- Provide multiple complementary tools and let the agent choose which to call.
- Add **model diagnostics** to an existing regression agent.
- Inspect tool-call sequences and understand why a final answer alone is not sufficient for debugging.
- Recognise important limits of autonomous statistical analysis.

This is a learning project based on **synthetic**, not patient, data.

## 1. From One Tool Call to a Research Workflow

Imagine a user asks:

> Investigate whether age is associated with biomarker levels, adjusting for sex. Check the assumptions and summarise the findings.

One statistical computation is not enough. A reasonable workflow includes:

1. **Inspect** the dataset: do `age`, `sex`, and `biomarker` exist? Are there missing values?
2. **Summarise** relevant variables.
3. **Fit** an OLS regression, using `biomarker ~ age + sex`.
4. **Check** selected model diagnostics.
5. **Explain** coefficients, uncertainty, diagnostic results, and limitations.

A traditional script might call these five steps in a fixed order. With an LLM agent, we instead expose tools and let the model decide what actions to request according to the question and intermediate results.

**An agent is not necessarily better than a predetermined pipeline.** For a standardised or regulated statistical workflow, a validated fixed pipeline may be safer. Here, our aim is to *learn the mechanics and limits of agentic orchestration*.

## 2. Folder Structure and Setup

```text
05_multi_step_agent/
├── Lesson5.md
├── biomarker.csv
├── biomarker_replication.csv
├── generate_data.py
├── data_tools.py
├── diagnostic_tools.py
├── research_agent.py
├── interactive_agent.py
├── run_research_question.py
└── test_tools.py
```

The dataset files, generator, and ordinary data-analysis functions are carried forward from Lesson 04 so this lesson can also be run independently. `data_tools.py` contains functions that accept actual DataFrames; `research_agent.py` exposes LLM-friendly functions that accept dataset *names* instead.

Install dependencies in your Python environment:

```bash
python -m pip install openai-agents python-dotenv numpy pandas scipy statsmodels
```

Configure `OPENAI_API_KEY` as in the earlier lessons. Keep credentials out of GitHub. Model calls may incur usage charges.

Run commands below **from inside** `05_multi_step_agent/`.

## 3. The Statistical Example

Our synthetic data follow the generating model

\[
Y_i=100+0.8\,\mathrm{Age}_i+12\,\mathrm{Sex}_i+\varepsilon_i,
\qquad \varepsilon_i\sim N(0,15^2).
\]

`sex` is simulated as a binary 0/1 variable; it is just an illustrative coded predictor. Real biomedical analyses require careful study design, appropriate clinical interpretation, and protection of personal data.

Run `python generate_data.py` if you want to regenerate the CSV files. The generator uses fixed random seeds for reproducibility.

Our regression asks about **association**, not a causal effect. The estimated parameters differ from the generating values because of sampling variation.

## 4. Keep the Statistical Calculations Separate

Open `data_tools.py`. It provides functions for loading a registered CSV, inspecting a DataFrame, summarising a column, and fitting OLS. These functions do **not** depend on an LLM and can be tested independently.

The regression is implemented with statsmodels as an additive model:

```python
model = smf.ols("biomarker ~ age + sex", data=df).fit()
```

The model returns coefficient estimates, standard errors, 95% confidence intervals, p-values and \(R^2\).

Why limit dataset and predictor names? An LLM can make mistakes, and an unrestricted free-form formula or filesystem path can introduce risks. For this introductory lesson, we permit only named local CSV files, the outcome `biomarker`, and predictors `age` and/or `sex`.

## 5. Add an OLS Diagnostic Tool

New file: `diagnostic_tools.py`.

It fits the same model and returns:

- **Shapiro–Wilk p-value:** a test for departures from residual normality.
- **Breusch–Pagan p-values:** tests targeting particular patterns of heteroscedasticity.
- **Cook's distance:** an influence diagnostic, including the maximum and number of observations above the informal `4/n` heuristic.

For example, the Breusch–Pagan test is implemented using:

```python
from statsmodels.stats.diagnostic import het_breuschpagan
bp_lm, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals, model.model.exog)
```

**Important statistical caveat:** These are *screening checks*, not proof that the assumptions hold. None directly establishes linearity, independent observations, correct specification, or causality. Diagnostic plots and scientific judgement still matter. A non-significant p-value is not proof that an assumption is satisfied.

## 6. Expose Multiple Tools to the LLM

Open `research_agent.py`. The agent has five tools:

| Tool | When it may be useful |
|---|---|
| `list_datasets` | The user has not identified a dataset |
| `inspect_data` | Check column names, types and missingness |
| `describe_variable` | Compute summary statistics for a column |
| `fit_regression` | Estimate associations via OLS |
| `check_regression_assumptions` | Assess selected regression diagnostics |

Each tool accepts **simple JSON-compatible arguments**, such as:

```python
check_regression_assumptions(
    dataset_name="biomarker",
    outcome="biomarker",
    predictors=["age", "sex"],
)
```

This is conceptual usage: the decorated function is exposed to the SDK; the model supplies the argument values and the SDK executes it.

We don't pass a `pandas.DataFrame` from the LLM. Python loads the data internally using the validated `dataset_name`. This is the reusable design you learned in Lesson 04.

### How does the agent know what to do?

The LLM receives **tool names, docstrings, parameter schemas, agent instructions, and the user's question**. For example, the description of `check_regression_assumptions` says to use it when the user requests diagnostics. The model uses that description and prior tool results when selecting its next action.

Ordinary `#` comments inside the implementation are not automatically part of that tool description.

## 7. Reason–Act–Observe in Practice

A possible sequence is:

**Reason:** Need to know whether the required columns exist.

**Act:** Request `inspect_data(dataset_name="biomarker")`.

**Observe:** Receive the list of columns, row count and missingness information.

**Reason:** The data support the requested analysis.

**Act:** Request `fit_regression(...)`.

**Observe:** Receive estimated coefficients and uncertainty.

**Reason:** The user asked for model diagnostics.

**Act:** Request `check_regression_assumptions(...)`.

**Observe:** Receive diagnostic statistics and limitations; construct the final report.

The actual tool sequence is **not guaranteed** to match this example. The model might request variable summaries or additional tools, or it might make a mistake. The SDK manages the repeated model/tool loop; we don't have to write each step in a fixed order.

Also, "Reason" is a description of the decision-making pattern, **not a promise that the model's private internal reasoning is visible**.

## 8. First Run the Deterministic Tests (No API Needed)

```bash
python test_tools.py
```

Expected output begins with:

```text
PASS: inspect, summaries, regression, diagnostics, invalid-input handling
```

It also prints the model coefficients and some diagnostic results. The exact deterministic values can be inspected after running the script.

If this test fails, first resolve the Python/statistical-code issue before asking the LLM to use the tools. For example, a missing NumPy import can cause a tool to fail even when the LLM selected it correctly.

## 9. Run a Complete Agentic Investigation

```bash
python run_research_question.py
```

This asks the agent to analyse age and biomarker levels, adjusting for sex, and explicitly requests diagnostics. The script prints:

1. `TOOL EXECUTED: ...` messages as Python functions run.
2. A **tool-event sequence** showing requested tools and shortened returned outputs.
3. A final research summary produced by the agent.

You should generally see `fit_regression` and `check_regression_assumptions` among the tool calls, but tool choice and ordering are model-dependent. Check the numerical results against `python test_tools.py`, not just the apparent plausibility of the prose.

The example uses a maximum of 12 agent turns to prevent an unlimited execution loop. This is an upper bound, not a request to use all 12 turns.

## 10. Interactive Agent: Vary the Complexity of the Question

Run:

```bash
python interactive_agent.py
```

Try these prompts in order:

**A. Simple factual request**

> What columns are available in the biomarker dataset?

Expected: Usually `inspect_data`, without unnecessary regression.

**B. Regression only**

> Fit a regression of biomarker on age and sex in the biomarker dataset. Interpret the age coefficient.

Expected: `fit_regression`, potentially preceded by inspection.

**C. Comprehensive investigation**

> Investigate age and biomarker, adjusting for sex. Summarise the data, fit OLS, check diagnostics and explain limitations.

Expected: Multiple relevant tools including regression and diagnostic checks.

**D. Invalid dataset**

> Fit biomarker on age in the imaginary_dataset.

Expected: A transparent error, not fabricated coefficients.

The terminal prints the requested tool names after each run. Since the history is preserved in memory, follow-up questions can refer to previous context. History is lost when you close the script.

## 11. How to Interpret the Results

For the synthetic generating mechanism, a positive age coefficient is expected, but the estimated coefficient will differ from `0.8` in a finite sample. The adjusted sex coefficient will likewise vary around `12`.

The agent should report effect estimates and uncertainty, avoid causal language, and explain what its diagnostics can and cannot establish.

**Tool execution vs agent correctness:** A correctly executed regression is not enough to guarantee that the model interpreted it properly. Later we will evaluate tool selection, numerical reporting, and statistical interpretation separately.

## 12. Common Problems and Limitations

- **Unknown dataset:** Tools accept only `biomarker` or `biomarker_replication`.
- **Unsupported formula:** Only `biomarker` as outcome and `age`/`sex` predictors are accepted.
- **Missing package or import:** Test the statistical functions independently first.
- **Tool not called:** Inspect the tool names and the agent's descriptions; tool availability does not guarantee selection.
- **Misleading diagnostics:** A large p-value does not prove that an assumption is true. Residual plots, independence, and functional form still require attention.
- **Model-generated mistakes:** The agent can misuse results or provide an unsupported interpretation.
- **Unbounded execution/cost:** Tool-using agents may require multiple model requests. We set `max_turns` and log tool calls.

This is an educational prototype, not a validated clinical decision-support system.

## 13. Practice Exercises

1. Compare a simple question with a comprehensive question. Does the agent request different numbers of tools?
2. Ask about an unavailable dataset. Does the error reach the final answer accurately?
3. Ask for an inappropriate causal conclusion. Does the agent distinguish association from causation?
4. Compare `biomarker` and `biomarker_replication`. Do the fitted coefficients vary?
5. Temporarily remove the diagnostic tool from `tools=[...]`. Can the agent still honestly describe the diagnostic limitation?
6. Look at the sequence of tool outputs. Could a validated fixed pipeline be preferable for some tasks?

## 14. Key Takeaways

- Agentic workflows can coordinate **several statistical tools** rather than just one.
- The LLM chooses actions based on the current question, tool metadata and observations.
- The **Reason–Act–Observe** pattern describes this iterative behaviour.
- Python performs computations; the agent orchestrates and explains them.
- Tool-call traces and offline tests help separate software errors from model decisions.
- Statistical validity requires more than a successful agent run.

**Next: Lesson 06 — Evaluation.** We will quantitatively assess tool-selection behaviour, errors, and reproducibility.

## References

- [OpenAI Agents SDK — Running agents](https://openai.github.io/openai-agents-python/running_agents/)
- [OpenAI Agents SDK — Tools](https://openai.github.io/openai-agents-python/tools/)
- [OpenAI Agents SDK — Tracing](https://openai.github.io/openai-agents-python/tracing/)
- [statsmodels diagnostic tests](https://www.statsmodels.org/stable/diagnostic.html)
