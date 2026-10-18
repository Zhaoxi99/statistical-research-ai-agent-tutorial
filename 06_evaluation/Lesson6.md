# Lesson 06: Evaluating a Statistical Research AI Agent

## Learning objectives

In Lesson 05, we built an agent that inspects biomedical data, fits OLS regression, and checks diagnostics. **Now we ask whether it behaves correctly.**

By the end, you should be able to:

- Distinguish **statistical computation**, **tool selection**, **tool execution**, and **final-answer quality**.
- Build an evaluation set with known expectations.
- Run offline, deterministic statistical checks separately from API-based agent tests.
- Measure whether required tools were selected and whether unnecessary tools were called.
- Inspect failed runs and evaluate conclusions critically.
- Explain why a plausible answer is not sufficient evidence of reliability.

## 1. What does evaluating an agent mean?

Imagine asking:

> Fit a regression of biomarker on age and sex, check its assumptions, and interpret the findings.

A plausible final answer is not enough. We want to verify:

1. Did the agent use the registered biomedical dataset?
2. Did it request the correct regression and diagnostic tools?
3. Did those Python functions execute without errors?
4. Were coefficients and uncertainty reported correctly?
5. Did the conclusion avoid asserting causality from observational association?

These are different dimensions of quality. We measure some automatically and inspect others manually.

## 2. The files

```text
06_evaluation/
├── Lesson6.md
├── evaluation_cases.py          # Questions and expected tool behaviour
├── offline_tests.py             # Tests of Python functions only
├── evaluate_agent.py            # Live API-based evaluations
├── research_agent.py            # Agent copied from Lesson 05
├── data_tools.py                # Reusable statistical calculations
├── diagnostic_tools.py          # Regression diagnostics
├── biomarker.csv                # Synthetic observations
└── biomarker_replication.csv    # Second synthetic dataset
```

This folder is **self-contained**; we include a copy of Lesson 05's agent, tools, and synthetic datasets. Later, a larger project could share code through an importable package instead of copying it.

## 3. Setup

Install the dependencies in your virtual environment:

```bash
python -m pip install openai-agents python-dotenv numpy pandas scipy statsmodels
```

Set `OPENAI_API_KEY` as in previous lessons. You need an API key **only for live evaluation**, not for the offline tests.

Run all commands from the `06_evaluation` directory.

## 4. First, test the statistical Python functions

Run:

```bash
python offline_tests.py
```

The tests use Python's built-in `unittest` framework. They verify:

- The example data contain 200 observations and the expected columns.
- pandas returns the correct mean for age.
- The OLS coefficients are close to values independently checked for the fixed synthetic data.
- Diagnostic p-values are in the interval [0,1].
- Invalid dataset names and unsupported outcomes raise appropriate errors.

Expected: **six tests pass** with an `OK` message. The exact display depends on your Python version.

**Why start here?** A failure in the underlying statistical code is not automatically an LLM reasoning error. We should validate the computational tools before measuring the agent.

## 5. Define an evaluation dataset

Open `evaluation_cases.py`. Each test case has:

- `question`: what the user asks.
- `required`: tools necessary for the task.
- `allowed`: tools considered acceptable (including optional preliminary inspection).
- `expect_error`: whether the case deliberately tests invalid input.

For example:

```python
{
    "id": "T3",
    "question": "Fit biomarker on age and sex in the biomarker dataset; report the coefficients.",
    "required": ["fit_regression"],
    "allowed": ["fit_regression", "inspect_data", "describe_variable", "list_datasets"],
}
```

A preliminary data inspection is reasonable even when only regression is required. This is why we do not insist on an exact set of tool calls for every question.

The seven cases include inspection, descriptive statistics, regression, diagnostics, a conceptual question requiring no tools, a multi-step investigation, and an invalid dataset name.

## 6. Run the agent evaluation

```bash
python evaluate_agent.py
```

The script runs each question in a **fresh, independent agent run**, rather than sharing a conversation history across tests. It prints requested tools and whether the decision passed the defined expectations.

It writes:

- `evaluation_results.jsonl`: detailed outcomes, requested tools, tool outputs and final answers.
- `evaluation_summary.csv`: one row per test, convenient for Excel or pandas.

Both files may contain user prompts and model output. Keep them private if you use sensitive data.

Example **illustrative** console output (not a promised result):

```text
Running T1 (repeat 1)...
  calls: ['inspect_data'] decision pass: True execution pass: True
...
SUMMARY
Decision pass: 6/7
Execution pass: 7/7
Saved evaluation_results.jsonl and evaluation_summary.csv
```

Tool selection is stochastic and can differ between runs. The evaluation makes live API requests and incurs token costs.

### Repeating experiments

```bash
python evaluate_agent.py --repeats 3
```

This performs 21 independent runs (7 questions × 3 repeats). Start with one repeat to control costs.

## 7. How the automatic metrics work

Let **R** be the set of required tools and **A** the set of tool names the agent actually requested.

Required-tool recall is:

\[
\mathrm{Recall}_{\mathrm{required}}=\frac{|R\cap A|}{|R|}.
\]

For a conceptual question where no tool is required, the script defines this recall as 1; unnecessary calls are penalised separately.

**Decision pass** means all required tool types were called and every called tool type was in the allowed set:

```python
required <= actual and actual <= allowed
```

This deliberately ignores call order, repeated calls, and correctness of tool arguments. It measures a **coarse decision criterion**, not full task success.

**Tool execution pass** checks whether captured tool outputs contain an `error` field. For the deliberately invalid dataset case, an informative tool error is expected. This check cannot detect every silent error or incorrect computation.

An agent may pass both checks and still deliver an incorrect final answer.

## 8. Evaluate numerical and scientific correctness

For the example regression, calculate a reference answer independently:

```python
from data_tools import load_dataset, fit_ols

df = load_dataset("biomarker")
reference = fit_ols(df, "biomarker", ["age", "sex"])
print(reference["coefficients"])
```

Compare the agent's reported coefficients against the reference output. A simple metric for parameter \(j\) is:

\[
|\widehat\beta_{j,\mathrm{agent}}-\widehat\beta_{j,\mathrm{reference}}|.
\]

For **this lesson**, inspect the final answers in `evaluation_summary.csv` manually; the script does **not** claim to automatically extract every number accurately from free-form prose.

Use this suggested checklist:

| Criterion | Question |
| --- | --- |
| Model specification | Was the correct outcome and predictor set used? |
| Numerical reporting | Do coefficients, uncertainty and p-values match tool results? |
| Interpretation | Are units and conditional associations explained correctly? |
| Statistical caution | Does the agent avoid unsupported causal conclusions? |
| Diagnostic interpretation | Are tests presented as evidence, not proof of assumptions? |
| Error handling | Did the agent clearly report missing datasets or tool failures? |

You can mark each criterion `pass`, `fail`, or `not applicable` in a separate review spreadsheet. Do not conflate manual review with the automated tool-selection score.

## 9. What could go wrong?

- The agent may answer a numerical question without requesting a tool.
- It may request the right tool but supply the wrong dataset, outcome, or predictors.
- The Python tool may fail, even if the LLM selected it correctly.
- The tool may return correct numbers while the final answer misreports them.
- The agent may overinterpret a small p-value as proof of a causal effect.
- Repeated identical prompts may generate different tool sequences.

Record failures instead of deleting them. In a research-oriented tutorial, understanding failure modes is part of the result.

## 10. Exercises

1. Run `offline_tests.py`, then run the live evaluation once. Compare their purposes.
2. Examine test T6. Were all required tools used? In what order?
3. Inspect T7. Does the agent report that `phantom` does not exist, rather than fabricating results?
4. Repeat the live evaluation three times. Do results vary between repetitions?
5. Change only the agent instructions and rerun the **same cases**. Did tool selection or answer quality change?
6. Add a case where the user asks for an unsupported statistical model. Decide how a reliable agent should respond.

## 11. Limitations of this evaluation

This is a small, educational evaluation set, not a general benchmark. Several test cases are similar to the agent's own examples. We do not use a blinded holdout set, systematic human annotation, or confidence intervals for these very small samples.

The current script evaluates tool **names**, not exact tool-call arguments or complete chronological execution traces. It stores raw tool outputs for subsequent inspection. It also does not automatically validate all numbers in free-form final answers.

For a stronger study, add varied datasets, structured model outputs, independent reference calculations, repeated runs, human-review rubrics, and separate development and test sets.

## 12. Key takeaways

- **Offline testing** establishes that statistical tools work independently of the LLM.
- **Agent evaluation** checks tool selection, execution, and final-answer quality as separate dimensions.
- Required and optional tools should be distinguished.
- Correct-looking output alone does not establish reliability.
- Evaluation should be repeated after changing agent instructions, tools, or models.

**Next: Lesson 07 — LLMOps.** We will reuse these evaluation cases while adding tracing, structured logs, latency and token monitoring, and configuration versioning.

## References

- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [Running Agents](https://openai.github.io/openai-agents-python/running_agents/)
- [Python unittest](https://docs.python.org/3/library/unittest.html)
- [statsmodels](https://www.statsmodels.org/stable/)
