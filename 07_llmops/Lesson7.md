# Lesson 07: LLMOps — Tracing, Monitoring and Reproducibility

## Learning Objectives

In Lessons 01–05, we built a statistical research AI agent; in Lesson 06, we evaluated whether it selected appropriate tools and produced reasonable statistical outputs. In this lesson, we learn to **operate** the agent: observe what happens during a run, record measurements, and compare versions.

By the end, you should be able to:

- Explain the difference between **evaluation** and **LLMOps**.
- Inspect model/tool activity using **traces**.
- Record **latency, tool counts, errors and token usage** in a machine-readable log.
- Summarise and compare runs under two agent configurations.
- Understand why operational metrics **do not prove answer correctness**.
- Keep API credentials and private research data out of GitHub and local telemetry logs.

You don't need Docker, a cloud server, or a database to follow this lesson.

## 1. What Is LLMOps?

**LLMOps (Large Language Model Operations)** encompasses practices for developing, testing, releasing, observing and improving LLM applications. It is related to MLOps, but addresses additional concerns such as prompt versions, external tools, context, token costs, and non-deterministic outputs.

Two questions help distinguish our last two lessons:

| Lesson 06 — Evaluation | Lesson 07 — LLMOps |
|---|---|
| Did the agent choose the appropriate tools? | Which tools did the agent actually call in each run? |
| Were numbers and conclusions correct? | How many tokens and seconds did the run consume? |
| How often did the agent pass the test cases? | Did reliability, cost or latency change after an update? |

**Evaluation is a component of LLMOps.** This lesson reuses our statistical agent but deliberately does not repeat Lesson 06's scoring logic.

## 2. Project Files and Setup

```text
07_llmops/
├── Lesson7.md
├── research_agent.py              # Copy of the Lesson 06 agent
├── data_tools.py                  # Reusable pandas/statsmodels functions
├── diagnostic_tools.py            # Regression diagnostics
├── biomarker.csv                  # Synthetic dataset
├── biomarker_replication.csv      # Second synthetic dataset
├── evaluation_cases.py            # Reusable future evaluation prompts
├── monitoring.py                  # Run wrapper that records telemetry
├── run_monitored.py               # Run and monitor one question repeatedly
├── summarize_runs.py              # Summarise JSONL run logs
├── compare_configs.py             # Compare two instruction configurations
├── offline_tests.py               # Check logging without API access
├── .env.example                   # Placeholder (no real key)
└── .gitignore                     # Exclude local credentials and logs
```

From inside this folder, install:

```bash
python -m pip install openai-agents openai python-dotenv pandas numpy scipy statsmodels
```

Configure `OPENAI_API_KEY` as in previous lessons. The `.env` file is a **template**, not a working credential file. A local `.env` file is ignored by Git.

Run scripts from the **`07_llmops` directory** so that local imports and dataset paths resolve correctly.

## 3. Tracing: What Happened Inside the Agent?

When an agent answers a question, it may make multiple model calls and invoke several tools. A final answer alone cannot tell us what happened.

A **trace** records an observable sequence of execution events. Depending on the SDK and configuration, traces can include model requests, tool calls, handoffs, timing and failures. Tracing does **not** necessarily expose a model's private internal reasoning.

In `monitoring.py`, we use:

```python
from agents import Runner, RunConfig, trace

with trace('research-agent-baseline'):
    result = Runner.run_sync(
        agent,
        question,
        max_turns=12,
        run_config=RunConfig(trace_include_sensitive_data=False),
    )
```

`max_turns=12` helps prevent an unexpectedly long tool-using workflow. `trace_include_sensitive_data=False` reduces sensitive content included in SDK traces; review SDK tracing configuration and your data-handling policies before using real patient information.

If supported for your account and configuration, inspect traces at [OpenAI Platform Traces](https://platform.openai.com/traces). Actual visibility depends on tracing being enabled and exported successfully.

**Try it:**

```bash
python run_monitored.py --repeats 5
```

Use `--repeats 1` (the default) for a quick test, or increase the number to study run-to-run variability. Each repetition is an independent agent execution and creates a separate log record; API usage and cost increase accordingly.

This runs the familiar research question: investigate age and biomarker levels, adjusting for sex, and check regression assumptions.

In your terminal, look for the `TOOL EXECUTED:` messages. In the trace viewer, inspect the sequence of model and tool events. The order is selected by the agent, so it need not be identical between runs.

## 4. Local Monitoring: From Individual Traces to Measurements

While tracing is useful for debugging an individual run, monitoring creates **structured metrics** that can be summarised across many runs.

Our `run_monitored()` wrapper in `monitoring.py` records:

| Field | Meaning |
|---|---|
| `timestamp_utc` | When the run finished (UTC) |
| `config_id` | A label identifying the agent configuration |
| `status` | Whether the SDK run completed or raised an exception |
| `error_type` | The exception class for a failed run |
| `latency_seconds` | Elapsed wall-clock execution time |
| `tool_names` | Tool requests made during the run |
| `tool_calls` and `tool_outputs` | Number of observed tool call/output events |
| `input_tokens`, `output_tokens`, `total_tokens` | Token usage when reported by the SDK |

We save **one JSON object per line** in `agent_runs.jsonl` (JSON Lines format). For example, this is an *illustrative* record, not measured performance:

```json
{
  "config_id": "baseline",
  "status": "success",
  "latency_seconds": 4.2,
  "tool_names": ["inspect_data", "fit_regression", "check_regression_assumptions"],
  "tool_calls": 3,
  "tool_outputs": 3,
  "input_tokens": 1200,
  "output_tokens": 350,
  "total_tokens": 1550
}
```

The actual log contains additional fields. Token counts may be missing for failed runs; missing values are recorded as `null` rather than guessed.

**What do we intentionally avoid logging?** Prompts, model answers, tool arguments, full tool outputs and API keys. This minimises accidental disclosure, but the tool names, timing and run metadata may still be sensitive in some settings. Logs are ignored by Git and should be retained only as needed.

**Note:** `status="success"` only means the agent run returned without throwing an exception. It does **not** mean the answer was correct or all tools succeeded. Lesson 06 covers quality checks.

## 5. Summarise the Run Logs

After running `run_monitored.py`, execute:

```bash
python summarize_runs.py
```

The script reads the JSONL file, groups runs by `config_id`, and reports:

- Number of runs and proportion with successful SDK completion.
- Mean and median latency.
- Average token usage when available.
- Counts of tools requested.

Example (illustrative):

```text
baseline:
  runs: 3
  success_rate: 1.0
  mean_latency_seconds: 5.4
  median_latency_seconds: 5.1
  mean_total_tokens: 1800.0
  token_records_available: 3
  tool_counts: {'inspect_data': 3, 'fit_regression': 3}
```

If you have not executed any live runs, the script explains that no log exists. It does not make an API call.

## 6. Track Versions: Compare Two Agent Configurations

A useful LLMOps question is whether a change to the agent **increases latency or cost without improving quality**.

`compare_configs.py` compares two instruction configurations while keeping the same dataset, model choice and available tools:

- **baseline:** The existing multi-step research instructions.
- **concise:** Shorter instructions emphasising numerical tool use and concise answers.

Run:

```bash
python compare_configs.py --repeats 5
python summarize_runs.py
```

This makes **five independent runs per configuration (ten runs total)**. Each run is logged separately. Runs are interleaved by configuration to lessen time/order effects. Start with `--repeats 1` to check your API setup.

The monitoring wrapper labels the records using the configuration names, so you can compare tool call counts, tokens and latency in one summary.

### Important: Avoid confounding

For a stronger experiment, keep the model identifier, dataset, SDK version and test cases fixed. Repeat runs for each configuration and **evaluate answer quality using Lesson 06's methodology**. Do not conclude that faster responses are better without examining accuracy.

Model identifiers and settings can change over time. In a mature project, you should additionally record the Git commit, package versions, model snapshot (when available), and evaluation dataset version.

## 7. Offline Test: Verify the Monitoring Code

Before spending API tokens, test the log aggregation using synthetic monitoring records:

```bash
python offline_tests.py
```

Expected end of output:

```text
PASS: offline log aggregation and privacy fields
```

This test confirms that the summary calculations work and verifies that the example record contains no prompt or final-answer field. It does not validate real API telemetry.

## 8. Experiments

### Experiment A: Simple versus complex questions

In `run_monitored.py`, compare a question that only asks for the column names with one that requests a full regression investigation. Record tool calls, latency and token usage. Do more complex requests consistently require more resources?

### Experiment B: Repeated runs

Run the same prompt three to five times with the same configuration. Compare tool paths, timing and token use. Different runs may take different paths even when the final answers are similar.

### Experiment C: Configuration comparison

Run `compare_configs.py` repeatedly. After collecting logs, reuse your Lesson 06 test cases to compare the **quality–latency–token trade-off**. The provided script illustrates configuration tracking; it is not a statistically powered comparison.

### Experiment D: Failure handling

Try an invalid dataset name. The tool may return an error dictionary while the overall SDK run still completes. Explain why operational *run success* and statistical *task success* are different metrics.

## 9. Limitations and Safety

- Traces are useful for debugging, but logging should be reviewed before handling real biomedical or confidential data.
- `trace_include_sensitive_data=False` is a mitigation, not a guarantee that all metadata is non-sensitive.
- The local JSONL logger tracks SDK-level exceptions, not every error returned *inside* a tool output.
- Token usage may be unavailable for a failed run or depend on SDK configuration.
- Latency depends on network conditions and server load; isolated timings are not reliable estimates of performance differences.
- API token counts are **not** currency costs. Cost estimation also requires the relevant model pricing and billing information.
- Operational metrics should be combined with correctness evaluation before deploying an agent for consequential research work.

## 10. Key Takeaways

1. **Evaluation** measures task quality; **LLMOps** manages and observes application operation and change over time.
2. A **trace** helps explain the tool and model events within an individual agent run.
3. **Structured logs** make it possible to summarise latency, errors, tool usage and token consumption.
4. **Configuration labels** let us compare agent versions, although proper conclusions require repeated tests and quality evaluation.
5. Privacy-aware monitoring avoids recording more user and dataset content than necessary.

**Next extension:** Connect Lesson 06 evaluation scores with this lesson's operational telemetry, then build a lightweight performance report or dashboard.

## References

- [OpenAI Agents SDK — Tracing](https://openai.github.io/openai-agents-python/tracing/)
- [OpenAI Agents SDK — Running Agents](https://openai.github.io/openai-agents-python/running_agents/)
- [OpenAI Agents SDK — Sessions](https://openai.github.io/openai-agents-python/sessions/)
- [OpenAI API — Evaluation](https://platform.openai.com/docs/guides/evals)
