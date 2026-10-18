# Lesson 09 — Multi-Agent Systems: A Statistical Research Team

**Audience:** Python beginners who completed Lessons 01–08.  
**Outcome:** Compare handoffs with manager-style delegation, build a coordinator with statistical and literature specialists, exchange validated structured outputs, and evaluate against a single-agent baseline.

## 1. Why multiple agents?

In Lesson 05 one agent could run statistical tools. In Lesson 08 an agent could find evidence on the web or in Qdrant. Lesson 09 puts these capabilities together by **delegating** to specialists. An agent is not necessarily a different model: the same underlying LLM can have a different system prompt and a restricted set of tools in each role.

Our real question is *not* “can we make three agents?”, but: **when does splitting the work improve quality enough to justify extra calls, latency and complexity?** Don't assume more agents are better.

### Two patterns

- **Handoff / routing:** A triage agent transfers control to a specialist. The specialist may own the final answer. Ideal when a user request mainly belongs to one domain.
- **Agent as tool / manager:** The coordinator calls specialist agents as tools, receives their findings and remains responsible for synthesising a final answer. Ideal when a request combines multiple domains.

A handoff isn't simply a regular Python function call; it changes which agent owns the running conversation. An agent-as-tool is a *nested* agent execution, controlled by the manager.

## 2. Project layout

```text
09_multi_agent/
├── README.md
├── 01_basic_handoff.py              # Transfer control to one specialist
├── 02_agents_as_tools.py            # Manager calls specialist agents
├── 03_statistical_research_team.py  # Combined multi-step report
├── 04_structured_communication.py   # Pydantic intermediate results
├── 05_evaluate_multi_agent.py       # Repeated single-vs-multi comparison
├── offline_tests.py                # No OpenAI API or network required
├── requirements.txt
├── .env.example
├── .gitignore
├── specialists/
│   ├── __init__.py
│   ├── statistical_agent.py
│   ├── literature_agent.py
│   ├── coordinator_agent.py
│   └── schemas.py
├── tools/
│   ├── __init__.py
│   ├── statistical_tools.py        # Plain, deterministic Python
│   ├── agent_tools.py              # @function_tool interfaces
│   └── retrieval_tools.py          # Optional Lesson 08 Qdrant search
└── data/
    ├── biomarker.csv
    └── biomarker_replication.csv
```

We deliberately use the package name `specialists` instead of `agents`: naming a local directory `agents/` would **shadow the installed OpenAI Agents SDK** and cause confusing imports.

## 3. Setup — PowerShell and PyCharm

Unzip into your tutorial repository, activate the **same** `.venv` you used in previous lessons, and open a terminal inside `09_multi_agent/`.

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and fill `OPENAI_API_KEY` (keep `.env` out of Git). The OpenAI API has usage costs. Scripts 01–05 call the API; `offline_tests.py` does not.

**Optional Qdrant:** If you want your Literature Specialist to search the pre-indexed Lesson 08 abstracts, add `QDRANT_URL`, `QDRANT_API_KEY` and the actual **existing** `QDRANT_COLLECTION` to `.env`, and add `--use-qdrant` when running scripts 02 or 03. The tools don't automatically recreate or re-embed the full research corpus. Qdrant text may be absent in a different collection schema, so inspect returned metadata. The default literature agent uses live OpenAI web search; Qdrant is optional.

## 4. Start with deterministic statistical functions

```powershell
python offline_tests.py
```

Six offline tests verify dataset validation, summaries, OLS, diagnostics, error handling and schema validation. The bundled biomedical data are **synthetic**. Results are associations, not causal effects. Shapiro–Wilk tests residual normality and Breusch–Pagan provides one diagnostic of heteroskedasticity; neither a high p-value nor a low p-value exhausts model checking.

Why keep plain Python functions? So you can debug the *statistics* independently from the *agent*. In `tools/statistical_tools.py`, `regression()` returns actual coefficients and p-values. The tool interfaces in `tools/agent_tools.py` expose narrowly typed inputs to the LLM, and catch common validation errors.

## 5. Stage 1: Handoffs

```powershell
python 01_basic_handoff.py
python 01_basic_handoff.py --question "Find recent methodological guidance for OLS regression diagnostics and cite sources."
```

**Expected:** The first question generally routes to the Statistical Specialist; the second to the Literature Specialist. Look at `Final agent:`. Routing is decided by the model and isn't guaranteed for every prompt. The `handoffs=[...]` list tells the SDK which transfer targets exist.

**Try:** Ask for both statistics and literature in one query. Why is a one-way handoff less convenient for synthesising both? Once a specialist receives control, the original triage agent does not automatically resume to coordinate another specialist.

## 6. Stage 2: Agents as tools

```powershell
python 02_agents_as_tools.py
```

The coordinator has two tool-like interfaces created by `.as_tool()`:

- `statistical_specialist`: inspects synthetic data, fits OLS and checks assumptions.
- `literature_specialist`: looks for methodological evidence through OpenAI Web Search, optionally through Qdrant too.

The coordinator is instructed to delegate both pieces when the question requires them. Its **own** final output synthesises those results. Running additional specialist agents means additional LLM calls and expense. The coordinator may still choose an imperfect plan, so verify the trace.

`Agent.as_tool(...)` is different from `@function_tool`: the former invokes another agent with its own instructions and tools; the latter wraps a Python function.

## 7. Stage 3: Full research team

```powershell
python 03_statistical_research_team.py
```

The default question requests four actions: inspect `biomarker` data; fit `biomarker ~ age + sex`; check selected OLS diagnostics; consult external literature on diagnostic limitations. Expected final report: grounded regression results, diagnostic caveats, links to literature, and no unsupported causal statements.

The script prints *top-level* specialist tool calls from `result.new_items`. Nested tools run **inside** specialists and may not appear in that top-level list. Use the Agents SDK tracing dashboard to inspect detailed nested activity. Built-in traces may capture content; use synthetic data only in this tutorial and review privacy settings for real biomedical data.

If the agent does not call one of the specialists, that is a meaningful limitation to record, not something to silently hide.

## 8. Stage 4: Structured communication

```powershell
python 04_structured_communication.py
```

Pydantic describes *what shape* an intermediate specialist result should have. In `specialists/schemas.py`, a `StatisticalFinding` contains dataset, outcome, predictors, sample size, coefficients, limitations. A `LiteratureFinding` contains summary, actual source URLs and limitations.

This example uses a controlled sequence: statistics agent -> literature agent -> evidence formatter -> report writer. That makes structured messages easier to inspect, although it is **not** the same as a free-form autonomous coordinator. The extra formatter call is a teaching choice; in production you could directly give a literature specialist `output_type=LiteratureFinding` and avoid the formatter.

**Key limitation:** Valid JSON/Pydantic output is **not** proof that coefficients or sources are true. Independently compare numerical values against `regression()` and verify cited URLs.

## 9. Stage 5: Compare one agent versus a team

```powershell
python 05_evaluate_multi_agent.py --repeats 1
python 05_evaluate_multi_agent.py --repeats 3
```

Each repeat makes **two** live runs: a single agent with all the relevant tools, then the manager with two specialist agents. A 3-repeat experiment means 6 top-level runs, plus specialist calls inside the multi-agent runs. API charges may increase. Each record is appended to `evaluation_runs.jsonl` (ignored by Git).

Saved fields: architecture, repetition, latency, observed total tokens if SDK exposes them, top-level tool calls, successful execution, final answer and error. **Important:** `run_success=True` only means the SDK run finished. It is not a statistical correctness score. Also, nested token accounting may vary with SDK version; verify trace-level usage before claiming an apples-to-apples cost comparison. Answers and questions may contain sensitive information; local evaluation logs save generated answers for review, so don't use patient data without a suitable logging/privacy policy.

For a serious benchmark, add a manually labelled rubric:

| Criterion | How to inspect |
| --- | --- |
| Regression correctness | Compare `age`, `sex` and intercept to deterministic OLS |
| Diagnostic completion | Confirm Shapiro–Wilk, Breusch–Pagan, Cook's distance used |
| Evidence quality | Verify cited source URLs and claim-source support |
| Coverage | Did both statistical and literature subtasks finish? |
| Causal caution | Does the answer distinguish association from causation? |
| Efficiency | Compare latency, top-level calls, trace-level total tokens |

Keep model/version and evaluation questions fixed; run more repetitions to assess variability. Prompt differences between architectures are unavoidable, so interpret the results as an *architecture-and-instructions comparison*, not an isolated causal effect of agent count.

## 10. Exercises

1. Ask a statistics-only question and observe whether the manager unnecessarily calls literature search.
2. Add one deliberately nonexistent dataset name. Check that it reports an error instead of inventing a result.
3. Modify the Statistical Specialist instructions to request a missingness table before fitting OLS. Verify actual calls in traces.
4. Run the optional Qdrant version. Compare its answer with the live web-search answer, noting recency and source coverage.
5. Add an explicit fail condition to your evaluation when a report contains no numerical coefficients or no source links.
6. Investigate when the single-agent baseline is faster or more reliable than the multi-agent version.

## 11. What is intentionally not covered here?

This lesson uses **predefined** specialists. Autonomous plan–execute–replan, on-the-fly specialist generation, and complex concurrent fan-out/fan-in execution are advanced orchestration patterns for a later lesson. Don't introduce them until simpler alternatives have been measured.

## 12. Troubleshooting

- `No module named agents`: install **`openai-agents`**, not the unrelated package named `agents`. Ensure `python -m pip` targets your active `.venv`.
- `OPENAI_API_KEY` missing: copy `.env.example` to `.env`, fill the key, run from this directory.
- Literature sources absent: web search may be blocked or insufficient. The agent should say so; it must not invent citations.
- Qdrant HTTP 403: verify the **database** API key and matching cluster URL (not a Cloud Management key). Never print keys.
- Local ChromaDB crashing on Windows: this lesson deliberately **does not** import ChromaDB or use native HNSW locally.
- Unexpected columns: use `python offline_tests.py`; the actual bundled CSV columns are validated by the tools.
- `agents` import conflicts: ensure the local folder is named `specialists`, not `agents`.

## References

- [OpenAI Agents SDK — agents and patterns](https://openai.github.io/openai-agents-python/agents/)
- [OpenAI Agents SDK — handoffs](https://openai.github.io/openai-agents-python/handoffs/)
- [OpenAI Agents SDK — running agents](https://openai.github.io/openai-agents-python/running_agents/)
- [OpenAI Agents SDK — tools](https://openai.github.io/openai-agents-python/tools/)
- [Qdrant documentation](https://qdrant.tech/documentation/)

**Next:** Lesson 10 could investigate planning, dynamic specialists and controlled parallel workflows.
