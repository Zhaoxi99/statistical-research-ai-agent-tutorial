# Lesson 10 — Advanced Agent Orchestration

**Level:** after Lesson 09 (Multi-Agent Systems).  **Goal:** learn the *engineering* of planning, role configuration, concurrency, dependency management, and bounded replanning. This is a standalone folder: its statistical datasets and tools are included.

## What changes from Lesson 09?

In Lesson 09, we defined a research coordinator and a fixed pair of specialist agents. Lesson 10 asks how to plan several steps **explicitly**, select from pre-approved specialist roles at runtime, run independent steps in parallel, and recover from failures. Multi-agent is not synonymous with good planning; explicit constraints and evaluation matter.

| Technique | Key question | Where used |
| --- | --- | --- |
| Planning | What are the subtasks, and which depend on others? | `planning.py`, `01_planning.py` |
| Dynamic specialists | Which approved capabilities are needed? | `specialists/factory.py`, `02_dynamic_specialists.py` |
| Parallel tasks | Which tasks can run at the same time? | `orchestration.py`, `03_parallel_workflow.py` |
| Bounded replanning | What happens when a step fails? | `04_replanning.py` |
| Full system | How are findings synthesised into a report? | `05_research_orchestrator.py` |
| Evaluation | Does parallelism reduce elapsed time? | `06_compare_orchestration.py` |

## 1. Environment

In PowerShell, open `10_advanced_orchestration` using your existing project `.venv`:

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env to include your OPENAI_API_KEY
python offline_tests.py
```

In PyCharm confirm that the configured interpreter is the same `.venv` used by the terminal. The supplied statistical data are **synthetic**, not clinical records. Optional Qdrant requires the existing Lesson 08 Qdrant Cloud credentials and a previously indexed collection; it is not required for any offline example. Web search incurs model/tool usage costs.

## 2. Explicit plans and dependency graphs

A plan is a small directed acyclic graph (DAG). A step has an ID, a plain-English objective, an approved role, and `depends_on` IDs. For example:

```text
    analysis ---------\
                        >---- review ---> final synthesis
    literature -------/
```

Analysis and literature search have no dependencies and can run concurrently. Review waits for **both**. The data class `ResearchPlan` checks that IDs are unique, roles are pre-approved, and dependencies refer only to earlier steps. We bound each plan to at most six steps to discourage endless plans.

```powershell
python 01_planning.py
```

Expected: JSON for a deterministic, validated three-step plan, no API call. To ask an LLM to propose a plan, run:

```powershell
python 01_planning.py --live
```

`Agent(..., output_type=ResearchPlan)` requests structured output; Pydantic then validates the parsed plan. The model may still produce a *bad research plan* that passes structural validation. Human oversight and substantive validation remain necessary.

**Experiment:** Try a plan with a forward dependency and see why validation rejects it.

## 3. Dynamic specialists, safely

Dynamic creation does **not** train a new model. It configures a new `Agent` object (instructions and tool permissions) for an approved role. We use a deterministic role selector in `choose_roles()`. The three possible roles are `statistics`, `literature`, and `reviewer`.

```powershell
python 02_dynamic_specialists.py
python 02_dynamic_specialists.py --live
```

The first run prints selected roles and tool counts without making API calls. The second also asks a statistical specialist to analyse the bundled dataset. `make_specialist()` refuses unknown roles. This is safer than accepting model-generated Python code or arbitrary tool permissions. The literature specialist has hosted web search, with optional Qdrant retrieval enabled using `--use-qdrant` in the full workflow.

## 4. Parallel execution and a synchronization barrier

We use `asyncio.gather` to launch *ready* steps. A semaphore caps simultaneous specialists at `max_parallel` (default: two). Each wave completes before downstream tasks can start. Thus this is parallel *I/O concurrency*, not multiple CPU cores simultaneously running Python.

```powershell
python 03_parallel_workflow.py
python 03_parallel_workflow.py --max-parallel 1
python 03_parallel_workflow.py --live
```

**Default:** safe offline simulation; it performs a short `asyncio.sleep()` per task, returns synthetic placeholder results, and displays waves `[['analysis', 'literature'], ['review']]`. **`--live`:** calls specialist LLMs and web search, so it costs tokens and network access. Timing and actual tool choices vary.

**Note about concurrency:** both tasks are launched in the same wave, but `--max-parallel 1` executes them one at a time. If the first task fails, the worker reports `failed`; its downstream reviewer step is marked `skipped`. Other independent steps may still complete.

## 5. Bounded replanning

A fully autonomous system could keep generating more tasks indefinitely. That is undesirable for a beginner project, and risky for scientific research. Here `revise_plan()` demonstrates *one explicit recovery decision* when literature retrieval fails: retry with a narrower search, then review. It does **not** automatically execute retries indefinitely.

```powershell
python 04_replanning.py
```

Expected: a new two-step plan, printed as JSON. **Experiment:** change the simulated literature outcome to `ok` and see that no replacement is proposed.

## 6. End-to-end orchestration

```powershell
python 05_research_orchestrator.py
# Optional if you already indexed a Lesson 08 Qdrant collection:
python 05_research_orchestrator.py --use-qdrant
```

Flow: construct fixed validated plan → run analysis and literature specialists in parallel → pass their outputs to reviewer → ask a final coordinator to synthesise. The statistical specialist uses bundled statistical tools (regression and diagnostics). Literature search uses OpenAI WebSearchTool, plus optional Qdrant. If a step fails, the final coordinator sees that failure rather than a fabricated output. This is not an automated clinical-decision system.

**Caveat:** the bounded replanning demo is intentionally separate from the live workflow. Combining automatic replanning with tool retries would need additional budgets, approval policy, observability, and robust stopping criteria.

## 7. Comparing sequential and parallel workflows

```powershell
python 06_compare_orchestration.py --repeats 5
```

Output: mean elapsed times and `orchestration_results.csv`. This example uses **simulated asynchronous work**, so that latency differences are reproducible and API-free. It is **not** evidence that live multi-agent research is better or more accurate. For a real benchmark, combine latency and token monitoring from Lesson 07 with grounded-answer and statistical-correctness criteria from Lesson 06.

Metrics to consider: task completion, dependency correctness, tool error rate, factual support, numeric accuracy, latency, and token usage. Parallel execution reduces wall-clock time only when sufficient work is independent; it can increase API request concurrency and cost pressure.

## 8. Offline test suite

```powershell
python offline_tests.py
```

Tests cover: valid plans, forward-dependency rejection, approved roles, correct dependency waves, failure propagation, and bounded replanning. These tests do not prove quality of live LLM outputs.

## 9. Common problems

- **`ModuleNotFoundError: agents`** — ensure `openai-agents` is installed in the interpreter running your scripts.
- **`OPENAI_API_KEY` missing** — copy `.env.example` to `.env`, add your key and run from this directory.
- **Web search access or cost issues** — use offline scripts first; live research relies on supported OpenAI tool access.
- **Qdrant 403** — check `QDRANT_URL`, collection permissions, and database API key. Never commit `.env`.
- **No speedup** — expected if tasks are sequentially dependent or if overhead dominates.
- **Some reviewer conclusions seem unsupported** — verify references and the numerical results, not just successful API calls.

## 10. Suggested exercises

1. Add a new approved `visualisation` role that only generates a chart description, without running untrusted code.
2. Compare `--max-parallel 1` and `--max-parallel 2`, explain why the dependency waves stay the same.
3. Simulate a failed analysis. Verify that the reviewer is skipped and the literature step still succeeds.
4. Add a `max_retry=1` budget, requiring explicit approval before any automatic retry.
5. Record token consumption per specialist and compare with a single-agent baseline from Lesson 09.

## Further reading

- [OpenAI Agents SDK documentation](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents SDK — Agents and tools](https://openai.github.io/openai-agents-python/agents/)
- [Python asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [Pydantic models and validation](https://docs.pydantic.dev/latest/)

**Next:** After Lesson 10, you can build a practical capstone with a user-facing interface, deployment, scientific-quality checks, and a complete monitoring/evaluation pipeline.
