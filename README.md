# Statistical Research AI Agent — A Hands-On Tutorial

**A ten-lesson, Python-based introduction to building, evaluating, and operating statistical research agents with the OpenAI Agents SDK.**

Start with a basic LLM API request; progressively add conversations, Python function tools, reproducible statistical analysis, multi-step orchestration, evaluation, monitoring, retrieval-augmented generation (RAG), multiple specialised agents, and advanced task planning.

The tutorial uses **synthetic biomedical-style datasets** for statistical exercises and publicly accessible scientific abstracts for literature retrieval. It is intended for learning, **not clinical decision-making**. 
## Learning path

| Lesson | Folder | What you will build |
| --- | --- | --- |
| 01 | `01_llm_api/` | Make your first OpenAI API calls and inspect responses. |
| 02 | `02_basic_agent/` | Create an agent and maintain interactive conversations. |
| 03 | `03_function_calling/` | Expose Python functions as tools; inspect tool schemas and compare tool descriptions. |
| 04 | `04_data_analysis/` | Let an agent inspect CSV data, summarise variables, and fit regressions. |
| 05 | `05_multi_step_agent/` | Orchestrate data inspection, OLS models, and regression diagnostics. |
| 06 | `06_evaluation/` | Test tool decisions, failures, numerical results, and interpretation quality. |
| 07 | `07_llmops/` | Trace and monitor runs; record latency, tokens, and configuration changes. |
| 08 | `08_rag/` | Retrieve scientific abstracts; implement embeddings, NumPy similarity search, RAG, web search, and **Qdrant Cloud** retrieval. |
| 09 | `09_multi_agent/` | Compare handoffs and agents-as-tools; combine statistical and literature specialists. |
| 10 | `10_advanced_orchestration/` | Build validated plans, select specialist roles, execute independent tasks concurrently, and demonstrate bounded replanning. |

Follow the lessons in order if you're new to agents. Each folder has its own **.md** tutorial, scripts, and (where applicable) offline tests. Run scripts **from their lesson directory** so relative file paths resolve correctly.

## Prerequisites

- **Python:** A recent Python 3.11 or 3.12 environment is a sensible starting point for binary scientific dependencies. Other versions may work depending on the installed package wheels. You can reuse an existing working virtual environment; do not install another Python just for this tutorial.
- **Editor and terminal:** PyCharm, VS Code, or a terminal. Commands below are written for **Windows PowerShell**; macOS/Linux users can adapt activation and copy commands.
- **OpenAI API key and API billing** for the live model, embedding, and web-search examples. A ChatGPT subscription is separate from API billing.
- **Internet access** for downloading packages, retrieving Europe PMC abstracts, and accessing hosted APIs.
- **Optional for Lessons 08–10:** A Qdrant Cloud database endpoint and **database API key**. The basic RAG and offline lessons do not require Qdrant.

## Quick start

From the **repository root**, activate your existing virtual environment or create one if you do not already have one:

```powershell
# Optional: only when you do not already have a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install the packages used across the ten lessons
python -m pip install -r requirements.txt

# Create a local configuration file, then fill in your own values
Copy-Item .env.example .env
```

If you already have an active `.venv` in a parent directory (as in a PyCharm project), **continue using it**. Confirm that `python -m pip` targets that environment before installing packages. Running `pip` via `python -m pip` helps avoid interpreter mismatches.

Edit `.env`:

```dotenv
OPENAI_API_KEY=your_actual_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
OPENAI_EMBED_MODEL=text-embedding-3-small

# Only needed for optional Qdrant exercises
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=
```

**Never commit `.env` or API credentials.** Each lesson may also contain its own `.env.example`; the root file provides the common defaults. `python-dotenv` normally searches upwards from a script, so the root `.env` can be used for lessons with `load_dotenv()`; when a lesson uses its own configuration or requests a local `.env`, follow that lesson's .md file. Environment variables already set in your shell may override `.env` entries.

Then choose a lesson:

```powershell
cd 04_data_analysis
python test_tools.py
python interactive_agent.py
```

Or try a no-API advanced orchestration example:

```powershell
cd 10_advanced_orchestration
python offline_tests.py
python 01_planning.py
```

Other useful offline checks are `python offline_tests.py` in Lessons 06–10 (where present) and `python test_tools.py` in Lessons 04–05. These verify local code but **do not certify live LLM behaviour**.

## About Lesson 08: Retrieval and vector database setup

The RAG module is deliberately split into distinct stages:

1. Query **Europe PMC** for public article abstracts and cache the corpus.
2. Compute OpenAI text embeddings, then retrieve passages by explicit NumPy cosine similarity.
3. Feed retrieved evidence to a language model for grounded answers with source identifiers.
4. Compare this with OpenAI-hosted **web search** and agentic retrieval.
5. Optionally store the **same cached embeddings** in **Qdrant Cloud** and compare database-backed search with NumPy.

For the Qdrant version, put `QDRANT_URL` and `QDRANT_API_KEY` in `.env` and run from `08_rag/`:

```powershell
python 08_qdrant_retrieval.py
python 09_compare_retrieval.py --top-k 5
```

The current version of Lesson 08 uses **Qdrant Cloud, not ChromaDB**; the root requirements therefore do **not** install `chromadb` or `chroma-hnswlib`. Qdrant is a third-party service: upload only content you are authorised to send. Public abstracts are suitable for this educational example; avoid uploading confidential biomedical records.

## Project conventions and reproducibility

- **Code vs. model reasoning:** Python statistical functions produce numbers; the LLM selects tools and interprets outputs. A plausible explanation is not proof of numerical correctness.
- **Evaluation:** Lesson 06 evaluates task behaviour; Lesson 07 adds operational monitoring. Lesson 09 compares architectures; Lesson 10 demonstrates bounded planning and a simulation-based sequential/parallel timing experiment.
- **Generated data and caches:** Some lessons write results such as `cache/`, `.jsonl`, and `.csv` output files. Keep source code and small example datasets under version control, while excluding local caches, secrets, virtual environments, and sensitive logs.
- **Run locations:** The examples are designed to run from their own lesson folders. Some import sibling `.py` modules or load CSVs relative to the current working directory.
- **API reproducibility:** Model outputs, web search results, costs, and latency vary over time. Repeat evaluations, keep configuration records, and verify sources and statistics independently.

## Common troubleshooting

| Symptom | Check |
| --- | --- |
| `ModuleNotFoundError` | Verify the selected PyCharm interpreter, activate the intended `.venv`, and reinstall with `python -m pip install -r requirements.txt`. |
| Missing API key | Ensure `.env` is configured; start the script in the lesson directory and check whether the shell overrides your environment variables. |
| `403 Forbidden` from Qdrant | Use the **database API key** for the correct cluster endpoint and verify collection permissions. A Cloud Management API key is not interchangeable. |
| No RAG corpus found | Run `01_retrieve_research_text.py` first; corpus and embedding caches are created by earlier scripts. |
| Tool selection differs from an example | Agents are nondeterministic; inspect calls and use evaluation cases rather than assuming the same trace every time. |
| Unexpected API cost | Begin with offline tests and a single run; repeated live evaluations and web searches can incur charges. |

## Security and responsible use

Keep credentials in `.env`, use the repository-level `.gitignore`, and do not upload identifiable patient data to external model or retrieval providers without appropriate authorisation. Verify scientific claims and citations, especially those based only on abstracts. Inspect tools before allowing agents to run code or access external systems.

## Where to go next

Complete Lessons 01–10, then use the techniques to develop your own research problem: a statistical dataset, appropriate validated analysis tools, retrieval of relevant evidence, and a tested agent workflow. More elaborate agent structures or fine-tuning are optional—not prerequisites for a useful system.
