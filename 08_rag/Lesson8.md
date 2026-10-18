# Lesson 08 — Retrieval-Augmented Generation (RAG)

**Goal:** Extend our statistical research assistant with external scientific evidence. We first build retrieval and generation ourselves, then use OpenAI's hosted web search, and finally create a web-enabled agent. **No local PDFs or pre-downloaded research files are needed**: we query Europe PMC for publicly available research abstracts. The application stores fetched abstracts and embeddings in a local `cache/` folder for reuse.

> **Prerequisites:** Python 3.10+, an OpenAI API key with API billing, familiarity with Lessons 01–07. Work in the `08_rag` directory. These scripts use the same key for text generation and embeddings, while Europe PMC metadata search is public and requires no OpenAI key.

## 1. What exactly is RAG?

An LLM's pretrained parameters are not a dependable, up-to-date repository of research papers. **Retrieval-augmented generation** supplies relevant external content to an LLM when answering a question:

```
User question
    |
    v
RETRIEVAL: find relevant text from an external information source
    |
    v
AUGMENTATION: assemble evidence passages, titles and source URLs
    |
    v
GENERATION: answer using evidence, cite sources and disclose gaps
```

RAG is an *application workflow*, not model fine-tuning. Embeddings are one method of retrieval, **not a requirement of RAG**. Keyword/BM25 search and hosted web search are other options. Finding relevant text does **not** make it true, and an LLM can still misinterpret or miscite it.

**What will we build?** A literature assistant answering methodological questions about Bayesian adaptive clinical trials from live academic abstracts. This is an educational demonstration, not a systematic review or clinical decision-support system.

## 2. Setup on Windows / PyCharm

Extract this folder into the root of your existing GitHub tutorial. Open a PowerShell terminal in `08_rag` and activate your existing Python environment. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Copy `.env` to `.env`, replace the placeholder with your own key and keep `.env` private:

```powershell
Copy-Item .env.example .env
```

`OPENAI_MODEL` defaults to `gpt-4.1-mini`; `OPENAI_EMBED_MODEL` defaults to `text-embedding-3-small`. Check whether your API account has access to these models. The `.gitignore` excludes `.env` and `cache/`. Both generation, embeddings and hosted web search may incur API costs; consult current pricing before large experiments.

**Offline smoke test** (no network, no API key):

```powershell
python offline_tests.py
```

Expected: `Ran 8 tests ... OK`. The test covers chunking, cosine ranking, metadata, simple parsing and stable cache keys. This confirms only the local building blocks, **not** live API compatibility.

## 3. Stage 1 — Obtain research evidence from the web

```powershell
python 01_retrieve_research_text.py
```

The script calls the [Europe PMC REST API](https://europepmc.org/RestfulWebService), requests metadata including abstracts, filters out items with very short/missing abstracts, and saves `cache/papers.json` and `cache/chunks.json`. It prints a sample of paper titles and URLs. Exact results vary with the live database and search ranking. If too few papers are found, broaden the search:

```powershell
python 01_retrieve_research_text.py --query "Bayesian clinical trial" --limit 20
```

**Why chunk?** Embedding models operate on text segments, not entire knowledge bases. `split_text(text, size=900, overlap=150)` divides an abstract into potentially overlapping character spans, preserving nearby context. Our simple character splitter may cut sentences; a later production version should use sentence/token-aware splitting, deduplicate papers, and preserve section/page offsets. We keep `source_id`, `title`, `url`, and `chunk_id` for citations and traceability.

**Think about it:** Europe PMC returns article metadata and abstracts, not necessarily full article text. Summaries based only on abstracts can omit crucial methods and caveats. Avoid interpreting this as a comprehensive literature search.

## 4. Stage 2 — Understand embeddings

```powershell
python 02_embeddings.py
```

An **embedding model** converts a string into a numerical vector:

\[e(t) \in \mathbb{R}^d.\]

Related texts may have vectors pointing in similar directions. The script embeds three short examples and prints their cosine similarities. Your actual results depend on the embedding model.

**Cosine similarity** is

\[\operatorname{sim}(u,v)=\frac{u^\top v}{\lVert u\rVert_2\lVert v\rVert_2}.\]

A score near 1 means aligned vector directions, not a probability of relevance, factual truth or source quality. `rag_utils.cosine_search` calculates similarities explicitly with NumPy, so you can inspect every operation. We use a simple exact search rather than FAISS because the corpus is small.

**API distinction:** `client.embeddings.create(...)` computes vectors. `client.responses.create(...)` generates text. The same `OPENAI_API_KEY` can authenticate both; there is no additional embedding-specific key.

## 5. Stage 3 — Semantic retrieval (without generating an answer)

```powershell
python 03_semantic_retrieval.py
```

The program loads the research chunks, calls the embedding API once for the corpus if it is not cached, embeds the query, scores every chunk, and prints the three highest-scoring passages. Try a custom question:

```powershell
python 03_semantic_retrieval.py --question "How are interim analyses used in Bayesian clinical trials?" --top-k 5
```

`top-k` is the number of passages returned. Raising `k` may improve coverage but also add irrelevant material and increase prompt length. **Look at the printed text:** do the top passages actually support your question? Retrieval must be examined separately from text generation.

Caching: corpus embeddings are written to `cache/embeddings_<hash>.npy`. The hash includes chunk contents and embedding model. Query embeddings are recomputed for each run. Rebuilding the text collection produces a new cache entry. Do not commit caches containing sensitive content.

## 6. Stage 4 — Basic RAG: fixed retrieval → generation

```powershell
python 04_basic_rag.py
```

The script implements a deterministic workflow *structure*:

1. Embed the question.
2. Retrieve the top passages.
3. Construct a context string with identifiers `[S1]`, `[S2]` and original URLs.
4. Call the LLM to produce an evidence-grounded answer.

The retrieval and model outputs are not deterministic. Try `--top-k 2` and `--top-k 6`: note differences in completeness and citation coverage.

**Inspect `rag_utils.answer_with_context`:** the model is instructed to rely on provided passages, use `[S#]` citations and explicitly state when evidence is insufficient. These are *instructions*, not guarantees. You must verify that each cited passage supports the specific claim. A linked abstract is not a substitute for checking the paper.

**Security note:** retrieved documents can contain malicious instructions. The function explicitly treats retrieved text as untrusted data, but that reduces rather than eliminates prompt-injection risk.

## 7. Stage 5 — OpenAI built-in web search

```powershell
python 05_openai_web_search.py
```

This uses the official Responses API syntax:

```python
response = client.responses.create(
    model=MODEL,
    tools=[{"type": "web_search", "search_context_size": "medium"}],
    tool_choice="required",
    input=question,
)
```

**What changed?** We no longer implement a corpus, embeddings, chunks or vector ranking. The hosted tool handles web searching and supplies material to the model. `tool_choice="required"` requests a tool call for this example. Web search can be useful for fresh findings, but available sources and search behaviour are partly opaque. It is not equivalent to testing our NumPy retriever.

Reference: [OpenAI web search documentation](https://developers.openai.com/api/docs/guides/tools-web-search).

## 8. Stage 6 — Agentic web research

```powershell
python 06_agentic_rag.py
```

This uses `Agent`, `Runner.run_sync`, and `WebSearchTool` from the **OpenAI Agents SDK**, just as in your earlier lessons. The agent receives a question, may search the web, and can use search outputs to prepare an answer. Unlike basic RAG's fixed sequence, search use and the reasoning workflow are model-directed. For current research questions, this may involve one or multiple searches; multiple searches are **not guaranteed**.

Try contrasting:

```powershell
python 06_agentic_rag.py --question "Explain what a prior distribution is."
python 06_agentic_rag.py --question "Compare recent academic reviews of Bayesian adaptive clinical trials and cite URLs."
```

A conceptual question might not need search; a question about recent papers generally should. The second may incur more search cost. The SDK can expose traces for studying tool behaviour (Lesson 07).

## 9. Stage 7 — Evaluate retrieval and grounding

```powershell
python 07_evaluate_rag.py --top-k 3
```

The script reuses the locally cached research collection and three questions in `evaluation_questions.json`. It writes `cache/rag_evaluation_results.json`, recording retrieved chunk IDs, cosine scores, and a deliberately simple **keyword coverage heuristic**.

**Important:** Keyword coverage is *not* retrieval Recall@k. Real Recall@k requires a human-annotated relevant set for each query:

\[\mathrm{Recall@}k=\frac{|\mathrm{Relevant}(q)\cap\mathrm{TopK}(q)|}{|\mathrm{Relevant}(q)|}.\]

Similarly, Precision@k counts relevant passages among the retrieved top-k. Mean reciprocal rank (MRR) rewards retrieving a relevant passage near the top. We do not pretend to calculate these metrics without relevance labels.

To generate final answers to review manually:

```powershell
python 07_evaluate_rag.py --top-k 3 --generate
```

Manually check: (a) did the retrieved abstracts support the factual claims, (b) were URLs and `[S#]` IDs used correctly, (c) did the answer acknowledge limits, and (d) did it refuse to fabricate results for the fictional drug question? An **abstention question** is included because high semantic similarity does not prove the answer exists in the corpus.

## 10. Suggested experiments

**Experiment A — Retrieval versus generation error:** Find a question where the correct information is present in a retrieved passage but missing or distorted in the answer. Conversely, find a question for which none of the passages provide enough information. Explain why these are different failure modes.

**Experiment B — Vary top-k:** Compare `k=2`, `k=4` and `k=8` on the same question. Record retrieved titles, evidence quality, length and citation accuracy. Do not assume larger k is always better.

**Experiment C — Compare approaches:** Ask the same question using `04_basic_rag.py`, `05_openai_web_search.py` and `06_agentic_rag.py`. Compare sources, recency, interpretive limits and cost. The methods have different search spaces, so this is a qualitative comparison, not a controlled benchmark.

**Experiment D — Update the research query:** Use the Europe PMC retrieval script with a different topic (e.g., `Bayesian biomarkers`). Re-run semantic retrieval. The first embedding call for a new corpus costs more than later cached runs.

## 11. Common troubleshooting

- `ModuleNotFoundError: agents`: install `openai-agents` **in the Python interpreter you are running**, not a different environment.
- `OPENAI_API_KEY` missing: copy `.env` to `.env`, set your key, verify PyCharm uses the correct environment and working directory.
- `FileNotFoundError: Run 01...`: run `python 01_retrieve_research_text.py` first.
- Europe PMC network failure: check connectivity, proxy settings and the Europe PMC service; this first stage requires internet.
- Few or no abstracts: simplify `--query` or raise `--limit`; a result might have no abstract.
- 400 or unavailable model: check API model access and update `OPENAI_MODEL`; hosted search support depends on model/API availability.
- Unexpected answers: inspect the top-k passages before changing the generation prompt.
- Sensitive data: do not send protected patient text to third-party APIs or commit it into caches.

## 12. Lessons learned

You have now seen **two retrieval implementations**: a transparent embedding-based pipeline you control and OpenAI's managed web search. You have also seen **two orchestration styles**: a fixed RAG pipeline and an agent that decides when to search. Neither automatically guarantees source quality or factual correctness.

**Next direction:** combine this research retrieval capability with the statistical computation tools from Lesson 05, while preserving tool input validation and the evaluation/monitoring practices of Lessons 06–07.

## References

- [Europe PMC REST services](https://europepmc.org/RestfulWebService)
- [OpenAI embeddings](https://developers.openai.com/api/docs/guides/embeddings)
- [OpenAI web search](https://developers.openai.com/api/docs/guides/tools-web-search)
- [OpenAI Agents SDK Python](https://openai.github.io/openai-agents-python/)

---

## 10. Stage 8 — Persistent vector database with Qdrant Cloud

We keep the original exact NumPy cosine-similarity search to teach retrieval mathematics. In this stage we use **Qdrant Cloud** to store and query the **same precomputed OpenAI embeddings**. This avoids installing another Python interpreter, Docker, or a native Windows ChromaDB HNSW backend.

### Create your Qdrant Cloud cluster

1. Go to [Qdrant Cloud](https://cloud.qdrant.io/), register, and create a **Free Cluster** if eligible.
2. Copy the **cluster endpoint** (a URL beginning `https://`) and the **API key**. Keep the key private.
3. Install dependencies in your existing `.venv`:

```powershell
python -m pip install -r requirements.txt
```

4. If you already have `.env`, **add** the following lines; don't overwrite your existing OpenAI key:

```dotenv
QDRANT_URL=https://YOUR-CLUSTER-ENDPOINT
QDRANT_API_KEY=YOUR_SECRET_API_KEY
```

Never commit `.env` or paste API keys into GitHub. Data (the public abstracts, vectors and metadata) is transmitted to Qdrant Cloud; don't use private biomedical data without appropriate data-governance approval.

### Run the vector database stage

If you've completed Stage 3 and the existing `cache/` still holds chunks and vectors, you can reuse them. Otherwise:

```powershell
python 01_retrieve_research_text.py
python 03_semantic_retrieval.py
```

Then run:

```powershell
python 08_qdrant_retrieval.py
```

The script loads cached `chunks` and 1536-dimensional (for `text-embedding-3-small`) vectors, connects to Qdrant, creates a cosine-distance collection if necessary, and **upserts** vectors with document IDs, source IDs, titles, URLs and text in the payload. It embeds the **question** once via OpenAI, then prints matching passages with similarity scores. The corpus embeddings are not regenerated when cached.

Try:

```powershell
python 08_qdrant_retrieval.py --question "How do Bayesian trials use interim analyses?" --top-k 5
```

For metadata filtering, copy an actual `source_id` from the output and use `--source-id "ACTUAL_ID"`. Re-running with the same corpus updates points with stable IDs rather than duplicating them. A change to corpus or embedding model creates a different collection name, leaving old collections until you delete them from Qdrant Cloud.

**Local vs cloud:** NumPy vectors are stored in `cache/` on your computer, while the Qdrant collection is stored on Qdrant Cloud and accessed over HTTPS. This is why Qdrant's network latency will affect timing comparisons. Qdrant computes cosine similarity from uploaded vectors; the OpenAI embedding API remains responsible for creating those vectors.

## 11. Stage 9 — Compare NumPy and Qdrant Cloud retrieval

```powershell
python 09_compare_retrieval.py --top-k 5
```

This script sends **the same question vector** to both retrieval backends and reports top-k ranking overlap and per-query times. Qdrant's query time includes network transmission; NumPy runs in process. Results should usually be similar for the same nonzero vectors and cosine metric, but ranking differences may arise from ties, precision or approximate indexing. One timing observation is **not** a rigorous benchmark.

| Concern | NumPy | Qdrant Cloud |
|---|---|---|
| Vectors | Local `ndarray` | Cloud collection |
| Similarity | Exact cosine | Indexed cosine search |
| Persistence | Manual local files | Cloud-managed |
| Metadata filtering | Implement manually | Built-in payload filtering |
| Update records | Manual code | Upsert by point ID |
| Access | Local computation | Authenticated HTTPS |
| Extra costs/privacy | Local computation | Cloud plan and external data transfer |

### Offline vector database test

```powershell
python test_vectordb.py
```

This test uses **QdrantClient(':memory:')**, so you don't need a Qdrant Cloud connection or OpenAI key. It checks indexing, idempotent upserts, ranked retrieval and metadata filtering. It still requires the `qdrant-client` package to be installed. The **live cloud connection** has not been tested here; verify it with your own credentials using Stage 8.

### Troubleshooting

- `Missing QDRANT_URL / QDRANT_API_KEY`: set both values in `.env` within `08_rag/`.
- Authentication failed (401/403): copy the API key supplied for your cluster, and verify its permissions.
- Connection refused / DNS error: check your cluster endpoint, that the cloud cluster is running, and network access.
- Collection dimensions differ: this lesson assumes the same cached embedding model for all vectors within one collection. Avoid mixing vectors from different models.
- If the output shows `No passages matched`, check your `--source-id` value and whether your collection has been indexed.
- If you've already run the earlier Chroma version, **you do not need to delete** your existing local embeddings or reinstall Python. The replacement version no longer imports ChromaDB.

## 12. Consolidated commands

```powershell
python offline_tests.py              # Existing NumPy/text tests; offline
python test_vectordb.py              # Qdrant in-memory; offline
python 01_retrieve_research_text.py  # Europe PMC
python 02_embeddings.py              # OpenAI embeddings
python 03_semantic_retrieval.py      # NumPy semantic retrieval
python 04_basic_rag.py               # Retrieval + generation
python 05_openai_web_search.py       # Managed web search
python 06_agentic_rag.py             # Agentic retrieval
python 07_evaluate_rag.py           # Retrieval evaluation
python 08_qdrant_retrieval.py        # Qdrant Cloud storage and query
python 09_compare_retrieval.py       # NumPy vs Qdrant Cloud
```

**Suggested exercise:** Keep the corpus, model and question fixed and compare the top-5 retrievals. Independently label genuinely relevant chunks before claiming one backend has better retrieval accuracy.

### Additional references

- [Qdrant Cloud Quickstart](https://qdrant.tech/documentation/cloud-quickstart/)
- [Qdrant Python client](https://github.com/qdrant/qdrant-client)
