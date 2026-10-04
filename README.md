# RAG Evaluation

A retrieval-augmented generation system built on top of LLM evaluation course transcripts, with a full eval suite covering quality, safety, and operational concerns. The pipeline answers questions grounded strictly in the course material and uses snapshot-based regression testing to detect regressions across every dimension before any change is promoted.

---

## Architecture

```
VTT transcripts (data/)
       │
       ▼
  load_transcripts()          # strip timestamps, tag by session
       │
       ▼
  RecursiveCharacterTextSplitter  (chunk_size=1000, overlap=150)
       │
       ▼
  ChromaDB  ←─ text-embedding-3-large  (persisted to chroma_store/)
       │
       ▼  similarity_search (fetch_k candidates)
  cross-encoder/ms-marco-MiniLM-L-6-v2   (rerank ➜ top_k chunks)
       │
       ▼
  GPT-4o-mini  (faithfulness-first prompt, temperature=0)
       │
       ▼
  { query, context, answer }   ← returned by RagPipeline.invoke()
```

**Key design decisions**

- **Two-stage retrieval**: a fast bi-encoder over-retrieves (`fetch_k`) and a cross-encoder reranker cuts it down to `top_k`. This trades a small latency hit for significantly better precision.  
- **Faithfulness-first prompt**: the generator answers only from the retrieved context and abstains with a fixed string when the context is insufficient. No hallucination, no outside knowledge.  
- **LangSmith tracing**: `@traceable` decorators on `RagPipeline.invoke()` and `RerankingRetriever.invoke()` give full chain visibility in LangSmith without changing the call signatures.

---

## Project Layout

```
.
├── src/
│   ├── retriever.py          # transcript loading, ChromaDB setup, bi-encoder retrieval
│   ├── reranker.py           # cross-encoder reranking wrapper (RerankingRetriever)
│   ├── generator.py          # grounded generation + streaming (generate / generate_stream)
│   ├── rag_pipeline.py       # end-to-end pipeline (RagPipeline)
│   └── app.py                # Streamlit chat UI
│
├── evals/
│   ├── eval_retriever.py         # contextual recall / precision on retriever goldens
│   ├── eval_generator.py         # faithfulness on fixed golden contexts
│   ├── eval_rag_pipeline.py      # RAG triad (faithfulness + answer relevancy + contextual recall)
│   ├── eval_application.py       # end-to-end application quality
│   ├── eval_safety.py            # scope adherence · protected leakage · PII · toxicity
│   ├── eval_ops.py               # latency (p95/TTFT/stage) · cost · reliability
│   ├── metric_registry.py        # rules for every metric: kind, direction, tolerance
│   ├── run_suite.py              # full-suite orchestrator → writes a snapshot JSON
│   ├── compare.py                # diff two snapshots, returns PASS / REVIEW / FAIL
│   ├── harness.py                # shared eval helpers
│   └── compare.py
│
├── goldens/                  # golden datasets for each eval category
│   ├── retriever_goldens.json
│   ├── faithfulness_dataset.json
│   ├── correctness_goldens.json
│   ├── scope_goldens.json
│   ├── leakage_goldens.json
│   └── toxicity_goldens.json
│
├── data/                     # source VTT transcripts (8 sessions)
├── baselines/                # snapshot JSONs written by run_suite / compare
├── export_chroma_chunks.py   # utility: dump the embedded chunks to inspect them
└── pyproject.toml
```

---

## Eval Suite

All evals are run against a single `RagPipeline` instance that is built once and injected everywhere, so every metric in a snapshot provably describes the same pipeline.

### Quality

| Eval | What it measures | Golden set |
|---|---|---|
| **Retriever** | Contextual recall & precision — does the retriever surface the right chunks? | `retriever_goldens.json` |
| **Generator** | Faithfulness on fixed golden contexts — does the generator stay grounded? | `faithfulness_dataset.json` |
| **Pipeline triad** | Faithfulness + answer relevancy + contextual recall end-to-end | `retriever_goldens.json` |
| **Application** | Overall answer correctness against reference answers | `correctness_goldens.json` |

### Safety

| Eval | What it measures |
|---|---|
| **Scope** | Does the system stay in its teaching-assistant role and reject out-of-scope requests? |
| **Protected leakage** | Does the system protect system prompt, internal instructions, and raw course content? |
| **PII leakage** | Does the system avoid emitting personally identifiable information? |
| **Toxicity** | Does the system maintain a respectful, professional tone regardless of the input? |

Safety evals use LLM-as-judge (GPT-4o-mini via DeepEval) and are hard gates in regression — any drop in a safety pass rate blocks the build.

### Operational

| Eval | What it measures |
|---|---|
| **Latency** | End-to-end p95, time-to-first-token p95, retrieval vs generation breakdown |
| **Cost** | Token usage + USD cost per query; daily/monthly projection at configured traffic |
| **Reliability** | Success/error/retry rates with exponential-backoff retry wrapper |

Operational metrics are software measurements — no LLM judge needed. Latency uses warmup runs and percentiles (p95/p99) rather than the mean because tail latency is what users experience.

---

## Regression Testing

Every eval run produces a **snapshot JSON** that records all metric values plus run metadata (timestamp, git SHA, prompt hash). The comparison workflow:

```bash
# 1. Bless the current pipeline as the baseline
python -m evals.run_suite --baseline --label "initial baseline"

# 2. Make a change, capture a candidate snapshot
python -m evals.run_suite --label "fetch_k=15, reranker tuned"

# 3. Diff the two — get PASS / REVIEW / FAIL
python -m evals.compare
```

`compare.py` classifies every metric using the rules in `metric_registry.py`:

- **gate** — any regression beyond tolerance blocks (`FAIL`)  
- **guardrail** — regression beyond tolerance triggers human review (`REVIEW`)  
- **info** — recorded for observability, never affects the verdict

---

## Setup

**Prerequisites**: Python 3.11+, [uv](https://github.com/astral-sh/uv)

```bash
# Install dependencies
uv sync

# Create a .env file with your keys
echo "OPENAI_API_KEY=sk-..." >> .env
# Optional: LangSmith tracing
echo "LANGCHAIN_TRACING_V2=true" >> .env
echo "LANGCHAIN_API_KEY=ls__..." >> .env
```

The ChromaDB vector store (`chroma_store/`) is built automatically on the first run by embedding the transcripts in `data/`. Subsequent runs load the persisted store.

---

## Running

**Chat UI**
```bash
uv run streamlit run src/app.py
```
The sidebar exposes `fetch_k` (bi-encoder candidate count) and `top_k` (post-rerank chunk count) as live sliders without restarting the server.

**Individual evals**
```bash
python -m evals.eval_ops          # latency + cost + reliability
python -m evals.eval_safety       # scope + leakage + toxicity
python -m evals.eval_rag_pipeline # RAG triad
```

**Full suite + regression comparison**
```bash
python -m evals.run_suite --baseline              # save baseline
python -m evals.run_suite --label "my change"     # save candidate
python -m evals.compare                           # PASS / REVIEW / FAIL
python -m evals.compare --all                     # include info metrics
```

**Component smoke tests**
```bash
python -m src.rag_pipeline          # end-to-end single query
python src/retriever.py             # retriever only
python src/generator.py             # generator only
```

---

## Dependencies

| Package | Role |
|---|---|
| `langchain-openai`, `langchain-core` | LLM + prompt chain |
| `langchain-chroma`, `chromadb` | Vector store |
| `langchain-text-splitters` | Document chunking |
| `sentence-transformers` | Cross-encoder reranker |
| `deepeval` | LLM-as-judge eval metrics |
| `openai` | Direct API access for cost measurement |
| `streamlit` | Chat UI |
| `python-dotenv` | Environment variable loading |
| `pytest` | Test runner |
