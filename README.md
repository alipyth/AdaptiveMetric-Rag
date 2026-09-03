# AdaptiveMetric RAG

**Not every query should use the same notion of similarity.**

AdaptiveMetric RAG is a self-hosted, multilingual knowledge assistant that changes its retrieval strategy for every question. A temporal question emphasizes dates, a factual question emphasizes entities and exact terms, and a conceptual question emphasizes semantic similarity. Every answer includes inspectable source citations.

[راهنمای فارسی](README.fa.md)

## What is included

- Adaptive query analyzer with factual, conceptual, causal, numeric, temporal, and technical/code intents
- Per-query mixture of dense, BM25, entity, numeric, temporal, and metadata scores
- Multilingual multi-keyword expansion with blended query vectors for cross-language retrieval
- Three-stage flow: fast candidate selection → adaptive scoring → grounded generation
- Confidence scoring and early-exit signals
- PDF, DOCX, TXT, Markdown, CSV, JSON, and HTML ingestion
- Inline citations with source excerpts, page numbers, chunk IDs, and retrieval scores
- Local no-key extractive mode, Ollama, OpenAI-compatible/dedicated endpoints, and Google Gemini
- Selectable local or Ollama embedding models with automatic full re-indexing when the model changes
- Persistent conversations, documents, and settings in SQLite
- Responsive Persian-first UI with a knowledge library and retrieval diagnostics
- Rich Markdown answer rendering plus persistent dark and light themes
- Per-conversation JSON export with messages, citations, and safe runtime metadata
- Single-command Docker deployment on port `2266`

## Architecture

```text
Query → Query Analyzer → Metric Router
                            │
          Dense + BM25 + Entity + Number + Time + Metadata
                            │
                    Candidate pool (10–500)
                            │
                  Adaptive metric scoring
                            │
              Confidence / early-exit decision
                            │
                 Top grounded context chunks
                            │
             Local / Ollama / OpenAI / Gemini
                            │
                    Answer + citations
```

The built-in embedding uses deterministic 384-dimensional multilingual feature hashing. It starts instantly, runs offline, and is appropriate for small and medium personal knowledge bases. The retrieval layer is isolated in `app/retrieval.py`, making it straightforward to replace candidate selection with Qdrant or FAISS for million-chunk deployments while retaining the adaptive scoring layer.

For higher multilingual accuracy, select `bge-m3` from Ollama. Document vectors include the filename, section, and chunk content; query vectors blend the original question with compact Persian/English keyword-expanded variants.

## Quick start with Docker

```bash
git clone https://github.com/alipyth/AdaptiveMetric-Rag
cd AdaptiveMetric-RAG
docker compose up --build -d
```

Open **http://localhost:2266**. The default local provider requires no API key.

To stop the service:

```bash
docker compose down
```

Data is stored in the named Docker volume `adaptive_rag_data` and survives container recreation.

## Provider setup

Open **Settings → Model provider** in the UI.

Generation and embedding are configured independently. Under **Settings → Retrieval**, choose the built-in local 384-dimensional feature hashing or select an embedding model already installed in Ollama. The UI can load installed Ollama models and automatically re-embeds all existing chunks before activating a changed model.

### Ollama

1. Run Ollama on the host and pull a model, for example `ollama pull llama3.2`.
2. Select **Ollama**.
3. Use model `llama3.2` and URL `http://host.docker.internal:11434`.

### OpenAI or a dedicated OpenAI-compatible endpoint

Select **OpenAI / Dedicated compatible**, enter the model, API key, and base URL. The default is `https://api.openai.com/v1`; a vLLM, LM Studio, corporate gateway, or dedicated endpoint can be used if it implements `POST /chat/completions`.

You can alternatively provide `OPENAI_API_KEY` in `.env`.

### Google Gemini

Select **Google Gemini**, enter a Gemini model such as `gemini-2.5-flash`, and add the API key. You can alternatively provide `GEMINI_API_KEY` in `.env`.

API keys submitted through the interface are stored server-side and never returned to the browser. For public or multi-user production deployments, inject secrets through environment variables or a secrets manager and place the application behind authentication and TLS.

## Retrieval settings

| Setting | Default | Purpose |
|---|---:|---|
| Candidate pool | 100 | Number of fused dense/BM25 candidates evaluated by the adaptive metric |
| Context chunks | 5 | Sources passed to the answer provider |
| Chunk size | 900 | Approximate characters per chunk |
| Chunk overlap | 140 | Character overlap between adjacent chunks |
| Confidence threshold | 0.58 | Threshold exposed for confidence-aware flows |
| Early exit | On | Marks decisive retrievals so expensive optional stages can be skipped |
| Query expansion | On | Enables expansion behavior in confidence-aware extensions |

Chunk settings apply to newly uploaded documents. Re-upload existing documents after changing them.

## Local development

Requires Python 3.11+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 2266
```

Run tests:

```bash
pytest -q
```

API documentation is available at `http://localhost:2266/docs`.

## Citation behavior

Retrieved chunks are numbered before generation. The model is instructed to cite factual claims as `[1]`, `[2]`, etc. The API also returns a structured `citations` array independently of model formatting, so clients can always display the source document, page, excerpt, chunk ID, and adaptive score.

## API example

```bash
curl -X POST http://localhost:2266/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"When does contract 137 expire?"}'
```

## Production notes

- Put the service behind an authenticated reverse proxy before exposing it publicly.
- SQLite is deliberately simple for a single-node deployment. Use PostgreSQL for multi-user writes.
- For very large collections, use Qdrant/FAISS for first-stage ANN retrieval and retain adaptive scoring over the top 50–200 candidates.
- Add OCR before ingestion for scanned PDFs; `pypdf` extracts text but does not perform OCR.
- Benchmark retrieval on your own labeled query/chunk pairs using Recall@K, MRR, nDCG, and p95 latency.

## License

Add the license appropriate for your intended distribution before publishing the repository.
