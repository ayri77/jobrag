# JobRAG

Local RAG assistant for job search workflows.

The project indexes CVs, profiles, recommendation letters, and related
career documents, then allows:

-   asking grounded questions over the indexed documents
-   generating concise, role‑relevant summaries for a specific job
    description
-   returning answers with source citations

The system is designed as both a practical learning project and a useful
personal tool for job applications.

------------------------------------------------------------------------

## Features

-   PDF ingestion with page-level metadata
-   text chunking with overlap
-   multilingual embeddings using `BAAI/bge-m3`
-   FAISS vector index
-   local LLM generation through Ollama
-   two main modes:
    -   `ask` --- question answering over documents
    -   `jd` --- job-description-based summary generation
-   source-aware answers with citations
-   CLI interface for indexing and querying

------------------------------------------------------------------------

## Current Status

MVP is implemented and working locally.

Supported workflow:

1.  Put PDF documents into `data/raw/`
2.  Build FAISS index
3.  Ask questions over documents
4.  Generate tailored summaries for a job description

------------------------------------------------------------------------

## Architecture

Pipeline:

```text
PDF files
   ↓
Text extraction
   ↓
Chunking
   ↓
Embeddings (bge-m3)
   ↓
FAISS index
   ↓
Retrieval
   ↓
Prompt assembly with sources
   ↓
Local LLM via Ollama
```

RAG principle used here:

-   retrieval finds relevant chunks
-   LLM does not access the vector store directly
-   LLM receives only retrieved text as context
-   output must be grounded in retrieved sources

------------------------------------------------------------------------

## Project Structure

The project is organized around the main RAG pipeline components.

```text
jobrag/
├─ data/
│  ├─ raw/                 # input documents (PDF CVs, profiles, etc.)
│  └─ index/               # generated FAISS index + chunk metadata
│
├─ src/
│  └─ jobrag/
│     ├─ ingest.py        # document loading and page extraction
│     ├─ chunk.py         # text chunking logic
│     ├─ embed.py         # embedding model wrapper
│     ├─ index.py         # FAISS index construction
│     ├─ search.py        # vector retrieval from index
│     ├─ rag.py           # RAG pipeline (retrieve → prompt → generate)
│     ├─ llm.py           # Ollama LLM client
│     ├─ cli.py           # command line interface
│     ├─ settings.py      # central configuration
│     └─ store.py         # metadata storage helpers
│
├─ src/tests/             # simple smoke tests for pipeline components
│
├─ pyproject.toml         # project dependencies and packaging
├─ uv.lock                # locked dependency versions
└─ README.md
```

------------------------------------------------------------------------

## Tech Stack

-   Python
-   uv
-   PyPDF
-   sentence-transformers
-   FAISS
-   Ollama
-   Typer
-   Rich

------------------------------------------------------------------------

## Requirements

-   Python environment managed with `uv`
-   Ollama installed locally
-   at least one local LLM pulled in Ollama
-   PDF documents placed into `data/raw/`

Recommended Ollama model for current MVP:

`qwen3.5:4b`

This model showed a better latency/quality balance than `qwen3.5:9b`
during testing.

------------------------------------------------------------------------

## Installation

### Clone the repository

```bash
git clone https://github.com/ayri77/jobrag.git
cd jobrag
```

### Install dependencies

```bash
uv sync
```

If needed:

```bash
uv pip install typer rich
```

### Install Ollama model

```bash
ollama pull qwen3.5:4b
```

### Create .env

```
HF_TOKEN=your_huggingface_read_token
```

The Hugging Face token is optional but prevents anonymous hub warnings.

------------------------------------------------------------------------

## Configuration

Main configuration is located in:

src/jobrag/settings.py

Typical parameters include:

-   embedding model
-   Ollama model
-   chunk size
-   chunk overlap
-   retrieval top-k
-   prompt limits
-   logging flags

------------------------------------------------------------------------

## Usage

### Add documents

Place PDF files into:

data/raw/

### Build index

```
uv run jobrag index
```

Output:

data/index/faiss.index\
data/index/chunks_meta.jsonl

### Ask a question

```
uv run jobrag ask -q "FastAPI, Docker, deploying ML services"
```

### Ask with debug

```
uv run jobrag ask -q "FastAPI, Docker, deploying ML services" --debug
```

### Generate summary for a job description
```
uv run jobrag jd --jd "We are looking for a Python/AI Engineer with
FastAPI, RAG, Docker, and LLM experience."
```
or
```
uv run jobrag jd --jd-file data/jd_sample.txt
```

------------------------------------------------------------------------

## Example Use Cases

Ask mode:

-   What experience do I have with FastAPI?
-   Which projects are related to LLM systems?
-   Do my documents mention Docker deployment?
-   What cloud platforms are referenced in my CV?

JD mode:

-   create a short tailored candidate summary
-   identify relevant projects for a specific role
-   prepare role-focused bullet points for applications

------------------------------------------------------------------------

## Output Grounding

Answers are grounded in retrieved document chunks.

Example citation:

\[CV_Python_BE_Borysov_Pavlo_02_2026_EN.pdf p1\]

------------------------------------------------------------------------

## Known Limitations

-   PDF extraction may introduce encoding artifacts
-   chunking is currently character-based
-   generation latency depends on the local LLM
-   citation validation is still lightweight
-   embeddings currently run on CPU in the Windows / Python 3.14 setup

------------------------------------------------------------------------

## Performance Notes

-   retrieval is fast (FAISS)
-   generation via local LLM is the main latency source
-   `qwen3.5:4b` significantly reduces response time vs `qwen3.5:9b`

------------------------------------------------------------------------

## Roadmap

Planned improvements:

-   improved PDF cleaning
-   sentence-aware chunking
-   reranking stage
-   stricter citation validation
-   streaming responses
-   FastAPI or web interface
-   support for additional document formats

------------------------------------------------------------------------

## Development Notes

This project was built as:

-   a practical RAG learning project
-   a personal job-search assistant
-   a foundation for experimenting with local LLM workflows

The architecture is modular:

-   indexing separate from CLI
-   retrieval separate from generation
-   configuration centralized in `settings.py`

------------------------------------------------------------------------

## License

MIT License
