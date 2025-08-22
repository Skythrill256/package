# WebVector – Web Scraping to Vector DB with LLM Querying

Turn websites into a searchable knowledge base: scrape pages, generate embeddings, store them in a vector database (Qdrant), and query using an LLM. This package orchestrates web scraping, embedding generation, vector storage, and natural language querying via a single high-level client.

> Python 3.9–3.13 supported


## Features

* **End‑to‑end pipeline** — scrape URLs, chunk/process, embed, and store in Qdrant.
* **LLM‑powered queries** — retrieve relevant chunks and synthesize answers.
* **Typed config** — `WebVectorConfig` validates all key parameters.
* **Async by design** — efficient scraping and I/O with asyncio.
* **Composable** — use the high‑level `WebVectorClient` or individual subsystems.


## Architecture Overview

The `WebVectorClient` coordinates four components:

1. **Scraper** (`package.crawling.scraping.WebScraper`) — fetch and extract page content.
2. **Embeddings** (`package.embeddings.embeddings.EmbeddingManager`) — generate vector embeddings (OpenAI by default).
3. **Vector Store** (`package.storage.storage.VectorStorage`) — persist vectors to Qdrant.
4. **Query Engine** (`package.queries.query.QueryEngine`) — retrieve relevant chunks and answer queries.

Key public APIs live in:

* `package/core/core.py` — `WebVectorClient`
* `package/config/config.py` — `WebVectorConfig`


## Installation

This project uses `pyproject.toml` (PEP 621) and `setuptools`.

```bash
uv sync
```

Runtime dependencies (from `pyproject.toml`):

* `openai`
* `qdrant-client`
* `llama-index`, `llama-index-vector-stores-qdrant`, `llama-index-embeddings-openai`, `llama-index-llms-openai`
* `numpy`, `aiohttp`, `asyncio-throttle`, `python-dotenv`, `pydantic`, `crawl4ai`


## Prerequisites

* Python >= 3.9
* An OpenAI API key
* A running Qdrant instance (local Docker example below)


## Quickstart (Python)

```python
import asyncio
from package.config.config import WebVectorConfig
from package.core.core import WebVectorClient

async def main():
    config = WebVectorConfig(
        openai_api_key="<YOUR_OPENAI_API_KEY>",
        qdrant_url="http://localhost:6333",
        collection_name="my_collection",
        max_pages=10,
        max_depth=1,
    )

    client = WebVectorClient(config)

    # 1) Scrape + embed + store
    summary = await client.scrape_and_store([
        "https://example.com",
    ], recreate_collection=False)
    print(summary)

    # 2) Ask questions
    answer = await client.query(
        question="What does the site say about X?" )
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
```


## Quickstart (CLI example)

An interactive example is provided in `examples/cli.py`.

```bash
export OPENAI_API_KEY=sk-...  # or enter when prompted
uv run examples/cli.py
```

What it does:

1. Prompts for URLs, scrapes them, generates embeddings, and stores in Qdrant (collection is hardcoded to `new`).
2. Starts a query loop so you can ask questions and see sources.


## Configuration

All configuration is centralized in `WebVectorConfig` (`package/config/config.py`). Required fields:

* `openai_api_key: str` — OpenAI API key.
* `qdrant_url: str` — Qdrant HTTP endpoint (e.g., `http://localhost:6333`).
* `collection_name: str` — Name of the Qdrant collection.

Notable optional fields (with defaults):

* `openai_model` (default: `gpt-4`) — LLM used for answer synthesis.
* `embedding_model` (default: `text-embedding-3-large`) — OpenAI embedding model.
* `embedding_dimensions` (default: `3072`) — must match the embedding model you use.
* Scraping: `max_pages` (50), `max_depth` (2), `include_external` (False), `score_threshold` (0.3)
* Processing: `chunk_size` (1000), `chunk_overlap` (200), `batch_size` (50)
* Filters: `keywords`, `url_patterns`, `exclude_patterns`

Validation rules are enforced in `WebVectorConfig.validate()`; misconfigurations raise `ConfigurationError`.


## Running Qdrant locally

```bash
docker run -p 6333:6333 -p 6334:6334 -v qdrant_storage:/qdrant/storage qdrant/qdrant:latest
```

Then set `qdrant_url` to `http://localhost:6333`.


## Troubleshooting

* **No content scraped** — `scrape_and_store()` raises `WebVectorError` if pages were empty or unreachable.
* **No embeddings generated** — confirm OpenAI API key and embedding model permissions.
* **Dimension mismatch** — ensure `embedding_dimensions` matches the embedding model.
* **Qdrant connection issues** — verify the container is running and the URL is reachable.
* **Silent logs** — the client avoids configuring handlers; configure logging in your app if you need visibility.


## Development

```bash
pytest -q
```

Formatting and typing:

```bash
black .
flake8
mypy
```





