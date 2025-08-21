"""
Embedding generation using LlamaIndex's OpenAIEmbedding.
"""

import asyncio
import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

try:
    # LlamaIndex OpenAI embeddings integration
    from llama_index.embeddings.openai import OpenAIEmbedding
except ImportError:
    raise ImportError(
        "llama-index-embeddings-openai is required. Install with: pip install llama-index-embeddings-openai"
    ) from None

from ..config.config import WebVectorConfig
from ..errors.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# LlamaIndex core and Qdrant vector store
try:
    from llama_index.core import (
        Document,
        Settings,
        StorageContext,
        VectorStoreIndex,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
except ImportError:
    # Keep lazy import errors until features are used
    Document = None  # type: ignore
    StorageContext = None  # type: ignore
    VectorStoreIndex = None  # type: ignore
    Settings = None  # type: ignore
    SentenceSplitter = None  # type: ignore
    QdrantVectorStore = None  # type: ignore
    QdrantClient = None  # type: ignore


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: list[float]
    token_count: int
    chunk_id: str
    metadata: dict[str, Any]


class EmbeddingManager:
    """Manages embedding generation using LlamaIndex OpenAIEmbedding."""
    
    def __init__(self, config: WebVectorConfig):
        """Initialize the embedding manager."""
        self.config = config

        # Ensure API key is available for LlamaIndex/OpenAI
        if self.config.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = self.config.openai_api_key

        # Initialize LlamaIndex embedding model
        try:
            self.embed_model = OpenAIEmbedding(
                model=self.config.embedding_model,
                dimensions=self.config.embedding_dimensions,
                embed_batch_size=self.config.batch_size,
            )
        except TypeError:
            # Fallback for older versions without dimensions/embed_batch_size kwargs
            self.embed_model = OpenAIEmbedding(model=self.config.embedding_model)
        
        # Validate embedding model and dimensions
        self._validate_embedding_config()

        # Placeholders for LlamaIndex components
        self._splitter = None
        self._vector_store = None
        self._index = None
        self._query_engine = None

    def _ensure_llamaindex_ready(self):
        """Initialize LlamaIndex ServiceContext and VectorStore if available."""
        if Document is None:
            raise ImportError(
                "llama-index core/vector-store packages are required. Install with: pip install llama-index llama-index-vector-stores-qdrant"
            )

        # Configure global Settings for LlamaIndex (embed model and parser)
        if Settings is not None:
            Settings.embed_model = self.embed_model
            if self._splitter is None:
                self._splitter = SentenceSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                )
            Settings.node_parser = self._splitter

        if self._vector_store is None:
            client = QdrantClient(url=self.config.qdrant_url)
            self._vector_store = QdrantVectorStore(
                client=client,
                collection_name=self.config.collection_name,
            )

    async def ingest_with_llamaindex(self, scraped_data: list[dict[str, Any]], recreate: bool = False) -> dict[str, Any]:
        """
        End-to-end ingestion using LlamaIndex: split -> embed -> store in Qdrant.

        Args:
            scraped_data: List of page dicts with 'content' and 'metadata'
            recreate: If True, drop and recreate the collection

        Returns:
            Summary dict with counts and collection info
        """
        try:
            self._ensure_llamaindex_ready()

            # Optionally recreate collection
            if recreate and self._vector_store is not None:
                # QdrantVectorStore exposes client via .client
                with suppress(Exception):
                    self._vector_store.client.delete_collection(self.config.collection_name)  # type: ignore[attr-defined]

            # Build Documents
            documents: list[Document] = []  # type: ignore[assignment]
            for page in scraped_data:
                content = page.get("content", "")
                metadata = page.get("metadata", {}) or {}
                if not content.strip():
                    continue
                documents.append(Document(text=content, metadata=metadata))  # type: ignore[call-arg]

            if not documents:
                logger.warning("No documents to index")
                return {"pages": 0, "nodes": 0, "collection": self.config.collection_name}

            storage_context = StorageContext.from_defaults(vector_store=self._vector_store)  # type: ignore[arg-type]

            # Indexing is synchronous; run in a thread to avoid blocking
            def _build_index():
                return VectorStoreIndex.from_documents(  # type: ignore[call-arg]
                    documents,
                    storage_context=storage_context,
                    show_progress=False,
                )

            self._index = await asyncio.to_thread(_build_index)
            self._query_engine = self._index.as_query_engine(similarity_top_k=self.config.batch_size)  # reuse batch_size as top_k default

            # Rough counts: nodes equal to total chunks created by splitter
            # LlamaIndex doesn't directly expose node count here; skip precise count.
            return {
                "pages": len(documents),
                "nodes": None,
                "collection": self.config.collection_name,
            }
        except Exception as e:
            logger.error(f"LlamaIndex ingestion failed: {e}")
            raise EmbeddingError(f"LlamaIndex ingestion failed: {e}") from e

    async def query_llamaindex(self, question: str, top_k: int = 5) -> dict[str, Any]:
        """Query using LlamaIndex's QueryEngine built on the vector store."""
        try:
            self._ensure_llamaindex_ready()
            if self._index is None:
                raise EmbeddingError("Index not initialized. Call ingest_with_llamaindex() first.")

            # Re-create query engine if top_k changes
            if self._query_engine is None or top_k != getattr(self._query_engine, "similarity_top_k", top_k):
                self._query_engine = self._index.as_query_engine(similarity_top_k=top_k)

            # Execute query in a thread (sync under the hood)
            def _run_query():
                return self._query_engine.query(question)

            response = await asyncio.to_thread(_run_query)

            # Collect sources
            sources = []
            try:
                for sn in getattr(response, "source_nodes", []) or []:
                    meta = getattr(sn.node, "metadata", {})
                    sources.append({
                        "text": sn.node.get_text() if hasattr(sn.node, "get_text") else None,
                        "score": getattr(sn, "score", None),
                        "metadata": meta,
                    })
            except Exception:
                pass

            return {
                "answer": str(response),
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"LlamaIndex query failed: {e}")
            raise EmbeddingError(f"LlamaIndex query failed: {e}") from e
    
    def _validate_embedding_config(self):
        """Validate embedding model and dimensions compatibility."""
        model_dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        
        if self.config.embedding_model in model_dimensions:
            expected_dim = model_dimensions[self.config.embedding_model]
            if self.config.embedding_dimensions != expected_dim:
                logger.warning(
                    f"Dimension mismatch: {self.config.embedding_model} "
                    f"typically uses {expected_dim} dimensions, "
                    f"but config specifies {self.config.embedding_dimensions}"
                )
    
    async def generate_embeddings(self, scraped_data: list[dict[str, Any]]) -> list[EmbeddingResult]:
        """
        Generate embeddings for scraped content using LlamaIndex's splitter and embedder.
        
        Args:
            scraped_data: List of scraped page data
            
        Returns:
            List of embedding results
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Ensure LlamaIndex components
            self._ensure_llamaindex_ready()

            # Build Documents
            documents: list[Document] = []  # type: ignore[assignment]
            for page_data in scraped_data:
                content = page_data.get('content', '')
                metadata = page_data.get('metadata', {}) or {}
                if not content.strip():
                    logger.warning(f"No content found for {metadata.get('url', 'unknown URL')}")
                    continue
                documents.append(Document(text=content, metadata=metadata))  # type: ignore[call-arg]

            if not documents:
                logger.warning("No chunks to embed")
                return []

            # Split into nodes using LlamaIndex's splitter
            splitter = self._splitter or SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

            def _split_docs():
                return splitter.get_nodes_from_documents(documents)

            nodes = await asyncio.to_thread(_split_docs)

            logger.info(f"Generating embeddings for {len(nodes)} chunks via LlamaIndex")

            # Embed each node text via LlamaIndex embedding model
            def _embed_nodes(texts: list[str]) -> list[list[float]]:
                return [self.embed_model.get_text_embedding(t) for t in texts]

            texts = [n.get_text() if hasattr(n, 'get_text') else getattr(n, 'text', '') for n in nodes]
            vectors = await asyncio.to_thread(_embed_nodes, texts)

            # Assemble EmbeddingResult; compute per-document chunk indices
            embedding_results: list[EmbeddingResult] = []
            doc_counters: dict[str, int] = {}
            for node, vec in zip(nodes, vectors):
                # Derive a doc key from metadata URL if present, else a generic key
                meta = getattr(node, 'metadata', {}) or {}
                doc_key = meta.get('url') or meta.get('source') or 'doc'
                idx = doc_counters.get(doc_key, 0)
                doc_counters[doc_key] = idx + 1

                # Build metadata with chunk info
                merged_meta = dict(meta)
                merged_meta.update({
                    'chunk_index': idx,
                })

                chunk_id = getattr(node, 'node_id', None) or f"{doc_key}_{idx}"

                embedding_results.append(
                    EmbeddingResult(
                        text=texts[embedding_results.__len__()],
                        embedding=vec,
                        token_count=0,
                        chunk_id=str(chunk_id),
                        metadata=merged_meta,
                    )
                )

            logger.info(f"Successfully generated {len(embedding_results)} embeddings")
            return embedding_results
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise EmbeddingError(f"Failed to generate embeddings: {str(e)}") from e
    
    
    
    async def generate_query_embedding(self, query: str) -> list[float]:
        """
        Generate embedding for a query string using LlamaIndex.
        """
        try:
            # Run sync LlamaIndex call in a thread
            return await asyncio.to_thread(self.embed_model.get_text_embedding, query)
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate query embedding: {str(e)}") from e
    
    
