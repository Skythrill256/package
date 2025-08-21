"""
Query functionality using LlamaIndex integration with OpenAI models.
"""

import asyncio
import logging
from typing import Any, Optional

try:
    from llama_index.core import Settings, VectorStoreIndex
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.schema import NodeWithScore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from qdrant_client import AsyncQdrantClient
except ImportError:
    raise ImportError(
        "llama-index packages are required. Install with: "
        "pip install llama-index llama-index-vector-stores-qdrant "
        "llama-index-embeddings-openai llama-index-llms-openai"
    ) from None

from ..config.config import WebVectorConfig
from ..embeddings.embeddings import EmbeddingManager
from ..errors.exceptions import QueryError
from ..storage.storage import VectorStorage

logger = logging.getLogger(__name__)


class QueryEngine:
    """Advanced query engine using LlamaIndex with OpenAI integration."""
    
    def __init__(self, config: WebVectorConfig):
        """Initialize the query engine."""
        self.config = config
        self.embedding_manager = EmbeddingManager(config)
        self.vector_storage = VectorStorage(config)
        
        # Initialize LlamaIndex components
        self._setup_llamaindex()
        
        # Query engine components
        self.vector_store = None
        self.index = None
        self.query_engine = None
        
    def _setup_llamaindex(self):
        """Setup LlamaIndex global settings."""
        # Configure OpenAI LLM
        Settings.llm = OpenAI(
            api_key=self.config.openai_api_key,
            model=self.config.openai_model,
            temperature=0.1,
            max_tokens=1000
        )
        
        # Configure OpenAI embeddings
        Settings.embed_model = OpenAIEmbedding(
            api_key=self.config.openai_api_key,
            model=self.config.embedding_model,
            dimensions=self.config.embedding_dimensions
        )
        
        # Set chunk size for text processing
        Settings.chunk_size = self.config.chunk_size
        Settings.chunk_overlap = self.config.chunk_overlap
    
    async def initialize(self):
        """Initialize the query engine with vector store connection."""
        try:
            # Create Qdrant client
            qdrant_client = AsyncQdrantClient(url=self.config.qdrant_url)
            
            # Create vector store
            self.vector_store = QdrantVectorStore(
                aclient=qdrant_client,
                collection_name=self.config.collection_name,
                enable_hybrid=False  # Use pure vector search
            )
            
            # Create index from existing vector store
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store
            )
            
            # Create query engine
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=10,
                vector_store_query_mode="default"
            )
            
            response_synthesizer = get_response_synthesizer(
                response_mode="tree_summarize",
                use_async=True
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )
            
            logger.info("Query engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing query engine: {str(e)}")
            raise QueryError(f"Failed to initialize query engine: {str(e)}") from e
    
    async def query(
        self, 
        question: str, 
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_sources: bool = True,
        filters: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        """
        Query the vector database using natural language.
        
        Args:
            question: Natural language question
            top_k: Number of top results to retrieve
            similarity_threshold: Minimum similarity score
            include_sources: Whether to include source information
            filters: Optional filters for search
            
        Returns:
            Query response with answer and sources
            
        Raises:
            QueryError: If query execution fails
        """
        if not self.query_engine:
            await self.initialize()
        
        try:
            logger.info(f"Processing query: {question}")
            
            # Execute query using LlamaIndex
            response = await self.query_engine.aquery(question)
            
            # Format response
            result = {
                'question': question,
                'answer': str(response.response),
                'confidence_score': getattr(response, 'confidence', 0.0),
                'sources': []
            }
            
            # Extract source information if requested
            if include_sources and hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    if isinstance(node, NodeWithScore):
                        source_info = {
                            'text': node.node.text,
                            'score': node.score,
                            'metadata': node.node.metadata,
                            'node_id': node.node.node_id,
                            'url': node.node.metadata.get('url', ''),
                            'title': node.node.metadata.get('title', ''),
                            'chunk_id': node.node.metadata.get('chunk_id', '')
                        }
                        result['sources'].append(source_info)
            
            logger.info(f"Query completed with {len(result['sources'])} sources")
            return result
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise QueryError(f"Failed to execute query: {str(e)}") from e
    
    async def similarity_search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[dict[str, Any]] = None
    ) -> list[dict[str, Any]]:
        """
        Perform similarity search without LLM processing.
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            filters: Optional search filters
            
        Returns:
            List of similar documents with scores
            
        Raises:
            QueryError: If search fails
        """
        try:
            # Generate query embedding
            query_embedding = await self.embedding_manager.generate_query_embedding(query)
            
            # Search using vector storage
            async with self.vector_storage as storage:
                results = await storage.search_similar(
                    query_embedding=query_embedding,
                    limit=top_k,
                    score_threshold=similarity_threshold,
                    filters=filters
                )
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise QueryError(f"Failed to perform similarity search: {str(e)}") from e
    
    async def get_related_content(
        self,
        url: str,
        top_k: int = 5,
        similarity_threshold: float = 0.6
    ) -> list[dict[str, Any]]:
        """
        Get content related to a specific URL.
        
        Args:
            url: URL to find related content for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of related content
        """
        try:
            # Search for content from the same URL
            filters = {'url': url}
            
            async with self.vector_storage as storage:
                url_content = await storage.search_similar(
                    query_embedding=[0.0] * self.config.embedding_dimensions,  # Dummy vector
                    limit=1,
                    score_threshold=0.0,
                    filters=filters
                )
            
            if not url_content:
                return []
            
            # Use the first chunk as reference for finding similar content
            reference_text = url_content[0]['text']
            
            # Generate embedding for reference text
            reference_embedding = await self.embedding_manager.generate_query_embedding(reference_text)
            
            # Find similar content excluding the same URL
            async with self.vector_storage as storage:
                similar_results = await storage.search_similar(
                    query_embedding=reference_embedding,
                    limit=top_k + 10,  # Get extra to filter out same URL
                    score_threshold=similarity_threshold
                )
            
            # Filter out content from the same URL
            related_content = [
                result for result in similar_results 
                if result['url'] != url
            ][:top_k]
            
            return related_content
            
        except Exception as e:
            logger.error(f"Error getting related content: {str(e)}")
            raise QueryError(f"Failed to get related content: {str(e)}") from e
    
    async def get_collection_stats(self) -> dict[str, Any]:
        """
        Get statistics about the vector collection.
        
        Returns:
            Collection statistics
        """
        try:
            async with self.vector_storage as storage:
                info = await storage.get_collection_info()
                
                # Get additional stats
                total_points = await storage.count_points()
                
                stats = {
                    'collection_name': self.config.collection_name,
                    'total_documents': total_points,
                    'vector_dimensions': info.get('vector_size', self.config.embedding_dimensions),
                    'distance_metric': info.get('distance_metric', 'cosine'),
                    'indexed_vectors': info.get('indexed_vectors_count', 0),
                    'status': info.get('status', 'unknown')
                }
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise QueryError(f"Failed to get collection stats: {str(e)}") from e
    
    async def batch_query(
        self,
        questions: list[str],
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> list[dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            questions: List of questions to process
            top_k: Number of results per query
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of query results
        """
        results = []
        
        for question in questions:
            try:
                result = await self.query(
                    question=question,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
                results.append(result)
                
                # Small delay between queries to avoid rate limits
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                results.append({
                    'question': question,
                    'answer': f"Error: {str(e)}",
                    'confidence_score': 0.0,
                    'sources': []
                })
        
        return results
