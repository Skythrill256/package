"""
Vector storage functionality using Qdrant database.
"""

import logging
from typing import Any, Optional
from uuid import uuid4

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except ImportError:
    raise ImportError(
        "qdrant-client is required. Install with: pip install qdrant-client"
    ) from None

from ..config.config import WebVectorConfig
from ..embeddings.embeddings import EmbeddingResult
from ..errors.exceptions import StorageError

logger = logging.getLogger(__name__)


class VectorStorage:
    """Manages vector storage operations with Qdrant database."""
    
    def __init__(self, config: WebVectorConfig):
        """Initialize the vector storage manager."""
        self.config = config
        self.client = AsyncQdrantClient(url=config.qdrant_url)
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.client:
            await self.client.close()
    
    async def create_collection(self, recreate: bool = False) -> bool:
        """
        Create or recreate the collection for storing vectors.
        
        Args:
            recreate: If True, delete existing collection and create new one
            
        Returns:
            True if collection was created, False if it already existed
            
        Raises:
            StorageError: If collection creation fails
        """
        try:
            collection_name = self.config.collection_name
            
            # Check if collection exists
            collections = await self.client.get_collections()
            collection_exists = any(
                collection.name == collection_name 
                for collection in collections.collections
            )
            
            if collection_exists:
                if recreate:
                    logger.info(f"Deleting existing collection: {collection_name}")
                    await self.client.delete_collection(collection_name)
                else:
                    logger.info(f"Collection {collection_name} already exists")
                    return False
            
            # Create collection with vector configuration
            logger.info(f"Creating collection: {collection_name}")
            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.config.embedding_dimensions,
                    distance=Distance.COSINE
                )
            )
            
            logger.info(f"Successfully created collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise StorageError(f"Failed to create collection: {str(e)}") from e
    
    async def store_embeddings(self, embeddings: list[EmbeddingResult]) -> int:
        """
        Store embeddings in the Qdrant collection.
        
        Args:
            embeddings: List of embedding results to store
            
        Returns:
            Number of successfully stored embeddings
            
        Raises:
            StorageError: If storage operation fails
        """
        if not embeddings:
            logger.warning("No embeddings to store")
            return 0
        
        try:
            # Ensure collection exists
            await self.create_collection(recreate=False)
            
            # Convert embeddings to Qdrant points
            points = []
            for embedding in embeddings:
                point = PointStruct(
                    id=str(uuid4()),  # Generate unique ID
                    vector=embedding.embedding,
                    payload={
                        'text': embedding.text,
                        'chunk_id': embedding.chunk_id,
                        'token_count': embedding.token_count,
                        'url': embedding.metadata.get('url', ''),
                        'title': embedding.metadata.get('title', ''),
                        'description': embedding.metadata.get('description', ''),
                        'keywords': embedding.metadata.get('keywords', []),
                        'language': embedding.metadata.get('language', ''),
                        'word_count': embedding.metadata.get('word_count', 0),
                        'chunk_index': embedding.metadata.get('chunk_index', 0),
                        'total_chunks': embedding.metadata.get('total_chunks', 1),
                        'chunk_size': embedding.metadata.get('chunk_size', 0),
                        'scraped_at': embedding.metadata.get('scraped_at', 0),
                        'status_code': embedding.metadata.get('status_code', 200)
                    }
                )
                points.append(point)
            
            # Store points in batches
            batch_size = self.config.batch_size
            stored_count = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                await self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=batch
                )
                
                stored_count += len(batch)
                logger.info(f"Stored batch {i//batch_size + 1}: {len(batch)} embeddings")
            
            logger.info(f"Successfully stored {stored_count} embeddings")
            return stored_count
            
        except Exception as e:
            logger.error(f"Error storing embeddings: {str(e)}")
            raise StorageError(f"Failed to store embeddings: {str(e)}") from e
    
    async def search_similar(
        self,
        query_embedding: list[float],
        limit: int = 10,
        score_threshold: float = 0.7,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors in the collection.
        
        Args:
            query_embedding: Query vector to search for
            limit: Maximum number of results to return
            score_threshold: Minimum similarity score threshold
            filters: Optional filters to apply to search
            
        Returns:
            List of search results with scores and metadata
            
        Raises:
            StorageError: If search operation fails
        """
        try:
            # Build filter conditions if provided
            search_filter = None
            if filters:
                conditions = []
                
                for field, value in filters.items():
                    if isinstance(value, str):
                        conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=value))
                        )
                    elif isinstance(value, list) and value:
                        # For list values, match any item in the list
                        for item in value:
                            conditions.append(
                                FieldCondition(key=field, match=MatchValue(value=item))
                            )
                
                if conditions:
                    search_filter = Filter(should=conditions)
            
            # Perform vector search
            search_results = await self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=search_filter,
                with_payload=True,
                with_vectors=False  # Don't return vectors to save bandwidth
            )
            
            # Format results
            results: list[dict[str, Any]] = []
            for result in search_results:
                formatted_result = {
                    'id': result.id,
                    'score': result.score,
                    'text': result.payload.get('text', ''),
                    'chunk_id': result.payload.get('chunk_id', ''),
                    'url': result.payload.get('url', ''),
                    'title': result.payload.get('title', ''),
                    'description': result.payload.get('description', ''),
                    'keywords': result.payload.get('keywords', []),
                    'word_count': result.payload.get('word_count', 0),
                    'chunk_index': result.payload.get('chunk_index', 0),
                    'total_chunks': result.payload.get('total_chunks', 1),
                    'metadata': result.payload
                }
                results.append(formatted_result)
            
            logger.info(f"Found {len(results)} similar results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            raise StorageError(f"Failed to search vectors: {str(e)}") from e
    
    async def get_collection_info(self) -> dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information including point count and configuration
            
        Raises:
            StorageError: If operation fails
        """
        try:
            collection_info = await self.client.get_collection(self.config.collection_name)
            
            return {
                'name': collection_info.config.params.vectors.size,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.value,
                'points_count': collection_info.points_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'status': collection_info.status.value
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            raise StorageError(f"Failed to get collection info: {str(e)}") from e
    
    async def delete_collection(self) -> bool:
        """
        Delete the collection.
        
        Returns:
            True if collection was deleted successfully
            
        Raises:
            StorageError: If deletion fails
        """
        try:
            await self.client.delete_collection(self.config.collection_name)
            logger.info(f"Successfully deleted collection: {self.config.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise StorageError(f"Failed to delete collection: {str(e)}") from e
    
    async def count_points(self, filters: Optional[dict[str, Any]] = None) -> int:
        """
        Count points in the collection with optional filters.
        
        Args:
            filters: Optional filters to apply
            
        Returns:
            Number of points matching the criteria
        """
        try:
            # Build filter if provided
            count_filter = None
            if filters:
                conditions = []
                for field, value in filters.items():
                    conditions.append(
                        FieldCondition(key=field, match=MatchValue(value=value))
                    )
                if conditions:
                    count_filter = Filter(must=conditions)
            
            result = await self.client.count(
                collection_name=self.config.collection_name,
                count_filter=count_filter
            )
            
            return result.count
            
        except Exception as e:
            logger.error(f"Error counting points: {str(e)}")
            raise StorageError(f"Failed to count points: {str(e)}") from e
