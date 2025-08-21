"""
Main WebVector client that orchestrates all components.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from ..config.config import WebVectorConfig
from ..crawling.scraping import WebScraper
from ..embeddings.embeddings import EmbeddingManager
from ..errors.exceptions import WebVectorError
from ..queries.query import QueryEngine
from ..storage.storage import VectorStorage

logger = logging.getLogger(__name__)


class WebVectorClient:
    """
    Main client for WebVector operations.
    
    This class orchestrates web scraping, embedding generation,
    vector storage, and querying operations.
    """
    
    def __init__(self, config: WebVectorConfig):
        """
        Initialize WebVector client.
        
        Args:
            config: WebVector configuration object
        """
        self.config = config
        self.scraper = WebScraper(config)
        self.embedding_manager = EmbeddingManager(config)
        self.vector_storage = VectorStorage(config)
        self.query_engine = QueryEngine(config)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Do not configure logging handlers by default to keep package silent."""
        # Intentionally left blank. Application code should configure logging.
        pass
        return
    
    async def scrape_and_store(
        self, 
        urls: list[str], 
        recreate_collection: bool = False
    ) -> dict[str, Any]:
        """
        Complete pipeline: scrape websites, generate embeddings, and store in vector DB.
        
        Args:
            urls: List of URLs to scrape
            recreate_collection: Whether to recreate the collection
            
        Returns:
            Summary of the operation
            
        Raises:
            WebVectorError: If any step of the pipeline fails
        """
        try:
            logger.info(f"Starting scrape and store pipeline for {len(urls)} URLs")
            
            # Step 1: Scrape websites
            logger.info("Step 1: Scraping websites...")
            scraped_data = []
            
            async with self.scraper:
                if len(urls) == 1:
                    scraped_data = await self.scraper.scrape_website(urls[0])
                else:
                    scraped_data = await self.scraper.scrape_multiple_sites(urls)
            
            if not scraped_data:
                raise WebVectorError("No content was scraped from the provided URLs")
            
            logger.info(f"Scraped {len(scraped_data)} pages")
            
            # Step 2: Generate embeddings
            logger.info("Step 2: Generating embeddings...")
            embeddings = await self.embedding_manager.generate_embeddings(scraped_data)
            
            if not embeddings:
                raise WebVectorError("No embeddings were generated")
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Step 3: Store in vector database
            logger.info("Step 3: Storing in vector database...")
            async with self.vector_storage:
                # Create or recreate collection
                await self.vector_storage.create_collection(recreate=recreate_collection)
                
                # Store embeddings
                stored_count = await self.vector_storage.store_embeddings(embeddings)
            
            # Step 4: Initialize query engine
            logger.info("Step 4: Initializing query engine...")
            await self.query_engine.initialize()
            
            # Prepare summary
            summary = {
                'urls_processed': len(urls),
                'pages_scraped': len(scraped_data),
                'embeddings_generated': len(embeddings),
                'embeddings_stored': stored_count,
                'collection_name': self.config.collection_name,
                'embedding_model': self.config.embedding_model,
                'embedding_dimensions': self.config.embedding_dimensions,
                'success': True
            }
            
            logger.info("Pipeline completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise WebVectorError(f"Scrape and store pipeline failed: {str(e)}") from e
    
    async def query(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        include_sources: bool = True
    ) -> dict[str, Any]:
        """
        Query the vector database using natural language.
        
        Args:
            question: Natural language question
            top_k: Number of top results to consider
            similarity_threshold: Minimum similarity score
            include_sources: Whether to include source information
            
        Returns:
            Query response with answer and sources
        """
        try:
            # Ensure query engine is initialized
            if not self.query_engine.query_engine:
                await self.query_engine.initialize()
            
            return await self.query_engine.query(
                question=question,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                include_sources=include_sources
            )
            
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise WebVectorError(f"Query failed: {str(e)}") from e
    
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
        """
        try:
            return await self.query_engine.similarity_search(
                query=query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=filters
            )
            
        except Exception as e:
            logger.error(f"Similarity search failed: {str(e)}")
            raise WebVectorError(f"Similarity search failed: {str(e)}") from e
    
    async def get_collection_stats(self) -> dict[str, Any]:
        """Get statistics about the vector collection."""
        try:
            return await self.query_engine.get_collection_stats()
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            raise WebVectorError(f"Failed to get collection stats: {str(e)}") from e
    
    async def add_urls(self, urls: list[str]) -> dict[str, Any]:
        """
        Add new URLs to existing collection without recreating it.
        
        Args:
            urls: List of URLs to add
            
        Returns:
            Summary of the operation
        """
        return await self.scrape_and_store(urls, recreate_collection=False)
    
    async def recreate_collection(self, urls: list[str]) -> dict[str, Any]:
        """
        Recreate the collection with new URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            Summary of the operation
        """
        return await self.scrape_and_store(urls, recreate_collection=True)
    
    @asynccontextmanager
    async def batch_operations(self):
        """
        Context manager for batch operations to optimize performance.
        """
        try:
            # Pre-initialize components
            await self.query_engine.initialize()
            yield self
        finally:
            # Cleanup if needed
            pass
    
    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Health status of all components
        """
        health_status = {
            'overall_status': 'healthy',
            'components': {}
        }
        
        try:
            # Check vector storage
            async with self.vector_storage:
                try:
                    info = await self.vector_storage.get_collection_info()
                    health_status['components']['vector_storage'] = {
                        'status': 'healthy',
                        'collection_exists': True,
                        'points_count': info.get('points_count', 0)
                    }
                except Exception as e:
                    health_status['components']['vector_storage'] = {
                        'status': 'unhealthy',
                        'error': str(e)
                    }
                    health_status['overall_status'] = 'degraded'
            
            # Check embedding service
            try:
                test_embedding = await self.embedding_manager.generate_query_embedding("test")
                health_status['components']['embedding_service'] = {
                    'status': 'healthy',
                    'embedding_dimensions': len(test_embedding)
                }
            except Exception as e:
                health_status['components']['embedding_service'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'
            
            # Check query engine
            try:
                if not self.query_engine.query_engine:
                    await self.query_engine.initialize()
                health_status['components']['query_engine'] = {
                    'status': 'healthy',
                    'initialized': True
                }
            except Exception as e:
                health_status['components']['query_engine'] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'
            
            return health_status
            
        except Exception as e:
            return {
                'overall_status': 'unhealthy',
                'error': str(e),
                'components': health_status.get('components', {})
            }
