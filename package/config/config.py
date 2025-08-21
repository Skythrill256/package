"""
Configuration management for WebVector package.
"""

from dataclasses import dataclass
from typing import Any, Optional

from ..errors.exceptions import ConfigurationError


@dataclass
class WebVectorConfig:
    """Configuration class for WebVector operations."""
    
    # Required fields (no defaults) must come first
    openai_api_key: str
    # Qdrant Configuration (required)
    qdrant_url: str
    collection_name: str

    # Optional / defaulted fields
    openai_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072  # text-embedding-3-large default
    
    # Scraping Configuration
    max_pages: int = 50
    max_depth: int = 2
    include_external: bool = False
    score_threshold: float = 0.3
    
    # Processing Configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 50
    
    # Optional filters and keywords
    keywords: Optional[list[str]] = None
    url_patterns: Optional[list[str]] = None
    exclude_patterns: Optional[list[str]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.openai_api_key:
            raise ConfigurationError("OpenAI API key is required")
        
        if not self.qdrant_url:
            raise ConfigurationError("Qdrant URL is required")
        
        if not self.collection_name:
            raise ConfigurationError("Collection name is required")
        
        if self.max_pages <= 0:
            raise ConfigurationError("max_pages must be positive")
        
        if self.max_depth < 0:
            raise ConfigurationError("max_depth must be non-negative")
        
        if self.embedding_dimensions <= 0:
            raise ConfigurationError("embedding_dimensions must be positive")
        
        if self.chunk_size <= 0:
            raise ConfigurationError("chunk_size must be positive")
        
        if self.chunk_overlap < 0:
            raise ConfigurationError("chunk_overlap must be non-negative")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("chunk_overlap must be less than chunk_size")
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> 'WebVectorConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'openai_api_key': self.openai_api_key,
            'openai_model': self.openai_model,
            'embedding_model': self.embedding_model,
            'embedding_dimensions': self.embedding_dimensions,
            'qdrant_url': self.qdrant_url,
            'collection_name': self.collection_name,
            'max_pages': self.max_pages,
            'max_depth': self.max_depth,
            'include_external': self.include_external,
            'score_threshold': self.score_threshold,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'batch_size': self.batch_size,
            'keywords': self.keywords,
            'url_patterns': self.url_patterns,
            'exclude_patterns': self.exclude_patterns
        }
