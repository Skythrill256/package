"""
Custom exceptions for the WebVector package.
"""


class WebVectorError(Exception):
    """Base exception for all WebVector errors."""
    pass


class ScrapingError(WebVectorError):
    """Raised when web scraping operations fail."""
    pass


class EmbeddingError(WebVectorError):
    """Raised when embedding generation fails."""
    pass


class StorageError(WebVectorError):
    """Raised when vector storage operations fail."""
    pass


class QueryError(WebVectorError):
    """Raised when query operations fail."""
    pass


class ConfigurationError(WebVectorError):
    """Raised when configuration is invalid."""
    pass
