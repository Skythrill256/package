import pytest

from package.config.config import WebVectorConfig
from package.errors.exceptions import ConfigurationError


def test_config_validates_required_fields(openai_api_key):
    cfg = WebVectorConfig(
        openai_api_key=openai_api_key,
        qdrant_url="http://localhost:6333",
        collection_name="test"
    )
    # Should not raise
    cfg.validate()


def test_config_missing_api_key_raises():
    with pytest.raises(ConfigurationError):
        WebVectorConfig(
            openai_api_key="",
            qdrant_url="http://localhost:6333",
            collection_name="x"
        )


def test_config_invalid_numbers(openai_api_key):
    with pytest.raises(ConfigurationError):
        WebVectorConfig(
            openai_api_key=openai_api_key,
            qdrant_url="http://localhost:6333",
            collection_name="x",
            max_pages=0
        )
    with pytest.raises(ConfigurationError):
        WebVectorConfig(
            openai_api_key=openai_api_key,
            qdrant_url="http://localhost:6333",
            collection_name="x",
            chunk_size=100,
            chunk_overlap=100
        )
