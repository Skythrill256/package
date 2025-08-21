import pytest


@pytest.mark.asyncio
async def test_generate_query_embedding_live(openai_api_key, test_config):
    """
    Live test against OpenAI to verify embedding shape.
    Skipped automatically if OPENAI_API_KEY is not present.
    """
    # Import here to avoid ImportError at collection time if openai is unavailable
    from package.embeddings.embeddings import EmbeddingManager
    mgr = EmbeddingManager(test_config)
    emb = await mgr.generate_query_embedding("hello world")
    assert isinstance(emb, list)
    assert len(emb) == test_config.embedding_dimensions
