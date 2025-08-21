import pytest


@pytest.mark.asyncio
async def test_query_engine_initialize(qdrant_available, test_config):
    """Initialize the query engine with local Qdrant. Skips if llama-index not installed."""
    try:
        import llama_index  # noqa: F401

        from package.queries.query import QueryEngine
    except Exception:
        pytest.skip("llama-index not installed")

    engine = QueryEngine(test_config)
    await engine.initialize()
    assert engine.query_engine is not None
