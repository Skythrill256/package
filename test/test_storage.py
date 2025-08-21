import math

import pytest


@pytest.mark.asyncio
async def test_create_and_store_and_search_in_qdrant(qdrant_available, test_config, cleanup_collection):
    """End-to-end storage test against local Qdrant using synthetic embeddings."""
    try:
        from package.embeddings.embeddings import EmbeddingResult
        from package.storage.storage import VectorStorage
    except ImportError as e:
        pytest.skip(f"Optional dependency missing for storage tests: {e}")

    storage = VectorStorage(test_config)

    vec_dim = test_config.embedding_dimensions
    # two orthogonal unit vectors and one mid vector
    v1 = [1.0] + [0.0] * (vec_dim - 1)
    v2 = [0.0, 1.0] + [0.0] * (vec_dim - 2) if vec_dim >= 2 else [0.0] * vec_dim
    vm = [1/math.sqrt(2), 1/math.sqrt(2)] + [0.0] * (vec_dim - 2)

    data = [
        EmbeddingResult(text="doc one", embedding=v1, token_count=5, chunk_id="c1", metadata={"url": "http://a"}),
        EmbeddingResult(text="doc two", embedding=v2, token_count=5, chunk_id="c2", metadata={"url": "http://b"}),
        EmbeddingResult(text="doc mid", embedding=vm, token_count=5, chunk_id="c3", metadata={"url": "http://c"}),
    ]

    async with storage:
        created = await storage.create_collection(recreate=True)
        assert created is True
        stored = await storage.store_embeddings(data)
        assert stored == len(data)

        # search near v1 should retrieve doc one or doc mid with good score
        results = await storage.search_similar(query_embedding=v1, limit=2, score_threshold=0.5)
        assert len(results) >= 1
        urls = {r["url"] for r in results}
        assert "http://a" in urls

        info = await storage.get_collection_info()
        assert info["points_count"] >= 3

        count = await storage.count_points()
        assert count >= 3
