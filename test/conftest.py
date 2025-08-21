import asyncio
import os
import sys
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Ensure project root is on sys.path to import `package`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from package.config.config import WebVectorConfig  # noqa: E402

# Load .env once for all tests from project root
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests at session scope (pytest-asyncio >=0.21 default is function)."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        pytest.skip("OPENAI_API_KEY not set in environment/.env; skipping tests that require it.")
    return key


@pytest.fixture()
def test_config(openai_api_key: str) -> WebVectorConfig:
    """Config fixture with small embedding dimensions for faster tests and a unique collection."""
    collection = f"test_collection_{uuid.uuid4().hex[:8]}"
    return WebVectorConfig(
        openai_api_key=openai_api_key,
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection_name=collection,
        # Use small dims to speed up local qdrant ops in tests
        embedding_model="text-embedding-3-small",
        embedding_dimensions=16,
        chunk_size=200,
        chunk_overlap=50,
        batch_size=16,
        max_pages=5,
        max_depth=1,
    )


@pytest.fixture()
async def qdrant_available(test_config: WebVectorConfig):
    """Skip tests if local Qdrant is not reachable."""
    try:
        from qdrant_client import AsyncQdrantClient
    except ImportError:
        pytest.skip("qdrant-client is not installed")

    client = AsyncQdrantClient(url=test_config.qdrant_url)
    try:
        await client.get_collections()
    except Exception:
        await client.close()
        pytest.skip("Local Qdrant not reachable at http://localhost:6333")
    else:
        await client.close()
        return True


@pytest.fixture()
async def cleanup_collection(test_config: WebVectorConfig):
    """Auto-delete the test collection after a test finishes."""
    yield
    try:
        from package.storage.storage import VectorStorage
        storage = VectorStorage(test_config)
        async with storage:
            await storage.delete_collection()
    except Exception:
        # Best-effort cleanup
        pass
