import pytest


@pytest.mark.asyncio
async def test_health_check_smoke(test_config):
    """
    Smoke test for health_check. It will report degraded/unhealthy if services are missing,
    which is acceptable for the test as long as it returns the expected shape.
    """
    # Import inside test to avoid ImportError at collection time if optional deps are missing
    try:
        from package.core.core import WebVectorClient
    except ImportError as e:
        pytest.skip(f"Optional dependency missing for core client import: {e}")
    client = WebVectorClient(test_config)
    status = await client.health_check()
    assert isinstance(status, dict)
    assert "overall_status" in status
    assert "components" in status
