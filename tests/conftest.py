"""Session-level fixtures that ensure app.state is populated for all tests.

httpx.ASGITransport does not trigger the ASGI lifespan, so async tests that
use it directly would see an uninitialised app.state.  This module uses a
session-scoped TestClient context manager to run the lifespan once and leave
state populated for the duration of the test session.
"""

import pytest
from fastapi.testclient import TestClient

from llm_gateway.main import app


@pytest.fixture(scope="session", autouse=True)
def _run_lifespan():
    """Start the app lifespan once per test session."""
    with TestClient(app):
        yield
