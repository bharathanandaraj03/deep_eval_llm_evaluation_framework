import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def require_api_keys():
    missing = [k for k in ("ANTHROPIC_API_KEY","OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        pytest.skip(f"Missing {''.join(missing)}")
