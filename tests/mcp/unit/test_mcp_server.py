import pytest
from mcp import Client
from app.mcp_server.server import mcp, POLICIES_DIR

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    async with Client(mcp) as c:
        yield c


async def test_server_exposes_expected_tools(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {"list_policies", "read_policy", "get_leave_balance",
                          "submit_leave_request", "cancel_leave_request"}
    assert all(t.description for t in tools.values())


async def test_read_policy_schema_requires_name(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    schema = tools["read_policy"].input_schema
    assert schema["required"] == ["name"]
    assert schema["properties"]["name"]["type"] == "string"


async def test_list_policies_matches_files_on_disk(client):
    result = await client.call_tool("list_policies", {})
    assert not result.is_error
    assert result.structured_content["result"] == sorted(p.stem for p in POLICIES_DIR.glob("*.md"))


async def test_read_policy_returns_document_text(client):
    result = await client.call_tool("read_policy", {"name": "leave"})
    assert not result.is_error
    assert "annual leave" in result.content[0].text.lower()


@pytest.mark.parametrize("bad_name", ["foo", "../../.env", "", "LEAVE"])
async def test_read_policy_rejects_unknown_names(client, bad_name):
    result = await client.call_tool("read_policy", {"name": bad_name})
    assert result.is_error
    assert "list_policies" in result.content[0].text


async def test_read_policy_missing_argument_is_error(client):
    result = await client.call_tool("read_policy", {})
    assert result.is_error