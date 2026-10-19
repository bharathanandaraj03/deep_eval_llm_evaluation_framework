import copy
import pytest
from mcp import Client
from app.mcp_server import server
from app.mcp_server.server import mcp

pytestmark = pytest.mark.anyio

NOV_2_TO_3 = {"start_date": "2026-11-02", "end_date": "2026-11-03"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_state():
    balances = copy.deepcopy(server.LEAVE_BALANCES)
    yield
    server.LEAVE_BALANCES.clear()
    server.LEAVE_BALANCES.update(balances)
    server.LEAVE_REQUESTS.clear()


@pytest.fixture
async def client():
    async with Client(mcp) as c:
        yield c


async def submit(client, employee_id="E100", leave_type="annual", **dates):
    args = {"employee_id": employee_id, "leave_type": leave_type, **(dates or NOV_2_TO_3)}
    return await client.call_tool("submit_leave_request", args)


async def balance(client, employee_id="E100"):
    return (await client.call_tool("get_leave_balance", {"employee_id": employee_id})).structured_content


async def test_annotations_declare_safety(client):
    tools = {t.name: t.annotations for t in (await client.list_tools()).tools}
    assert tools["get_leave_balance"].read_only_hint is True
    assert tools["submit_leave_request"].destructive_hint is False
    assert tools["cancel_leave_request"].destructive_hint is True


async def test_submit_deducts_balance(client):
    result = await submit(client)
    assert not result.is_error
    assert result.structured_content == {"request_id": "LR001", "days": 2, "status": "submitted"}
    assert (await balance(client))["annual"] == 18


async def test_unpaid_leave_not_deducted(client):
    await submit(client, leave_type="unpaid")
    assert (await balance(client))["annual"] == 20


async def test_insufficient_balance_rejected_without_side_effects(client):
    result = await submit(client, employee_id="E200", start_date="2026-11-02", end_date="2026-11-06")
    assert result.is_error
    assert "insufficient annual balance" in result.content[0].text
    assert (await balance(client, "E200"))["annual"] == 3
    assert server.LEAVE_REQUESTS == {}


@pytest.mark.parametrize("args", [
    {"leave_type": "vacation"},
    {"start_date": "02/11/2026", "end_date": "2026-11-03"},
    {"start_date": "2026-11-05", "end_date": "2026-11-03"},
    {"employee_id": "E999"},
])
async def test_submit_rejects_invalid_input(client, args):
    result = await submit(client, **args)
    assert result.is_error
    assert server.LEAVE_REQUESTS == {}


async def test_cancel_refunds_and_is_not_repeatable(client):
    request_id = (await submit(client)).structured_content["request_id"]
    first = await client.call_tool("cancel_leave_request", {"request_id": request_id})
    second = await client.call_tool("cancel_leave_request", {"request_id": request_id})
    assert first.structured_content["status"] == "cancelled"
    assert (await balance(client))["annual"] == 20
    assert second.is_error