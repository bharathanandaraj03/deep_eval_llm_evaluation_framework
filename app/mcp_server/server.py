from pathlib import Path
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from datetime import date
from typing import Literal
from mcp_types import ToolAnnotations

POLICIES_DIR = Path(__file__).resolve().parents[2]/ "data" / "policies"

mcp = MCPServer("hr-policy")

@mcp.tool()
def list_policies() -> list[str]:
    """List the names of all available HR policy documents."""
    return sorted(p.stem for p in POLICIES_DIR.glob("*.md"))
    
    
@mcp.tool()
def read_policy(name: str) -> str:
    """Return the full text of one HR policy document, e.g. 'leave' or 'expenses'."""
    if name not in list_policies():
        raise ToolError(f"unknown policy '{name}'. Call list_policies for valid names.")

    return (POLICIES_DIR/f"{name}.md").read_text(encoding="utf-8")

LeaveType = Literal["annual", "sick", "unpaid"]

LEAVE_BALANCES = {"E100": {"annual": 20, "sick": 10}, "E200": {"annual": 3, "sick": 10}}
LEAVE_REQUESTS: dict[str, dict] = {}

@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def get_leave_balance(employee_id: str) -> dict[str, int]:
    """Return remaining leave days per leave type for an employee, e.g. employee_id 'E100'."""
    if employee_id not in LEAVE_BALANCES:
        raise ToolError(f"unknown employee '{employee_id}'.")
    return LEAVE_BALANCES[employee_id]

@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=False))
def submit_leave_request(employee_id: str, start_date: date, end_date: date, leave_type: LeaveType) -> dict[str,str | int]:
    """Submit a leave request for an employee. Dates are inclusive, format YYYY-MM-DD.
    Deducts the days from the employee's balance (unpaid leave is not deducted)."""
    balance = get_leave_balance(employee_id)
    if end_date < start_date:
        raise ToolError("end_date must be on or after start_date.")
    days = (end_date - start_date).days + 1
    if leave_type != "unpaid" and days > balance[leave_type]:
        raise ToolError(f"insufficient {leave_type} balance: requested {days}, available {balance[leave_type]}.")
    if leave_type != "unpaid":
        balance[leave_type] -= days
    request_id = f"LR{len(LEAVE_REQUESTS) + 1:03d}"
    LEAVE_REQUESTS[request_id] = {"employee_id": employee_id, "leave_type": leave_type, "days": days, "status": "submitted"}
    return {"request_id": request_id, "days": days, "status": "submitted"}


@mcp.tool(annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True))
def cancel_leave_request(request_id: str) -> dict[str,str]:
    """Cancel a submitted leave request and refund its days. Only call after the user has confirmed."""
    request = LEAVE_REQUESTS.get(request_id)
    if request is None or request["status"] != "submitted":
        raise ToolError(f"no active leave request '{request_id}'.")
    request["status"] = "cancelled"
    if request["leave_type"] != "unpaid":
        LEAVE_BALANCES[request["employee_id"]][request["leave_type"]] += request["days"]
    return {"request_id": request_id, "status": "cancelled"}

if __name__ == "__main__":
    mcp.run()