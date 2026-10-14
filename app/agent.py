import os
from deepeval.tracing import observe

def lookup_internal_docs(query:str) -> str:
    if "remote work" in query.lower():
        return "Company Policy: Employees are allowed up to 2 days of remote work per week with manager approval."
    return "Company Policy: Standard core working hours are 9 AM to 5 PM local time."

@observe(name="EnterpriseSupportAgent")

def run_enterprise_agent(user_input: str) -> dict:
    """
    Simulates a multi-step agent routing through tool calls and generating a final response.
    """
    
    #Step1:Planning / Tool selection
    tool_output = ""
    if "remote work" in user_input.lower() or "hours" in user_input.lower():
        tool_output = lookup_internal_docs(user_input)

    #step 2 Generation based on context
    response = (
        f"Based on internam guidelines: {tool_output}"
        "Please submit requests via the intenral portal"
    )

    return {
        "input": user_input,
        "output": response,
        "retrieval_context":[tool_output]
    }