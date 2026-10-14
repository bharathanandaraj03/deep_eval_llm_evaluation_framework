from deepeval.tracing import observe
from app.tools import lookup_internal_docs


@observe(name="EnterpriseSupportAgent")
def run_enterprise_agent(user_input: str) -> dict:
    """
    Simulates a multi-step agent routing through tool calls and generating a final response.
    """
    
    #Step1:Planning / Tool selection
    tools_called = []
    tool_output = ""
    if "remote work" in user_input.lower() or "hours" in user_input.lower():
        tool_output = lookup_internal_docs(user_input)
        tools_called.append("lookup_internal_docs")

    #step 2 Generation based on context
    response = (
        f"Based on internal guidelines: {tool_output} "
        "Please submit requests via the internal portal"
    )

    return {
        "input": user_input,
        "output": response,
        "retrieval_context":[tool_output],
        "tools_called": tools_called
    }