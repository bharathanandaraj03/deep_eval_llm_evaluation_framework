from deepeval.tracing import observe

@observe(type="tool")
def lookup_internal_docs(query: str) -> str:
    if "remote work" in query.lower():
        return "Company Policy: Employees are allowed up to 2 days of remote work per week with manager approval."
    return "Company Policy: Standard core working hours are 9 AM to 5 PM local time."
