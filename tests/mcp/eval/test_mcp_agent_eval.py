import asyncio
import pytest
import json
from deepeval.metrics import ToolCorrectnessMetric, MCPUseMetric, ToolPermissionMetric
from deepeval.test_case import LLMTestCase, MCPServer, MCPToolCall,ToolCall
from app.mcp_server.agent import run_mcp_agent
from tests.chatbot.eval.conftest import DATASETS, THRESHOLDS,JUDGE_MODEL
from tests.chatbot.reporting import evaluate_and_record

GOLDENS = json.loads((DATASETS/"mcp_goldens.json").read_text())

def build_test_case(golden:dict,result:dict) -> LLMTestCase:
    return LLMTestCase(
        input=golden["input"],
        actual_output=result["output"],
        expected_output=golden["expected_output"],
        mcp_servers=[MCPServer(server_name="hr-policy", transport="stdio",available_tools=result["available_tools"])],
        mcp_tools_called=[
            MCPToolCall(name=c["name"],args=c["input"],result=c["result"])
            for c in result["tools_called"]
        ],
        tools_called=[ToolCall(name=c["name"],input_parameters=c["input"]) for c in result["tools_called"]],
        expected_tools=[ToolCall(name=n) for n in golden["expected_tools"]],
        
    )

@pytest.mark.parametrize("golden",GOLDENS,ids=lambda g:g["input"][:40])
def test_mcp_agent(golden):
    result = asyncio.run(run_mcp_agent(golden["input"]))
    test_case = build_test_case(golden,result)

    metrics = [
        ToolCorrectnessMetric(threshold=THRESHOLDS["tool_correctness"]),
        ToolPermissionMetric(denied_tools=golden["denied_tools"], threshold=THRESHOLDS["tool_permission"]),
        MCPUseMetric(threshold=THRESHOLDS["mcp_use"],model=JUDGE_MODEL),
    ]

    evaluate_and_record(test_case,metrics)