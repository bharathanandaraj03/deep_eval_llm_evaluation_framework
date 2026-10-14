import json
from pathlib import Path
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase,ToolCall
from deepeval.metrics import FaithfulnessMetric,AnswerRelevancyMetric,HallucinationMetric,ToolCorrectnessMetric
from app.agent import run_enterprise_agent

GOLDENS_PATH = Path(__file__).parent.parent / "datasets"/ "goldens.json"
JUDGE_MODEL = "gpt-4.1"

#Load Golden dataset
def load_goldens():
        return json.loads(GOLDENS_PATH.read_text())

@pytest.mark.parametrize("golden",load_goldens(),ids=lambda g: g["input"][:40])
def test_enterprise_agent_eval(golden):
    #Execute agent workflow
    result = run_enterprise_agent(golden["input"])

    print(r'Result input: '+ result["input"])
    print(r'Result output: '+ result["output"])

    test_case = LLMTestCase(
        input = result["input"],
        actual_output=result["output"],
        retrieval_context=result["retrieval_context"],
        expected_output = golden.get("expected_output"),
        context = golden.get("context"),
        tools_called=[ToolCall(name=t) for t in result["tools_called"]],
        expected_tools=[ToolCall(name=t) for t in golden.get("expected_tools", [])])

    #Define enterprise qality gates(thresholds)
    metrics = [
       FaithfulnessMetric(threshold=.7,model=JUDGE_MODEL),
       AnswerRelevancyMetric(threshold=.8,model=JUDGE_MODEL),
       HallucinationMetric(threshold=0.3, model=JUDGE_MODEL),
        ToolCorrectnessMetric(threshold=1.0),
    ]
    

    #assert criteria pass
    assert_test(test_case,metrics)