import json
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric,AnswerRelevancyMetric
from app.agent import run_enterprise_agent

#Load Golden dataset
def load_goldens():
    path = "datasets/goldens.json"
    with open(path,"r") as f:
        return json.load(f)

@pytest.mark.parametrize("golden",load_goldens())
def test_enterprise_agent_eval(golden):
    #Execute agent workflow
    result = run_enterprise_agent(golden["input"])

    print(r'Result input: '+ result["input"])
    print(r'Result output: '+ result["output"])

    test_case = LLMTestCase(
        input = result["input"],
        actual_output=result["output"],
        retrieval_context=result["retrieval_context"],
        expected_output = golden.get("expected_output"))

    #Define enterprise qality gates(thresholds)
    faithfulness_metric = FaithfulnessMetric(threshold=.7)
    relvancy_metric = AnswerRelevancyMetric(threshold=.8)

    #assert criteria pass
    assert_test(test_case,[faithfulness_metric,relvancy_metric])