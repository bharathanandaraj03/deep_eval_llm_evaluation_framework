import json
import pytest
from deepeval import assert_test
from deepeval.metrics import (AnswerRelevancyMetric, 
                              ContextualPrecisionMetric, 
                              ContextualRecallMetric,
                              ContextualRelevancyMetric,
                              FaithfulnessMetric)
from deepeval.test_case import LLMTestCase
from app.chatbot.bot import chat
from tests.chatbot.eval.conftest import DATASETS,JUDGE_MODEL,THRESHOLDS
from tests.chatbot.reporting import evaluate_and_record

GOLDENS = json.loads((DATASETS/"chatbot_goldens.json").read_text())

@pytest.mark.parametrize("golden",GOLDENS, ids=lambda g: g["input"][:40])
def test_rag_answer_quality(golden):
    result = chat([], golden["input"])

    test_case = LLMTestCase(
        input=golden["input"],
        actual_output=result["reply"],
        expected_output=golden["expected_output"],
        retrieval_context=result["retrieval_context"]
    )

    metrics = [
        ContextualPrecisionMetric(threshold=THRESHOLDS["contextual_precision"],model=JUDGE_MODEL),
        ContextualRecallMetric(threshold=THRESHOLDS["contextual_recall"],model=JUDGE_MODEL),
        ContextualRelevancyMetric(threshold=THRESHOLDS["contextual_relvancy"],model= JUDGE_MODEL),
        FaithfulnessMetric(threshold=THRESHOLDS["faithfulness"],model= JUDGE_MODEL),
        AnswerRelevancyMetric(threshold=THRESHOLDS["answer_relevancy"],model=JUDGE_MODEL)
    ]
    
    evaluate_and_record(test_case,metrics)