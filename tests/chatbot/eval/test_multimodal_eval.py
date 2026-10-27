import json
import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, MLLMImage,SingleTurnParams
from app.chatbot.bot import chat
from tests.chatbot.eval.conftest import DATASETS,JUDGE_MODEL,THRESHOLDS
from tests.chatbot.reporting import evaluate_and_record

GOLDENS = json.loads((DATASETS/"multimodal_goldens.json").read_text())

@pytest.mark.parametrize("golden",GOLDENS, ids=lambda
                         g:g["image"].rsplit("/",1)[-1])
def test_receipt_answer(golden):
    image = MLLMImage(url=golden["image"])
    result = chat([],golden["input"],images=[image])

    test_case = LLMTestCase(
        input=f"{image}\n{golden['input']}",
        actual_output=result["reply"],
        expected_output=golden["expected_output"],
        retrieval_context=result["retrieval_context"],
    )

    evaluate_and_record(test_case,[
        AnswerRelevancyMetric(threshold=THRESHOLDS["answer_relevancy"],model=JUDGE_MODEL),
        FaithfulnessMetric(threshold=THRESHOLDS["faithfulness"],model=JUDGE_MODEL),
        GEval(name="Receipt Grounding",
              criteria="The answer correctly read the vendor,amount and expense category from the image"
                        "and applies the matching policy rule. inventing details not visible in the image is a failure.",
                        evaluation_params=[SingleTurnParams.INPUT,SingleTurnParams.ACTUAL_OUTPUT,SingleTurnParams.EXPECTED_OUTPUT],
                        threshold=THRESHOLDS["receipt_grounding"], model=JUDGE_MODEL,
                        )
    ])