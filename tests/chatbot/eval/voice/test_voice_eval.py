import json
import pytest

from functools import lru_cache
from deepeval.metrics import AnswerRelevancyMetric,FaithfulnessMetric,GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from app.voice.agent import synthesize, transcribe, voice_turn
from app.voice.audio import add_noise
from tests.chatbot.eval.conftest import DATASETS, JUDGE_MODEL, THRESHOLDS

from tests.chatbot.reporting import evaluate_and_record
from tests.chatbot.eval.voice.metrics import WERMetric

GOlDENS = json.loads((DATASETS / "chatbot_goldens.json").read_text())[:5]
MAX_WER = {None:0.10, 20:0.15, 10: 0.25} #Snr (db) -> Allowed ASR word error rate
LATENCY_BUDGET_S = 6.0

@lru_cache
def caller_audio(text:str, snr_db: float
                 | None) -> bytes:
    clean = synthesize(text)
    return clean if snr_db is None else add_noise(clean, snr_db)

@lru_cache
def run_turn(text: str) -> dict:
    return voice_turn([],caller_audio(text,None))

@pytest.mark.parametrize("snr_db",MAX_WER, ids=lambda s:"clean" if s is None else f"snr{s}")
@pytest.mark.parametrize("golden",GOlDENS,ids=lambda g: g["input"][:30])
def test_asr_accuracy(golden, snr_db):
    hypothesis = transcribe(caller_audio(golden["input"],snr_db))
    test_case = LLMTestCase(input=golden["input"],actual_output=hypothesis,expected_output=golden["input"])
    evaluate_and_record(test_case,[WERMetric(MAX_WER[snr_db], name = "ASR WER")])
                                        
@pytest.mark.parametrize("golden",GOlDENS,ids=lambda g: g["input"][:30])
def test_voice_answer_quality(golden):
    result = run_turn(golden["input"])
    test_case = LLMTestCase(input=result["transcript"],
                            actual_output=result["reply"],
                            expected_output=golden["expected_output"],
                            retrieval_context=result["retrieval_context"],
    )

    evaluate_and_record(test_case,[
        AnswerRelevancyMetric(threshold=THRESHOLDS["answer_relevancy"],model=JUDGE_MODEL),
        FaithfulnessMetric(threshold=THRESHOLDS["faithfulness"],model=JUDGE_MODEL,
                           ),
        GEval(name="Spoken Response Quality",
              criteria="The reply will be read aloud by a voice agent. It must be under 3 sentences, contain no "
                        "markdown, bullets, URLs, file names or tables, and answer only what was asked.",
             evaluation_params = [SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
                threshold = THRESHOLDS["spoken_style"], model = JUDGE_MODEL),
         ])
    

@pytest.mark.parametrize("golden",  GOlDENS, ids=lambda g : g["input"][:30])
def test_tts_intelligibility(golden):
    result = run_turn(golden["input"])
    test_case = LLMTestCase(input=result["reply"],actual_output=transcribe(result["reply_audio"]),
                            expected_output=result["reply"])
    evaluate_and_record(test_case,[WERMetric(0.10,name="TTS round-trip WER")])

@pytest.mark.parametrize("golden",GOlDENS,ids= lambda g: g["input"][:30])
def test_turn_latency(golden):
    latency = run_turn(golden["input"])["latency"]
    assert latency["total"] <= LATENCY_BUDGET_S,f"{latency=}"