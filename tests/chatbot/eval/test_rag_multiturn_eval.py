import pytest
import json
import datasets
from deepeval import assert_test
from deepeval.test_case import ConversationalTestCase,Turn
from deepeval.metrics import (TurnContextualRelevancyMetric,TurnContextualPrecisionMetric, 
TurnFaithfulnessMetric,KnowledgeRetentionMetric,RoleAdherenceMetric,ConversationCompletenessMetric
)
from tests.chatbot.eval.conftest import DATASETS,JUDGE_MODEL,THRESHOLDS,CHATBOT_ROLE
from app.chatbot.bot import chat

CONVERSATIONS = json.loads((DATASETS/"chatbot_conversations.json").read_text())

def run_conversation(user_turns: list[str]) -> list[Turn]:
    history,turns = [],[]
    for message in user_turns:
        result = chat(history,message)
        history += [
            {"role":"user","content":message},
            {"role":"assistant","content": result["reply"]},
        ]
        turns += [
                    Turn(role="user",content=message),
                    Turn(role="assistant",content=result["reply"], retrieval_context=result["retrieval_context"]),
        ]
    return turns
    
@pytest.mark.parametrize("convo",CONVERSATIONS,ids=lambda g:g["name"][:40])
def test_rag_multiturn_answer_quality(convo):
    test_case = ConversationalTestCase(
            turns = run_conversation(convo['user_turns']),
            scenario = convo["scenario"],
            expected_outcome = convo["expected_outcome"],
            chatbot_role = CHATBOT_ROLE,
    )

    metrics = [
        TurnFaithfulnessMetric(threshold=THRESHOLDS["turn_faithfulness"],model=JUDGE_MODEL),
        TurnContextualRelevancyMetric(threshold=THRESHOLDS["turn_relvancy"],model=JUDGE_MODEL),
        KnowledgeRetentionMetric(threshold=THRESHOLDS["knowldege_retention"],model=JUDGE_MODEL),
        RoleAdherenceMetric(threshold=THRESHOLDS["role_adherence"],model=JUDGE_MODEL),
        ConversationCompletenessMetric(threshold=THRESHOLDS["conversation_completeness"],model=JUDGE_MODEL),
    ]

    assert_test(test_case,metrics)