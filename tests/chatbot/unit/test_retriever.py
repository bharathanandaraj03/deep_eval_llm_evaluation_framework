import pytest
from app.chatbot import config
from app.chatbot.retriever import Retriever, Section, load_sections

@pytest.fixture(scope='module')
def retriever():
    return Retriever(load_sections(config.POLICY_DIR))

def test_load_setions_split_on_h2_headings():
    sections = load_sections(config.POLICY_DIR)
    assert len(sections) == 13
    assert all(isinstance(s, Section) for s in sections)

@pytest.mark.parametrize(
    "query, expected_source",
    [
        ("Can I work from home?", "remote_work.md"),
        ("How many vacation days do I get? annual leave", "leave.md"),
        ("What is the hotel limit when travelling?", "expenses.md"),
        ("Is multi-factor authentication required?", "security.md"),
        ("Do I get paid extra for overtime?", "hours.md"),
        ("Can I accept a gift from a vendor?", "conduct.md"),
    ],
)
def test_top_result_comes_from_expected_policy(retriever, query, expected_source):
    results = retriever.search(query, k=3)
    assert results[0].source == expected_source


def test_search_respects_k(retriever):
    assert len(retriever.search("leave policy days", k=2)) <= 2


def test_unrelated_query_returns_nothing(retriever):
    assert retriever.search("weather in Paris tomorrow", k=3) == []


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        Retriever([])