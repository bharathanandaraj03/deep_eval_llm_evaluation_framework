from types import SimpleNamespace
from unittest.mock import MagicMock

from app.chatbot import bot, config
from app.chatbot.retriever import Retriever, load_sections

RETRIEVER = Retriever(load_sections(config.POLICY_DIR))


def fake_client(text="Up to 2 days per week.", stop_reason="end_turn"):
    client = MagicMock()
    client.beta.messages.create.return_value = SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text=text)],
    )
    return client


def test_chat_returns_reply_sources_and_context():
    client = fake_client()
    result = bot.chat([], "Can I work remotely?", client=client, retriever=RETRIEVER)

    assert result["reply"] == "Up to 2 days per week."
    assert "remote_work.md" in result["sources"]
    assert any("2 days per week" in c for c in result["retrieval_context"])


def test_chat_sends_policies_history_and_fallbacks():
    client = fake_client()
    history = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello, how can I help?"},
    ]
    bot.chat(history, "Can I work remotely?", client=client, retriever=RETRIEVER)

    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["model"] == config.MODEL
    assert kwargs["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in kwargs["betas"]
    assert kwargs["messages"][:2] == history
    assert "<policies>" in kwargs["messages"][-1]["content"]


def test_chat_with_no_matching_policy_still_calls_claude():
    client = fake_client(text=bot.UNKNOWN_REPLY)
    result = bot.chat([], "weather in Paris tomorrow", client=client, retriever=RETRIEVER)

    client.beta.messages.create.assert_called_once()
    assert result["retrieval_context"] == []


def test_refusal_returns_decline_message():
    client = fake_client(text="", stop_reason="refusal")
    result = bot.chat([], "Can I work remotely?", client=client, retriever=RETRIEVER)
    assert result["reply"] == bot.DECLINE_REPLY
