from unittest.mock import patch

import anthropic
import httpx2
from fastapi.testclient import TestClient

from app.chatbot.api import app

client = TestClient(app)
RESULT = {"reply": "Up to 2 days.", "sources": ["remote_work.md"], "retrieval_context": ["..."]}


def test_chat_returns_bot_result():
    with patch("app.chatbot.api.chat", return_value=RESULT) as mock_chat:
        response = client.post(
            "/chat",
            json={"history": [{"role": "user", "content": "Hi"}], "message": "Remote work?"},
        )
    assert response.status_code == 200
    assert response.json() == RESULT
    mock_chat.assert_called_once_with([{"role": "user", "content": "Hi"}], "Remote work?")


def test_empty_message_rejected():
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_invalid_role_rejected():
    body = {"history": [{"role": "system", "content": "x"}], "message": "Hi"}
    assert client.post("/chat", json=body).status_code == 422


def test_upstream_unavailable_returns_503():
    error = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    with patch("app.chatbot.api.chat", side_effect=error):
        response = client.post("/chat", json={"message": "Remote work?"})
    assert response.status_code == 503