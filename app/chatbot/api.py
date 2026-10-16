import logging
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.chatbot.bot import chat

logger = logging.getLogger(__name__)
app = FastAPI(title="HR Policy Chatbot")


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    history: list[Message] = []
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    retrieval_context: list[str]


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    try:
        result = chat([m.model_dump() for m in request.history], request.message)
    except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError) as e:
        logger.warning("chat upstream unavailable error=%s", type(e).__name__)
        raise HTTPException(status_code=503, detail="Chat service temporarily unavailable") from e
    except anthropic.APIStatusError as e:
        logger.error("chat upstream error status=%s request_id=%s", e.status_code, e.request_id)
        raise HTTPException(status_code=502, detail="Chat service error") from e
    return ChatResponse(**result)