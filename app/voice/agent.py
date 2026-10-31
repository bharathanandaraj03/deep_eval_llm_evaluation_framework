import time

from openai import OpenAI
from app.chatbot.bot import SYSTEM_PROMPT, chat
from app.voice import config

VOICE_SYSTEM_PROMPT = SYSTEM_PROMPT + """

Your reply is spoken aloud by a voice assistant, so this overrides the length rule above: answer in at most
2 short sentences of plain speech. No markdown, bullets, headings or file names. Say the policy name in words,
for example "the Remote Work policy". Answer only what was asked."""

_client: OpenAI | None = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def transcribe(audio:bytes,client=None) -> str:
    client = client or _get_client()
    return client.audio.transcriptions.create(model=config.STT_MODEL, file=("turn.wav",audio)).text

def synthesize(text: str, client= None) -> bytes:
    client = client or _get_client()
    return client.audio.speech.create(
        model=config.TTS_MODEL,voice=config.TTS_VOICE,input=text, response_format="wav").read()

def voice_turn(history: list[dict],audio: bytes) -> dict:
    t0 = time.perf_counter()
    transcipt = transcribe(audio)
    t1 = time.perf_counter()
    result = chat(history, transcipt, system_prompt=VOICE_SYSTEM_PROMPT)
    t2 = time.perf_counter()
    reply_audio = synthesize(result["reply"])
    t3= time.perf_counter()
    return{

        **result,
        "transcript": transcipt,
        "reply_audio": reply_audio,
        "latency":{"stt": t1-t0, "llm": t2 - t1 , "tts": t3 - t2, "total": t3 - t0 },
    }
