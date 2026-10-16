import os
from pathlib import Path

MODEL = os.getenv("CHATBOT_MODEL", "claude-opus-5-5")
EFFORT = os.getenv("CHATBOT_EFFORT", "low")
TOP_K = int(os.getenv("CHATBOT_TOP_K", "3"))
MAX_TOKENS = int(os.getenv("CHATBOT_MAX_TOKENS", "16000"))
POLICY_DIR = Path(
    os.getenv("CHATBOT_POLICY_DIR", Path(__file__).resolve().parents[2] / "data" / "policies")
)