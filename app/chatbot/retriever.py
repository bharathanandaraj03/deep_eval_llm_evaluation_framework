import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "do", "does", "for", "from",
    "get", "how", "i", "if", "in", "is", "it", "many", "md", "me", "my", "of", "on",
    "or", "s", "that", "the", "there", "this", "to", "what", "when", "will", "with",
    "you", "your",
}


def _stem(token: str) -> str:
    # Plural folding so "gift" matches "gifts" and "day" matches "days".
    return token[:-1] if len(token) > 3 and token.endswith("s") else token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Section:
    source: str
    heading: str
    text: str


def load_sections(policy_dir: Path) -> list[Section]:
    sections = []
    for path in sorted(policy_dir.glob("*.md")):
        chunks = re.split(r"(?m)^## ", path.read_text(encoding="utf-8"))[1:]
        for chunk in chunks:
            heading, _, body = chunk.partition("\n")
            heading = heading.strip()
            sections.append(Section(path.name, heading, f"{heading}\n{body.strip()}"))
    return sections


class Retriever:
    def __init__(self, sections: list[Section]):
        if not sections:
            raise ValueError("No policy sections to index")
        self.sections = sections
        self._bm25 = BM25Okapi([tokenize(f"{s.source} {s.text}") for s in sections])

    def search(self, query: str, k: int = 3) -> list[Section]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(scores, self.sections), key=lambda pair: pair[0], reverse=True)
        return [section for score, section in ranked[:k] if score > 0]