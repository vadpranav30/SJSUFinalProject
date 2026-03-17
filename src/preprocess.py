from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .utils import normalize_skill, unique_preserve_order


_PUNCT_RE = re.compile(r"[^\w\s\-\+\/]")
_WS_RE = re.compile(r"\s+")


def clean_text(
    text: str,
    *,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords: bool = True,
) -> str:
    t = text or ""
    if lowercase:
        t = t.lower()
    if remove_punctuation:
        t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    if remove_stopwords and t:
        tokens = [w for w in t.split(" ") if w and w not in ENGLISH_STOP_WORDS]
        t = " ".join(tokens)
    return t


def extract_skills(text: str, skill_lexicon: Sequence[str]) -> List[str]:
    t = (text or "").lower()
    found: List[str] = []
    for raw in skill_lexicon:
        skill = normalize_skill(raw)
        if not skill:
            continue
        if " " in skill:
            if skill in t:
                found.append(skill)
        else:
            if re.search(rf"\b{re.escape(skill)}\b", t):
                found.append(skill)
    return unique_preserve_order(found)


def preprocess_corpus(
    texts: Sequence[str],
    preprocessing_cfg: Dict,
) -> List[str]:
    return [
        clean_text(
            t,
            lowercase=bool(preprocessing_cfg.get("lowercase", True)),
            remove_punctuation=bool(preprocessing_cfg.get("remove_punctuation", True)),
            remove_stopwords=bool(preprocessing_cfg.get("remove_stopwords", True)),
        )
        for t in texts
    ]

