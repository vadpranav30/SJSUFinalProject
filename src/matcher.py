from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import extract_skills, preprocess_corpus
from .embed import embed_texts
from .utils import default_skill_lexicon


@dataclass
class MatchResult:
    resume_id: str
    score: float
    matched_skills: Optional[List[str]] = None


def rank_resumes(
    *,
    resumes_df: pd.DataFrame,
    job_description: str,
    cfg: Dict,
    id_column: str = "resume_id",
) -> List[MatchResult]:
    data_cfg = cfg.get("data", {})
    preprocessing_cfg = cfg.get("preprocessing", {})
    embedding_cfg = cfg.get("embedding", {})
    matching_cfg = cfg.get("matching", {})
    output_cfg = cfg.get("output", {})
    system_cfg = cfg.get("system", {})

    resume_text_col = str(data_cfg.get("text_column_resume", "resume_text"))
    top_k = int(matching_cfg.get("top_k", 5))

    resumes_raw = resumes_df[resume_text_col].fillna("").astype(str).tolist()
    job_raw = str(job_description or "")

    resumes_clean = preprocess_corpus(resumes_raw, preprocessing_cfg)
    job_clean = preprocess_corpus([job_raw], preprocessing_cfg)[0]

    emb = embed_texts(
        resumes=resumes_clean,
        job_description=job_clean,
        embedding_cfg=embedding_cfg,
        system_cfg=system_cfg,
    )

    sims = cosine_similarity(emb.resume_embeddings, emb.job_embedding).reshape(-1)
    order = np.argsort(-sims)

    skill_lex = default_skill_lexicon()
    job_skills = set(extract_skills(job_raw, skill_lex))

    want_skills = bool(output_cfg.get("highlight_keywords", False))
    results: List[MatchResult] = []
    for idx in order[: max(top_k, 0)]:
        rid = str(resumes_df.iloc[int(idx)][id_column])
        score = float(sims[int(idx)])
        matched = None
        if want_skills:
            resume_skills = set(extract_skills(resumes_raw[int(idx)], skill_lex))
            matched = sorted(job_skills.intersection(resume_skills))
        results.append(MatchResult(resume_id=rid, score=score, matched_skills=matched))

    return results

