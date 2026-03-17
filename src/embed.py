from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class EmbeddingResult:
    method: str
    resume_embeddings: Any
    job_embedding: Any
    vectorizer: Optional[TfidfVectorizer] = None


def embed_texts(
    *,
    resumes: Sequence[str],
    job_description: str,
    embedding_cfg: Dict,
    system_cfg: Dict,
) -> EmbeddingResult:
    method = str(embedding_cfg.get("method", "tfidf")).strip().lower()

    def _tfidf() -> EmbeddingResult:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        X = vectorizer.fit_transform(list(resumes))
        q = vectorizer.transform([job_description])
        return EmbeddingResult(method="tfidf", resume_embeddings=X, job_embedding=q, vectorizer=vectorizer)

    if method == "tfidf":
        return _tfidf()

    if method == "sentence_transformer":
        try:
            from sentence_transformers import SentenceTransformer

            model_name = str(embedding_cfg.get("model_name", "all-MiniLM-L6-v2"))
            device = str(system_cfg.get("device", "cpu"))
            model = SentenceTransformer(model_name, device=device)
            resume_emb = model.encode(
                list(resumes),
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            job_emb = model.encode(
                [job_description],
                batch_size=1,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return EmbeddingResult(
                method="sentence_transformer",
                resume_embeddings=np.asarray(resume_emb),
                job_embedding=np.asarray(job_emb),
                vectorizer=None,
            )
        except Exception:
            return _tfidf()

    raise ValueError(f"Unknown embedding method: {method}. Use 'tfidf' or 'sentence_transformer'.")

