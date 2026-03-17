from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else (project_root() / "config.yaml")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_path(path_from_config: str) -> Path:
    p = Path(path_from_config)
    if p.is_absolute():
        return p
    return project_root() / p


def load_resumes_csv(path: Path, text_column: str, id_column: str = "resume_id") -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if id_column not in df.columns:
        df[id_column] = [str(i) for i in range(len(df))]
    if text_column not in df.columns:
        exclude_cols = {
            'job_position_name', 'educationaL_requirements', 
            'experiencere_requirement', 'age_requirement', 
            'responsibilities.1', 'skills_required', 'matched_score'
        }
        resume_cols = [c for c in df.columns if c not in exclude_cols]
        def assemble_text(row):
            return " ".join([str(x) for x in row[resume_cols].values if pd.notna(x)])
        df[text_column] = df.apply(assemble_text, axis=1)

    df = df[[id_column, text_column]].copy()
    df[text_column] = df[text_column].fillna("").astype(str)
    df[id_column] = df[id_column].astype(str)
    return df


def load_job_description(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def default_skill_lexicon() -> List[str]:
    return [
        "python",
        "pandas",
        "numpy",
        "scikit-learn",
        "sklearn",
        "pytorch",
        "tensorflow",
        "transformers",
        "sentence-transformers",
        "nlp",
        "information retrieval",
        "semantic search",
        "tf-idf",
        "cosine similarity",
        "sql",
        "docker",
        "kubernetes",
        "mlops",
        "airflow",
        "spark",
        "aws",
        "vector search",
        "bert",
        "evaluation",
        "ab testing",
        "a/b testing",
        "feature engineering",
        "model monitoring",
        "etl",
    ]


def normalize_skill(skill: str) -> str:
    s = skill.strip().lower()
    aliases = {
        "sklearn": "scikit-learn",
        "tfidf": "tf-idf",
        "a/b testing": "ab testing",
        "sentence transformers": "sentence-transformers",
    }
    return aliases.get(s, s)


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

