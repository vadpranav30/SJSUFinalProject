from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.matcher import rank_resumes
from src.utils import load_config, load_job_description, load_resumes_csv, resolve_data_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simple AI Resume Matcher")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml (optional)")
    p.add_argument("--job", type=str, default=None, help="Job description as a string (optional)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg.get("data", {})
    resumes_path = resolve_data_path(str(data_cfg.get("resume_dataset_path", "data/resumes.csv")))
    job_path = resolve_data_path(str(data_cfg.get("job_description_path", "data/job_description.txt")))
    resume_text_col = str(data_cfg.get("text_column_resume", "resume_text"))

    resumes_df = load_resumes_csv(resumes_path, text_column=resume_text_col, id_column="resume_id")

    job_description = args.job
    if job_description is None:
        job_description = load_job_description(job_path)

    results = rank_resumes(resumes_df=resumes_df, job_description=job_description, cfg=cfg, id_column="resume_id")

    top_k = int(cfg.get("matching", {}).get("top_k", 5))
    show_scores = bool(cfg.get("output", {}).get("show_scores", True))
    show_ids = bool(cfg.get("output", {}).get("show_resume_ids", True))
    highlight = bool(cfg.get("output", {}).get("highlight_keywords", False))

    print(f"Top {top_k} resume matches\n")
    for i, r in enumerate(results, start=1):
        parts = [f"{i}."]
        if show_ids:
            parts.append(f"resume_id={r.resume_id}")
        if show_scores:
            parts.append(f"score={r.score:.4f}")
        print(" ".join(parts))
        if highlight:
            print(f"  matched_skills={r.matched_skills or []}")


if __name__ == "__main__":
    main()

