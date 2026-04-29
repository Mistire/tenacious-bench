"""
contamination_check.py — Tenacious-Bench v0.1

Three contamination checks before any task enters the held-out partition:
  1. N-gram overlap   — held-out input shares < 8-gram overlap with any training input
  2. Embedding cosine — cosine similarity < 0.85 between held-out and any training task
  3. Time-shift       — all tasks with public-data references document source + date window

Output: contamination_report.json summarising all checks.

Usage:
    python3 contamination_check.py <train.jsonl> <held_out.jsonl>
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path


def _ngrams(text: str, n: int) -> set[tuple]:
    tokens = text.lower().split()
    return set(zip(*[tokens[i:] for i in range(n)]))


def _task_input_text(task: dict) -> str:
    """Concatenate all meaningful input fields into one string for n-gram checking."""
    brief = task.get("input", {}).get("hiring_signal_brief", {})
    parts = [
        task.get("input", {}).get("outreach_instruction", ""),
        brief.get("company", ""),
        " ".join(brief.get("honesty_flags", [])),
        str(brief.get("open_roles_today", "")),
        str(brief.get("segment_confidence", "")),
    ]
    return " ".join(str(p) for p in parts if p)


def ngram_check(train_tasks: list[dict], held_out_tasks: list[dict],
                n: int = 8) -> dict:
    """
    For each held-out task, check that no training task shares >= n-gram overlap.
    Returns dict mapping held-out task_id → {max_overlap_count, violates, matched_train_id}.
    """
    train_ngrams = {
        t["task_id"]: _ngrams(_task_input_text(t), n) for t in train_tasks
    }
    results = {}
    violations = 0
    for ho_task in held_out_tasks:
        ho_ng = _ngrams(_task_input_text(ho_task), n)
        max_overlap = 0
        matched = None
        for tr_id, tr_ng in train_ngrams.items():
            overlap = len(ho_ng & tr_ng)
            if overlap > max_overlap:
                max_overlap = overlap
                matched = tr_id
        violates = max_overlap >= n
        if violates:
            violations += 1
        results[ho_task["task_id"]] = {
            "max_ngram_overlap": max_overlap,
            "violates": violates,
            "matched_train_id": matched,
        }
    return {"n": n, "violations": violations, "per_task": results}


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return _dot(a, b) / (na * nb)


def _simple_bow_vector(text: str, vocab: dict) -> list[float]:
    """Bag-of-words vector using a shared vocabulary dict."""
    tokens = text.lower().split()
    vec = [0.0] * len(vocab)
    for tok in tokens:
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    return vec


def embedding_check(train_tasks: list[dict], held_out_tasks: list[dict],
                    threshold: float = 0.85) -> dict:
    """
    Approximate embedding similarity via TF-IDF bag-of-words.
    (Production would use a cheap embedding model; this is the offline fallback.)
    Returns dict mapping held-out task_id → {max_cosine, violates, matched_train_id}.
    """
    all_texts = [_task_input_text(t) for t in train_tasks + held_out_tasks]
    # Build shared vocabulary from all task texts
    all_tokens: set[str] = set()
    for text in all_texts:
        all_tokens.update(text.lower().split())
    vocab = {tok: i for i, tok in enumerate(sorted(all_tokens))}

    train_vecs = [(t["task_id"], _simple_bow_vector(_task_input_text(t), vocab))
                  for t in train_tasks]

    results = {}
    violations = 0
    for ho_task in held_out_tasks:
        ho_vec = _simple_bow_vector(_task_input_text(ho_task), vocab)
        max_cos = 0.0
        matched = None
        for tr_id, tr_vec in train_vecs:
            cos = _cosine(ho_vec, tr_vec)
            if cos > max_cos:
                max_cos = cos
                matched = tr_id
        violates = max_cos >= threshold
        if violates:
            violations += 1
        results[ho_task["task_id"]] = {
            "max_cosine_similarity": round(max_cos, 4),
            "violates": violates,
            "matched_train_id": matched,
        }
    return {"threshold": threshold, "violations": violations, "per_task": results,
            "note": "Similarity computed via bag-of-words approximation; production uses sentence-transformers"}


def time_shift_check(tasks: list[dict]) -> dict:
    """
    Verify that any task referencing public data documents the signal window.
    A task 'references public data' if it has a layoff_event, funding_event, or competitor_gap_brief.
    """
    results = {}
    violations = 0
    for task in tasks:
        brief = task.get("input", {}).get("hiring_signal_brief", {})
        gap = task.get("input", {}).get("competitor_gap_brief")
        issues = []

        layoff = brief.get("layoff_event", {})
        if layoff.get("detected") and not layoff.get("date") and not layoff.get("days_ago"):
            issues.append("layoff_event detected but no date or days_ago documented")

        funding = brief.get("funding_event", {})
        if funding.get("detected") and not funding.get("days_ago"):
            issues.append("funding_event detected but no days_ago documented")

        if gap and not gap.get("generated_at"):
            issues.append("competitor_gap_brief present but no generated_at timestamp")

        has_violation = len(issues) > 0
        if has_violation:
            violations += 1
        results[task["task_id"]] = {
            "has_public_data_ref": bool(layoff.get("detected") or funding.get("detected") or gap),
            "issues": issues,
            "violates": has_violation,
        }
    return {"violations": violations, "per_task": results}


def run_all_checks(train_path: Path, held_out_path: Path,
                   out_path: Path | None = None) -> dict:
    train_tasks = [json.loads(l) for l in open(train_path) if l.strip()]
    held_out_tasks = [json.loads(l) for l in open(held_out_path) if l.strip()]

    print(f"Running contamination checks: {len(train_tasks)} train / {len(held_out_tasks)} held-out")

    ng = ngram_check(train_tasks, held_out_tasks)
    em = embedding_check(train_tasks, held_out_tasks)
    ts_train = time_shift_check(train_tasks)
    ts_held = time_shift_check(held_out_tasks)

    report = {
        "train_count": len(train_tasks),
        "held_out_count": len(held_out_tasks),
        "ngram_check": {
            "n": ng["n"],
            "violations": ng["violations"],
            "pass": ng["violations"] == 0,
            "per_task": ng["per_task"],
        },
        "embedding_check": {
            "threshold": em["threshold"],
            "violations": em["violations"],
            "pass": em["violations"] == 0,
            "note": em["note"],
            "per_task": em["per_task"],
        },
        "time_shift_check": {
            "train_violations": ts_train["violations"],
            "held_out_violations": ts_held["violations"],
            "pass": ts_train["violations"] == 0 and ts_held["violations"] == 0,
            "train_per_task": ts_train["per_task"],
            "held_out_per_task": ts_held["per_task"],
        },
        "overall_pass": (
            ng["violations"] == 0
            and em["violations"] == 0
            and ts_train["violations"] == 0
            and ts_held["violations"] == 0
        ),
    }

    print(f"  N-gram check:    {'PASS' if report['ngram_check']['pass'] else 'FAIL'} "
          f"({ng['violations']} violations)")
    print(f"  Embedding check: {'PASS' if report['embedding_check']['pass'] else 'FAIL'} "
          f"({em['violations']} violations)")
    print(f"  Time-shift check:{'PASS' if report['time_shift_check']['pass'] else 'FAIL'} "
          f"(train={ts_train['violations']}, held_out={ts_held['violations']} violations)")
    print(f"  Overall: {'PASS ✓' if report['overall_pass'] else 'FAIL ✗'}")

    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Report → {out_path}")

    return report


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 contamination_check.py <train.jsonl> <held_out.jsonl> [report.json]")
        sys.exit(1)
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    run_all_checks(Path(sys.argv[1]), Path(sys.argv[2]), out)
