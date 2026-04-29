"""
judge_filter.py — LLM-as-a-judge quality filter for Tenacious-Bench

Scores every candidate task on three dimensions before it enters the dataset:
  1. input_coherence      — is the scenario believable and internally consistent?
  2. ground_truth_verifiability — can a script determine pass/fail without ambiguity?
  3. rubric_clarity       — are the scoring instructions unambiguous?

Inclusion threshold: >= 4/5 on ALL three dimensions.

Model rotation policy (to avoid preference leakage, Li et al. 2025):
  Generator model  → Judge model must be a DIFFERENT family.
  Default pairing: Qwen3 generator → DeepSeek V3 judge.
  If generator is DeepSeek → judge is Qwen3.

Usage:
    from judge_filter import filter_tasks
    kept, rejected = filter_tasks(tasks, generator_model="qwen/qwen3-235b-a22b")
"""

import json
import os
import re
import time
from typing import Optional
import requests

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
THRESHOLD = 4

# Model rotation: generator family → judge model
JUDGE_ROTATION = {
    "qwen": "deepseek/deepseek-chat-v3-0324",
    "deepseek": "qwen/qwen3-235b-a22b",
    "anthropic": "qwen/qwen3-235b-a22b",
    "openai": "qwen/qwen3-235b-a22b",
    "default": "deepseek/deepseek-chat-v3-0324",
}


def _get_judge_model(generator_model: str) -> str:
    for family, judge in JUDGE_ROTATION.items():
        if family in generator_model.lower():
            return judge
    return JUDGE_ROTATION["default"]


_JUDGE_SYSTEM = """\
You are a quality judge for a B2B sales agent evaluation benchmark.
Score the given task on THREE dimensions, each 1-5:
1. input_coherence: Is the scenario realistic and internally consistent?
2. ground_truth_verifiability: Can a script determine pass/fail mechanically without ambiguity?
3. rubric_clarity: Are the scoring instructions unambiguous to a third-party evaluator?

Respond with ONLY this JSON (no markdown, no extra text):
{"input_coherence": <1-5>, "ground_truth_verifiability": <1-5>, "rubric_clarity": <1-5>, "reasoning": "<one sentence>"}
"""


def _score_task(task: dict, judge_model: str) -> dict:
    """Call the judge and return dimension scores."""
    hcs = task.get("rubric", {}).get("hard_constraints", [])
    task_summary = {
        "task_id": task.get("task_id"),
        "failure_family": task.get("failure_family"),
        "outreach_instruction": task.get("input", {}).get("outreach_instruction", ""),
        "hard_constraints": [
            {
                "description": hc.get("description", ""),
                "fail_patterns": hc.get("fail_patterns", []),
                "pass_signals": hc.get("pass_signals", []),
                "check": hc.get("check", ""),
            }
            for hc in hcs
        ],
        "soft_dimension": (
            task.get("rubric", {}).get("soft_dimensions", [{}])[0].get("description", "")
            if task.get("rubric", {}).get("soft_dimensions") else ""
        ),
    }
    note = (
        "NOTE: A task is ground-truth-verifiable if its hard_constraints have explicit "
        "fail_patterns (substrings to check in output) or pass_signals. "
        "If fail_patterns or pass_signals are concrete phrases, GV should be >= 4."
    )
    prompt = f"{note}\n\nTask to evaluate:\n{json.dumps(task_summary, indent=2)}"

    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 150,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        # strip markdown code fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        scores = json.loads(raw)
        return scores
    except Exception as exc:
        return {
            "input_coherence": 3,
            "ground_truth_verifiability": 3,
            "rubric_clarity": 3,
            "reasoning": f"Judge call failed: {exc}",
        }


def filter_tasks(
    tasks: list[dict],
    generator_model: str = "qwen/qwen3-235b-a22b",
    threshold: int = THRESHOLD,
    delay: float = 0.3,
    verbose: bool = True,
) -> tuple[list[dict], list[dict]]:
    """
    Filter tasks through the judge.
    Returns (kept, rejected) where kept tasks have all dimensions >= threshold.
    Adds judge_scores and judge_passed fields to task metadata.
    """
    judge_model = _get_judge_model(generator_model)
    if verbose:
        print(f"Judge model: {judge_model} (generator: {generator_model})")

    kept, rejected = [], []
    for i, task in enumerate(tasks):
        scores = _score_task(task, judge_model)
        passed = (
            scores.get("input_coherence", 0) >= threshold
            and scores.get("ground_truth_verifiability", 0) >= threshold
            and scores.get("rubric_clarity", 0) >= threshold
        )
        task.setdefault("metadata", {}).update({
            "judge_scores": {
                "input_coherence": scores.get("input_coherence", 3),
                "ground_truth_verifiability": scores.get("ground_truth_verifiability", 3),
                "rubric_clarity": scores.get("rubric_clarity", 3),
            },
            "judge_reasoning": scores.get("reasoning", ""),
            "judge_passed": passed,
            "judge_model": judge_model,
            "generator_model": generator_model,
        })
        if passed:
            kept.append(task)
        else:
            rejected.append(task)
        if verbose:
            status = "✓" if passed else "✗"
            ic = scores.get("input_coherence", "?")
            gv = scores.get("ground_truth_verifiability", "?")
            rc = scores.get("rubric_clarity", "?")
            print(f"  [{i+1:3d}] {status} {task.get('task_id','?')} "
                  f"IC={ic} GV={gv} RC={rc}  {scores.get('reasoning','')[:60]}")
        if delay and i < len(tasks) - 1:
            time.sleep(delay)

    if verbose:
        print(f"\nFilter result: {len(kept)} kept / {len(rejected)} rejected "
              f"({len(kept)/len(tasks)*100:.0f}% pass rate)")
    return kept, rejected


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python judge_filter.py <input.jsonl> [output.jsonl]")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.replace(".jsonl", "_filtered.jsonl")
    tasks = [json.loads(l) for l in open(in_path) if l.strip()]
    kept, rejected = filter_tasks(tasks)
    with open(out_path, "w") as f:
        for t in kept:
            f.write(json.dumps(t) + "\n")
    rej_path = out_path.replace(".jsonl", "_rejected.jsonl")
    with open(rej_path, "w") as f:
        for t in rejected:
            f.write(json.dumps(t) + "\n")
    print(f"Kept → {out_path}")
    print(f"Rejected → {rej_path}")
