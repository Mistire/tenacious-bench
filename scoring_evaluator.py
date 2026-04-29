"""
scoring_evaluator.py — Tenacious-Bench v0.1

Scores an agent output against a task rubric from schema.json.
Every task produces a machine-verifiable score with no human in the loop
for hard constraints. Soft dimensions use an LLM judge call.

Usage:
    python scoring_evaluator.py                  # run all 3 dummy tasks
    python scoring_evaluator.py --task TB-001    # run one task

Environment:
    OPENROUTER_API_KEY — required for soft-dimension LLM judge calls
    JUDGE_MODEL        — optional, defaults to dev-tier model
"""

import json
import os
import re
import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

SCHEMA_PATH = Path(__file__).parent / "schema.json"
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "qwen/qwen3-235b-a22b")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class HardConstraintResult:
    constraint_id: str
    description: str
    passed: bool
    detail: str


@dataclass
class SoftDimensionResult:
    dimension_id: str
    dimension: str
    score: int
    threshold: int
    passed: bool
    judge_reasoning: str = ""


@dataclass
class TaskScore:
    task_id: str
    failure_family: str
    source_mode: str
    hard_constraints: list[HardConstraintResult] = field(default_factory=list)
    soft_dimensions: list[SoftDimensionResult] = field(default_factory=list)
    final_score: float = 0.0
    passed: bool = False
    hard_fail_reason: Optional[str] = None


def _word_count(text: str) -> int:
    return len(text.split())


def _subject_and_body(output: str) -> tuple[str, str]:
    """Split agent output into subject line and body."""
    lines = output.strip().splitlines()
    subject = ""
    body_lines = []
    in_body = False
    for line in lines:
        stripped = line.strip()
        if not in_body and stripped.lower().startswith("subject:"):
            subject = stripped[len("subject:"):].strip()
        elif not in_body and re.match(r"^(re:|fwd:|context:|request:|question:|note on)", stripped, re.I):
            subject = stripped
        else:
            in_body = True
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    if not subject and body_lines:
        # first non-empty line is treated as subject if no explicit label
        subject = body_lines[0].strip()
        body = "\n".join(body_lines[1:]).strip()
    return subject, body


def check_hard_constraints(task: dict, agent_output: str) -> list[HardConstraintResult]:
    results = []
    output_lower = agent_output.lower()
    subject, body = _subject_and_body(agent_output)

    for hc in task["rubric"]["hard_constraints"]:
        hc_id = hc["id"]
        desc = hc["description"]
        check = hc.get("check", "")
        passed = True
        detail = "OK"

        if "fail_patterns" in hc:
            violated = [p for p in hc["fail_patterns"] if p.lower() in output_lower]
            if violated:
                passed = False
                detail = f"Forbidden pattern(s) found: {violated}"

        if passed and "pass_signals" in hc:
            found = [s for s in hc["pass_signals"] if s.lower() in output_lower]
            if not found:
                passed = False
                detail = f"Required signal not found. Expected one of: {hc['pass_signals']}"
            else:
                detail = f"Required signal present: {found}"

        if passed and "len(subject_line) <= 60" in check:
            if len(subject) > 60:
                passed = False
                detail = f"Subject line too long: {len(subject)} chars (max 60). Subject: '{subject}'"
            else:
                detail = f"Subject length OK: {len(subject)} chars"

        if passed and "word_count(body) <= 120" in check:
            wc = _word_count(body)
            if wc > 120:
                passed = False
                detail = f"Body too long: {wc} words (max 120)"
            else:
                detail = f"Body word count OK: {wc} words"

        if passed and "subject does not start with" in check:
            fail_openers = hc.get("fail_patterns", [])
            for opener in fail_openers:
                if subject.lower().startswith(opener.lower()):
                    passed = False
                    detail = f"Subject starts with forbidden opener: '{opener}'"
                    break
            if passed:
                detail = f"Subject opener OK: '{subject[:20]}...'"

        results.append(HardConstraintResult(
            constraint_id=hc_id,
            description=desc,
            passed=passed,
            detail=detail,
        ))

    return results


SKIP_JUDGE = False  # set True via --skip-judge flag for offline testing


def _call_judge(prompt: str) -> tuple[int, str]:
    """Call OpenRouter dev-tier judge. Returns (score 1-5, reasoning)."""
    if SKIP_JUDGE:
        return 5, "skip-judge mode: auto-pass"
    if not REQUESTS_AVAILABLE:
        return 3, "requests library not installed; defaulting to neutral score"
    if not OPENROUTER_API_KEY:
        return 3, "OPENROUTER_API_KEY not set; defaulting to neutral score"

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge for a B2B sales agent benchmark. "
                    "Respond with exactly two lines:\n"
                    "SCORE: <integer 1-5>\n"
                    "REASON: <one sentence>"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 80,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=20)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        score_line = next((l for l in text.splitlines() if l.startswith("SCORE:")), "SCORE: 3")
        reason_line = next((l for l in text.splitlines() if l.startswith("REASON:")), "REASON: N/A")
        score = int(re.search(r"\d", score_line).group())
        reason = reason_line.replace("REASON:", "").strip()
        return score, reason
    except Exception as exc:
        return 3, f"Judge call failed: {exc}"


def score_soft_dimensions(task: dict, agent_output: str) -> list[SoftDimensionResult]:
    results = []
    for sd in task["rubric"]["soft_dimensions"]:
        judge_prompt = (
            f"Agent output to evaluate:\n\"\"\"\n{agent_output}\n\"\"\"\n\n"
            f"{sd['judge_prompt']}"
        )
        score, reasoning = _call_judge(judge_prompt)
        score = max(1, min(5, score))
        passed = score >= sd["threshold"]
        results.append(SoftDimensionResult(
            dimension_id=sd["id"],
            dimension=sd["dimension"],
            score=score,
            threshold=sd["threshold"],
            passed=passed,
            judge_reasoning=reasoning,
        ))
    return results


def score_task(task: dict, agent_output: str) -> TaskScore:
    result = TaskScore(
        task_id=task["task_id"],
        failure_family=task["failure_family"],
        source_mode=task["source_mode"],
    )

    result.hard_constraints = check_hard_constraints(task, agent_output)
    hard_passed = all(hc.passed for hc in result.hard_constraints)

    if not hard_passed:
        failed = [hc for hc in result.hard_constraints if not hc.passed]
        result.hard_fail_reason = "; ".join(f"{hc.constraint_id}: {hc.detail}" for hc in failed)
        result.final_score = 0.0
        result.passed = False
        return result

    result.soft_dimensions = score_soft_dimensions(task, agent_output)
    formula = task["rubric"].get("score_formula", "")

    # Apply scoring formula
    if "1.0 if all hard_constraints pass AND all soft_dimensions >= threshold" in formula:
        result.passed = all(sd.passed for sd in result.soft_dimensions)
        result.final_score = 1.0 if result.passed else (
            sum(1 for sd in result.soft_dimensions if sd.passed) / len(result.soft_dimensions) * 0.5
        )
    elif "SD-002-A >= 4" in formula or "SD-003-A >= 4" in formula:
        key_dim = next((sd for sd in result.soft_dimensions if sd.dimension_id.endswith("-A")), None)
        result.passed = key_dim is not None and key_dim.passed
        result.final_score = 1.0 if result.passed else 0.0
    else:
        result.passed = all(sd.passed for sd in result.soft_dimensions)
        result.final_score = 1.0 if result.passed else 0.0

    return result


def print_result(result: TaskScore) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"\n{'='*60}")
    print(f"Task {result.task_id} | {result.source_mode} | {result.failure_family} | [{status}] score={result.final_score:.2f}")
    print(f"{'='*60}")

    print("Hard constraints:")
    for hc in result.hard_constraints:
        mark = "✓" if hc.passed else "✗"
        print(f"  {mark} {hc.constraint_id}: {hc.detail}")

    if result.hard_fail_reason:
        print(f"  → Hard fail: {result.hard_fail_reason}")

    if result.soft_dimensions:
        print("Soft dimensions:")
        for sd in result.soft_dimensions:
            mark = "✓" if sd.passed else "✗"
            print(f"  {mark} {sd.dimension_id} [{sd.dimension}]: {sd.score}/5 (threshold {sd.threshold})")
            if sd.judge_reasoning:
                print(f"      Judge: {sd.judge_reasoning}")


def load_tasks(task_id: Optional[str] = None) -> list[dict]:
    schema = json.loads(SCHEMA_PATH.read_text())
    tasks = schema["example_tasks"]
    if task_id:
        tasks = [t for t in tasks if t["task_id"] == task_id]
    return tasks


# --- Dummy outputs for self-test ---

DUMMY_OUTPUTS = {
    "TB-001": (
        "Subject: Context: 3 open Python roles at Orrin — a question\n"
        "Elena,\n\n"
        "You have 3 open Python/FastAPI roles since January — is hiring velocity "
        "matching the runway right now?\n\n"
        "If you're hitting a recruiting-capacity wall, we have engineers ready to deploy. "
        "Worth a 15-minute call to see if there's a fit?\n\n"
        "Mistire\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
    ),
    "TB-002": (
        "Subject: Context: restructuring context — delivery continuity question\n"
        "Marcus,\n\n"
        "Noticed Ardent closed a Series B and had a restructuring event in the same quarter. "
        "Companies in that position often prioritize preserving delivery capacity over net-new hiring.\n\n"
        "We work with engineering teams navigating exactly that — keeping output steady through "
        "the transition without adding permanent headcount pressure.\n\n"
        "Worth 15 minutes to see if that's relevant?\n\n"
        "Mistire\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
    ),
    "TB-003": (
        "Subject: Request: ML team sizing — important caveat before we proceed\n"
        "Priya,\n\n"
        "I want to be straightforward: we have 5 ML engineers available now (1 senior, "
        "3 mid-level, 1 junior). An 8-person team would require a phased ramp — "
        "we'd need to confirm expansion capacity and timeline in a scoping call with "
        "our delivery lead before committing.\n\n"
        "Can we get that call on the calendar this week? 30 minutes.\n\n"
        "Mistire\nResearch Partner\nTenacious Intelligence Corporation\ngettenacious.com"
    ),
}


def main():
    parser = argparse.ArgumentParser(description="Tenacious-Bench scoring evaluator")
    parser.add_argument("--task", help="Run a single task by ID (e.g. TB-001)")
    parser.add_argument("--output", help="Agent output string (uses dummy if omitted)")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge calls; auto-pass soft dimensions (for offline testing)")
    args = parser.parse_args()

    global SKIP_JUDGE
    if args.skip_judge:
        SKIP_JUDGE = True

    tasks = load_tasks(args.task)
    if not tasks:
        print(f"No tasks found for id: {args.task}")
        sys.exit(1)

    all_passed = True
    for task in tasks:
        agent_output = args.output if args.output else DUMMY_OUTPUTS.get(task["task_id"], "")
        if not agent_output:
            print(f"No output for {task['task_id']} — skipping")
            continue
        result = score_task(task, agent_output)
        print_result(result)
        if not result.passed:
            all_passed = False

    print(f"\n{'='*60}")
    print(f"Summary: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'='*60}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
