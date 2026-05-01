"""
rerun_bad_pairs.py

Re-runs DeepSeek rewrite on the remaining bad pairs with a stricter prompt
that explicitly forbids compliance footnotes.

Run:
    export OPENROUTER_API_KEY=...
    python3 rerun_bad_pairs.py
"""

import json
import os
import time
from pathlib import Path

import requests

from build_preference_pairs import (
    _build_user_prompt,
    _judge_quality,
    _strip_compliance_notes,
    SYSTEM_PROMPT,
)

REJECTED_PATH = Path(__file__).parent.parent / "training_data" / "preference_pairs_rejected.jsonl"
KEPT_PATH     = Path(__file__).parent.parent / "training_data" / "preference_pairs.jsonl"
TRAIN_PATH    = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"
REWRITE_MODEL      = "deepseek/deepseek-chat-v3-0324"

# Stricter rewrite system prompt — no footnotes allowed
STRICT_REWRITE_SYSTEM = (
    SYSTEM_PROMPT + "\n\n"
    "TASK: Rewrite the failing output below to be fully policy-compliant.\n\n"
    "STRICT OUTPUT FORMAT — follow exactly:\n"
    "  Line 1: Subject: <your subject line>\n"
    "  Line 2: blank\n"
    "  Lines 3+: email body (≤120 words)\n"
    "  Final line: your name + Tenacious Intelligence Corporation\n\n"
    "DO NOT include any of the following after the email:\n"
    "  - Compliance notes\n"
    "  - Rationale sections\n"
    "  - Word counts\n"
    "  - Hard constraint checklists\n"
    "  - Markdown dividers (---)\n"
    "  - Asterisk-wrapped annotations\n"
    "Output ONLY the email. Nothing else."
)


def _call_llm(system: str, user: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": REWRITE_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": 350,
        "temperature": 0.2,
    }
    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=45)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def main():
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Set OPENROUTER_API_KEY first")

    tasks_by_id = {
        json.loads(l)["task_id"]: json.loads(l)
        for l in open(TRAIN_PATH) if l.strip()
    }

    pairs = [json.loads(l) for l in open(REJECTED_PATH) if l.strip()]
    print(f"Re-running DeepSeek on {len(pairs)} bad pairs...\n")

    rescued, still_bad = [], []

    for i, pair in enumerate(pairs):
        tid  = pair.get("task_id", "?")
        task = tasks_by_id.get(tid)

        if not task:
            print(f"  [{i+1}] SKIP {tid} — not in train partition")
            still_bad.append(pair)
            continue

        rubric_summary = json.dumps({
            "hard_constraints": [
                {
                    "id": hc["id"],
                    "description": hc["description"],
                    "fail_patterns": hc.get("fail_patterns", []),
                    "pass_signals": hc.get("pass_signals", []),
                }
                for hc in task["rubric"]["hard_constraints"]
            ]
        }, indent=2)

        user_prompt = (
            f"Task rubric (your email MUST satisfy every hard constraint):\n"
            f"{rubric_summary}\n\n"
            f"Input context:\n{_build_user_prompt(task)}\n\n"
            f"Write a fully compliant outreach email. "
            f"Start with 'Subject:' on line 1. Email body only — no notes."
        )

        try:
            raw = _call_llm(STRICT_REWRITE_SYSTEM, user_prompt)
            chosen = _strip_compliance_notes(raw)
            score  = _judge_quality(chosen, task)

            pair["chosen"]        = chosen
            pair["chosen_source"] = "deepseek_rewrite_v2"
            pair["quality_score"] = score

            if score >= 3:
                rescued.append(pair)
                print(f"  [{i+1}] RESCUED  {tid} | score={score} | {pair.get('failure_family','?')}")
            else:
                still_bad.append(pair)
                print(f"  [{i+1}] STILL BAD {tid} | score={score} | {pair.get('failure_family','?')}")
                print(f"         chosen preview: {chosen[:100].replace(chr(10),' ')}")

        except Exception as e:
            print(f"  [{i+1}] ERROR {tid} — {e}")
            still_bad.append(pair)

        if i < len(pairs) - 1:
            time.sleep(0.5)

    # Append rescued to main file
    if rescued:
        with open(KEPT_PATH, "a") as f:
            for p in rescued:
                f.write(json.dumps(p) + "\n")

    # Overwrite rejected with only truly unrescuable pairs
    with open(REJECTED_PATH, "w") as f:
        for p in still_bad:
            f.write(json.dumps(p) + "\n")

    total = sum(1 for _ in open(KEPT_PATH))
    print(f"\nRescued: {len(rescued)} | Still bad: {len(still_bad)}")
    print(f"Total preference pairs now: {total}")
    cost = len(pairs) * 0.0008
    print(f"Estimated cost: ~${cost:.3f}")


if __name__ == "__main__":
    main()
