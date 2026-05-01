"""
rescue_rejected_pairs.py

Re-evaluates the 23 pairs in preference_pairs_rejected.jsonl after stripping
compliance footnotes that DeepSeek appended. No API calls needed — the chosen
text is already there, just being mis-scored by the quality checker.

Run:
    python3 rescue_rejected_pairs.py
"""

import json
from pathlib import Path

from build_preference_pairs import _judge_quality, _strip_compliance_notes

REJECTED_PATH = Path(__file__).parent.parent / "training_data" / "preference_pairs_rejected.jsonl"
KEPT_PATH     = Path(__file__).parent.parent / "training_data" / "preference_pairs.jsonl"
TRAIN_PATH    = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"

tasks_by_id = {
    json.loads(l)["task_id"]: json.loads(l)
    for l in open(TRAIN_PATH) if l.strip()
}

rescued, still_bad = [], []

pairs = [json.loads(l) for l in open(REJECTED_PATH) if l.strip()]
print(f"Re-evaluating {len(pairs)} rejected pairs...\n")

for pair in pairs:
    tid  = pair["task_id"]
    task = tasks_by_id.get(tid)
    if not task:
        print(f"  SKIP {tid} — task not found in train partition")
        still_bad.append(pair)
        continue

    if "chosen" not in pair:
        print(f"  SKIP {tid} — no chosen output (was an API error)")
        still_bad.append(pair)
        continue

    cleaned_chosen = _strip_compliance_notes(pair["chosen"])
    new_score = _judge_quality(cleaned_chosen, task)
    pair["chosen"] = cleaned_chosen
    pair["quality_score"] = new_score

    if new_score >= 3:
        rescued.append(pair)
        print(f"  RESCUED  {tid} | score {new_score} | {pair['failure_family']}")
    else:
        still_bad.append(pair)
        print(f"  STILL BAD {tid} | score {new_score} | {pair['failure_family']}")

# Append rescued pairs to the main file
if rescued:
    with open(KEPT_PATH, "a") as f:
        for p in rescued:
            f.write(json.dumps(p) + "\n")

# Overwrite rejected file with only truly bad ones
with open(REJECTED_PATH, "w") as f:
    for p in still_bad:
        f.write(json.dumps(p) + "\n")

print(f"\nRescued: {len(rescued)} pairs → appended to preference_pairs.jsonl")
print(f"Still rejected: {len(still_bad)} pairs → preference_pairs_rejected.jsonl")
print(f"New total pairs: {sum(1 for _ in open(KEPT_PATH))}")
