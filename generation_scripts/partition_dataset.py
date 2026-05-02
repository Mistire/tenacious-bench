"""
partition_dataset.py — Tenacious-Bench v0.1

Merges all task sources, assigns final task IDs, shuffles, and partitions:
  train/     50% — for SFT/DPO training data preparation
  dev/       30% — public, for evaluation during development
  held_out/  20% — sealed, gitignored, for final ablation scoring

Then runs contamination_check.py on the partition.

Output:
  tenacious_bench_v0.1/train/tasks.jsonl
  tenacious_bench_v0.1/dev/tasks.jsonl
  tenacious_bench_v0.1/held_out/tasks.jsonl   (gitignored)
  tenacious_bench_v0.1/contamination_report.json
  tenacious_bench_v0.1/dataset_manifest.json

Usage:
    python3 partition_dataset.py
"""

import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).parent.parent / "tenacious_bench_v0.1"
SOURCES = [
    BASE / "programmatic_tasks.jsonl",
    BASE / "programmatic_expansion_tasks.jsonl",
    BASE / "trace_derived_tasks.jsonl",
    BASE / "synthesis_tasks_filtered.jsonl",
    BASE / "adversarial_tasks.jsonl",
]
TRAIN_RATIO = 0.50
DEV_RATIO   = 0.30
HELD_RATIO  = 0.20
SEED = 42


def load_all_tasks() -> list[dict]:
    tasks = []
    for src in SOURCES:
        if not src.exists():
            print(f"  [SKIP] {src.name} not found")
            continue
        count = 0
        for line in open(src):
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
                count += 1
        print(f"  Loaded {count:3d} tasks from {src.name}")
    return tasks


def reassign_task_ids(tasks: list[dict]) -> list[dict]:
    """Assign stable sequential IDs TB-0001…TB-NNNN after merging all sources."""
    for i, t in enumerate(tasks, start=1):
        t["task_id"] = f"TB-{i:04d}"
    return tasks


def partition(tasks: list[dict], seed: int = SEED) -> tuple[list, list, list]:
    random.seed(seed)
    shuffled = tasks[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_held = max(1, round(n * HELD_RATIO))
    n_dev  = max(1, round(n * DEV_RATIO))
    n_train = n - n_held - n_dev

    train    = shuffled[:n_train]
    dev      = shuffled[n_train:n_train + n_dev]
    held_out = shuffled[n_train + n_dev:]
    return train, dev, held_out


def write_partition(tasks: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")


def manifest(tasks: list[dict], train: list, dev: list, held_out: list) -> dict:
    family_counts = Counter(t["failure_family"] for t in tasks)
    source_counts = Counter(t["source_mode"] for t in tasks)
    diff_counts   = Counter(t["difficulty"] for t in tasks)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "total_tasks": len(tasks),
        "partitions": {
            "train": len(train),
            "dev": len(dev),
            "held_out": len(held_out),
        },
        "by_failure_family": dict(sorted(family_counts.items())),
        "by_source_mode": dict(sorted(source_counts.items())),
        "by_difficulty": dict(sorted(diff_counts.items())),
        "train_ratio": TRAIN_RATIO,
        "dev_ratio": DEV_RATIO,
        "held_out_ratio": HELD_RATIO,
    }


def main():
    print("── Loading tasks ──")
    tasks = load_all_tasks()
    if not tasks:
        print("No tasks found. Run generate_programmatic.py and generate_synthesis.py first.")
        return

    tasks = reassign_task_ids(tasks)
    print(f"  Total: {len(tasks)} tasks")

    print("\n── Partitioning ──")
    train, dev, held_out = partition(tasks)
    print(f"  Train: {len(train)}, Dev: {len(dev)}, Held-out: {len(held_out)}")

    write_partition(train,    BASE / "train" / "tasks.jsonl")
    write_partition(dev,      BASE / "dev" / "tasks.jsonl")
    write_partition(held_out, BASE / "held_out" / "tasks.jsonl")

    mf = manifest(tasks, train, dev, held_out)
    with open(BASE / "dataset_manifest.json", "w") as f:
        json.dump(mf, f, indent=2)
    print(f"\n  Manifest → {BASE / 'dataset_manifest.json'}")

    print("\n── Contamination checks ──")
    from contamination_check import run_all_checks
    report = run_all_checks(
        BASE / "train" / "tasks.jsonl",
        BASE / "held_out" / "tasks.jsonl",
        BASE / "contamination_report.json",
    )

    print("\n── Summary ──")
    print(f"  Failure families : {mf['by_failure_family']}")
    print(f"  Source modes     : {mf['by_source_mode']}")
    print(f"  Difficulty       : {mf['by_difficulty']}")
    print(f"  Contamination    : {'CLEAN ✓' if report['overall_pass'] else 'ISSUES FOUND ✗'}")


if __name__ == "__main__":
    main()
