---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - evaluation
  - sales
  - b2b
  - benchmark
  - preference-learning
pretty_name: Tenacious-Bench v0.1
size_categories:
  - n<1K
---

# Tenacious-Bench v0.1

**[Dataset on HuggingFace](https://huggingface.co/datasets/mistire37/tenacious-bench-v0.1)** · **[Model Adapter](https://huggingface.co/mistire37/tenacious-bench-lora-adapter)** · **Author: Mistire Daniel**

A domain-specific evaluation dataset for B2B sales agents operating within the Tenacious Intelligence Corporation workflow. Measures confidence-calibrated phrasing, ICP classification, bench capacity honesty, tone adherence, and data integrity — five failure dimensions that general-purpose benchmarks (including τ²-Bench retail) cannot grade.

## Dataset

204 tasks · train/dev/held-out split · contamination CLEAN · Cohen's κ = 0.66

| Partition | Tasks |
| --- | --- |
| Train | 102 |
| Dev (public) | 61 |
| Held-out (sealed) | 41 |

| Failure Family | Tasks |
| --- | --- |
| F1 — Confidence-Unaware Phrasing | 56 |
| F2 — ICP Classification | 34 |
| F3 — Bench Over-Commitment | 43 |
| F4 — Tone Drift | 38 |
| F5 — Thread / Data Integrity | 33 |

| Source Mode | Tasks |
| --- | --- |
| Programmatic (parameter sweep) | 103 |
| Adversarial hand-authored | 47 |
| Trace-derived (probe library) | 36 |
| Multi-LLM synthesis (Qwen3 + DeepSeek judge) | 18 |

## Evaluation Results (Held-out, n=41)

| Condition | Pass Rate |
| --- | --- |
| Trained adapter (Qwen 2.5 0.5B + CPO/SimPO) | 14.6% |
| Baseline (DeepSeek V3, no prompt) | 14.6% |
| Prompt-engineered (DeepSeek V3, 10-rule prompt) | 48.8% |

Delta A = +0.000 (p=0.585) · Delta B = −0.341 · Full results in [`ablations/ablation_results.json`](ablations/ablation_results.json)

## Act I Deliverables

| File | Description |
| --- | --- |
| [`audit_memo.md`](audit_memo.md) | Gap audit — why τ²-Bench retail cannot grade Tenacious behavior (600 words, 5 dimensions, 6 probe/trace IDs cited) |
| [`schema.json`](schema.json) | Task schema with 3 annotated examples (TB-001 programmatic, TB-002 trace-derived, TB-003 adversarial) |
| [`scoring_evaluator.py`](scoring_evaluator.py) | Machine-verifiable scorer: hard constraints via regex/substring, soft dimensions via LLM judge; `--skip-judge` flag for offline use |
| [`methodology.md`](methodology.md) | Path B (preference-tuned judge/critic) selection with evidence table, 4 paper citations, and training data format |

## Act II Deliverables

| File | Description |
| --- | --- |
| [`generation_scripts/generate_programmatic.py`](generation_scripts/generate_programmatic.py) | 55 tasks via combinatorial parameter sweep across F1–F5, no API |
| [`generation_scripts/generate_synthesis.py`](generation_scripts/generate_synthesis.py) | 36 trace-derived tasks (from 36 probe seeds) + 18 synthesis tasks (Qwen3-235B generator, DeepSeek V3-0324 judge) |
| [`generation_scripts/generate_adversarial.py`](generation_scripts/generate_adversarial.py) | 47 hand-authored adversarial tasks across all 5 failure families, no API |
| [`generation_scripts/judge_filter.py`](generation_scripts/judge_filter.py) | 3-dimension quality filter (IC, GV, RC ≥ 4/5); model rotation policy prevents preference leakage (Li et al. 2025) |
| [`generation_scripts/contamination_check.py`](generation_scripts/contamination_check.py) | N-gram (<8-gram overlap), BoW cosine (<0.85), and time-shift checks |
| [`generation_scripts/partition_dataset.py`](generation_scripts/partition_dataset.py) | Merges all sources, assigns stable IDs, shuffles (seed=42), splits 50/30/20, runs contamination checks |
| [`datasheet.md`](datasheet.md) | Dataset documentation following Gebru et al. 2021 (7 sections) |
| [`inter_rater_agreement.md`](inter_rater_agreement.md) | 30-task double-blind labeling exercise; agreement matrix, Cohen's κ=0.66, 5 disagreement analyses with rubric fixes |
| [`synthesis_memos/`](synthesis_memos/) | Common-reading memos: Liu et al. 2024 (synthetic data best practices), Gu et al. 2024–2025 (LLM-as-a-judge survey) |
| [`budget_log.md`](budget_log.md) | API cost log: $0.55 total (45 calls, ~179K tokens) |
| [`tenacious_bench_v0.1/`](tenacious_bench_v0.1/) | Generated dataset: source JSONLs, partitioned train/dev, manifest, contamination report |

## Quickstart

### 1. Environment setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy the env template and fill in your keys
cp .env.example .env    # or: cp .env .env.local and edit
# Required keys:
#   OPENROUTER_API_KEY   — synthesis, baseline scoring, eval judge (OpenRouter)
#   HF_TOKEN             — HuggingFace write token (publishing only)
# Optional:
#   LANGFUSE_*           — observability (skip if not using Langfuse)
```

### 2. Reproduce the dataset

```bash
# No API needed for programmatic + adversarial sources:
python3 generation_scripts/generate_programmatic.py
python3 generation_scripts/generate_adversarial.py

# Synthesis tasks require OPENROUTER_API_KEY:
python3 generation_scripts/generate_synthesis.py --mode all --limit 20
python3 generation_scripts/partition_dataset.py
```

### 3. Score a task against the rubric

```bash
python3 scoring_evaluator.py --skip-judge       # offline (skips LLM judge)
python3 scoring_evaluator.py                    # full (requires OPENROUTER_API_KEY)
```

## Training Path

### Path B — Preference-tuned judge/critic (SimPO)

The Week 10 probe evidence shows F1/F4 failures are *inconsistency* failures: the agent produces correct output without adversarial instructions but drifts under them. The existing `honesty_gate.py` handles pre-generation enforcement; Path B adds a post-generation rejection-sampling layer that catches what the gate misses.

References: Prometheus 2 (Kim et al. 2024), SimPO (Meng et al. NeurIPS 2024), Preference Leakage (Li et al. 2025), LLM-as-a-Judge Survey (Gu et al. 2024–2025).

## Contamination Controls

- N-gram: held-out shares < 8-gram overlap with any training input
- Embedding: cosine similarity < 0.85 (BoW approximation; production uses sentence-transformers)
- Time-shift: all tasks with public-data references document the signal window
- Model rotation: Qwen3 generates → DeepSeek V3 judges (never the same family)
- Held-out partition gitignored — not committed to this repository
