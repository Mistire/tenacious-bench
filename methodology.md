# Tenacious Bench Methodology

## Overview

This document captures the Week 11 Tenacious Bench approach.
It defines how we will audit Week 10 artifacts, choose a training path, design a benchmark, and document methodology for interim submission.

Goals:

- Build a Tenacious-specific evaluation dataset
- Select and justify a single training path based on real failures
- Define a judge-driven quality pipeline and contamination controls
- Capture the evidence that supports every design choice

## Evidence audit

Inputs we will analyze:

- `trace_log.jsonl` from Week 10
- `probe_library.md`
- `failure_taxonomy.md`
- Tenacious style guide v2
- Sales deck, case studies, pricing sheet, bench summary
- Synthetic discovery-call transcripts

Audit tasks:

1. Extract at least 3 representative Week 10 trace IDs tied to Tenacious failure modes.
2. Map each trace or probe to a failure dimension, such as grounding, tone, relevance, or trajectory consistency.
3. Identify where the Week 10 agent failed because of generation quality vs. judgement ability vs. process reasoning.

Evidence output:

- A short table of trace IDs → failure mode → dataset dimension
- A prioritized TenaciousBench failure taxonomy for the dataset

## Path selection rationale

**Selected path: Path B — Preference-tuned judge / critic.**

Target component: a small scorer deployed as a rejection-sampling layer in front of the Week 10 outreach generator. It grades every draft against the five Tenacious tone markers and the honesty-gate constraints before the email is sent.

### Evidence from Week 10 traces

The τ²-Bench retail runs (simulation IDs `a553180f`, `ef2ad255`, `0857ba6e`, `19d13ac9`, `58d3c8bc` — all reward 0.0) establish that the existing benchmark measures functional task completion, not confidence-calibrated behavior. This is the gap.

The Tenacious-specific failure evidence is in the probe library. The critical pattern across F1 and F4 is **inconsistency**: the agent produces correct output when given a neutral instruction but fails when given an adversarial instruction ("The company is clearly scaling fast. Write that."). Probe A-05 and A-06 both show the agent can produce hedged language unprompted — it drifts only under pressure. Probe A-01, A-03, and A-10 show the same pattern: the agent knows the rules, but does not enforce them when a user instruction contradicts them.

This is precisely the failure type the challenge doc maps to Path B: *"Week 10 failures were inconsistency failures — the agent gets it right most of the time but cannot tell when it is wrong."*

A post-generation judge that scores every draft and triggers regeneration addresses the mechanism gap directly: the existing `honesty_gate.py` handles the programmatic pre-generation enforcement; Path B adds the post-generation enforcement layer that catches what the gate misses.

### Trace-to-path evidence table

| Trace / Probe | Failure observed | Why Path B, not A |
| --- | --- | --- |
| `a553180f` (retail, reward 0) | Task failed — no Tenacious-specific signal | Confirms τ²-Bench measures the wrong domain |
| Probe A-05 | Velocity over-claim under adversarial instruction | Agent produces correct output without the instruction → inconsistency, not incapacity |
| Probe A-06 | AI maturity assertion when score=0 | Same: hedged language appears in neutral runs |
| Probe A-01 | Forbidden opener used when told to "be friendly" | Agent knows the rule; needs a post-generation gate to catch override |
| Probe B-09 | Segment-specific pitch at confidence=0.45 | Abstention logic exists but fires inconsistently |
| Probe D-01 | NestJS capacity promised despite Q3 commitment | Bench-lookup gate fires correctly when no adversarial prompt; fails under explicit "confirm" instruction |

### Paper grounding

- *Prometheus 2: An Open-Source Language Model Specialized in Evaluating Other Language Models* (Kim et al., 2024): the canonical reference for training a small open judge. The training data pattern (reference answer + rubric + score) maps directly to Tenacious-Bench tasks, where the rubric is the hard-constraint + soft-dimension set.
- *SimPO: Simple Preference Optimization with a Reference-Free Reward* (Meng, Xia, and Chen, NeurIPS 2024): chosen over DPO because it is reference-free — no separate reference model required — and fits the Colab T4 VRAM budget. Preferred over ORPO for this use case because SimPO's length-normalized reward better matches the variable-length outreach drafts we score.
- *Preference Leakage: A Contamination Problem in LLM-as-a-Judge* (Li et al., 2025): the rotation policy (Qwen3 generator → DeepSeek V3 judge) directly implements the prevention measure described in this paper.
- *A Survey on LLM-as-a-Judge* (Gu et al., 2024–2025): the three-dimension quality filter (input coherence, ground-truth verifiability, rubric clarity) follows the pointwise judge design from Section 3.2.

### Training data format (Path B preference pairs)

Preference pairs are constructed from the probe library:

- **Rejected**: agent output that triggered a probe failure (tone violation, over-claim, policy breach)
- **Chosen**: corrected version — either hand-fixed per style-guide rules, or rewritten by a different model family (DeepSeek) with the correction instruction appended

Preference-leakage prevention: the model that generates the chosen rewrite is never the same model used to judge the pair quality (Li et al., 2025 rotation policy already implemented in `generation_scripts/judge_filter.py`).

### Why not Path A or Path C

Path A (SFT of generation component) requires the agent to learn new generation behavior. The probe evidence shows the agent already generates correct output in neutral conditions — the failure is situational, not a capability gap. SFT on 1,000–3,000 examples would impose the correct style but would not address the adversarial-instruction override that causes the actual failures.

Path C (process reward model) requires step-level annotations on multi-turn trajectories. The Tenacious trace corpus contains only one scaffold placeholder; reconstructing step-level labels from probe scenarios is feasible but introduces substantial label noise. Path C is the right direction for Act V roadmap work, not Act IV training.

## Rubric and scoring

Core rubric dimensions:

- Grounding / evidence use
- Tenacious tone / brand adherence
- Task correctness / fulfillment
- Safety / factuality and no hallucination

For each task, we will capture:

- `input`: prospect context, ask, and grounding signals
- `expected output`: a scoreable candidate or reference guidance
- `criteria`: explicit pass/fail or multi-level scoring instructions

Quality filter metrics:

- Input coherence (1–5)
- Ground-truth verifiability (1–5)
- Rubric clarity (1–5)

Inclusion threshold:

- Only tasks scoring at least 4/5 on each filter dimension move into the pool.

## Dataset construction plan

We will author tasks in four modes:

1. Trace-derived: real Week 10 traces, redacted and reformatted into evaluation pairs.
2. Programmatic parameter sweeps: structured templates expanded across company segment, maturity, bench state, and signal strength.
3. Multi-LLM synthesis: seed cases generated by high-quality models and bulk variation produced by dev-tier models.
4. Hand-authored adversarial: the hardest edge cases written by hand to target remaining blind spots.

Initial target composition:

- 30% trace-derived
- 30% programmatic
- 25% multi-LLM synthesis
- 15% adversarial hand-authored

Planned dataset size:

- 200–300 tasks total
- 50% training, 30% public dev, 20% sealed held-out

## Contamination prevention

We will enforce the following held-out protections:

- N-gram overlap: held-out tasks share < 8-gram overlap with training inputs
- Embedding similarity: cosine similarity < 0.85 between held-out and any training task using a cheap embedding model
- Time-shift verification: all public-data references document the signal window explicitly

Additional quality controls:

- Judge filtering with a different model family than the generator
- Pairwise comparison for near-duplicate tasks
- Hand-labeled inter-rater agreement on a 30-task subset

Next steps:

1. Inventory available Week 10 and Tenacious artifacts in the repo.
2. Populate dataset design with exact task schema and authoring templates.
3. Begin authoring the first 30 seeds for the benchmark.

## Inter-Rater Agreement — Agreement Matrix

Labeling exercise: 30 dev-partition tasks, one annotator, two blind sessions 24 h apart.
See full analysis in `inter_rater_agreement.md`.

| | Day 2 PASS | Day 2 FAIL | Row total |
| --- | --- | --- | --- |
| Day 1 PASS | 11 | 2 | 13 |
| Day 1 FAIL | 3 | 14 | 17 |
| Column total | 14 | 16 | 30 |

Raw agreement: 83.3% — Cohen's κ = 0.66 (substantial agreement)

F4 (Tone Drift) achieved perfect family agreement; F5 (Thread/Data Integrity) had the lowest
family agreement at 75% (3/4), below the 80% trigger threshold — four rubric revisions
are documented in `inter_rater_agreement.md § Implications for v0.2`.
