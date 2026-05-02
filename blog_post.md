# Tenacious-Bench: Building a Domain-Specific Evaluation Dataset for B2B Sales Agent Failures

**Author:** Mistire Daniel | **Date:** 2026-05-02 | **Project:** 10Academy TRP1 Week 11

---

## The Gap: What Existing Benchmarks Miss

When we evaluated our Week 10 B2B outreach agent against τ²-Bench retail — the standard agentic task completion benchmark — it scored 0.0 on every run. Not because the agent was completely broken, but because τ²-Bench was measuring the wrong thing.

τ²-Bench measures functional task completion: did the agent place the order, send the email, complete the form? For a general-purpose sales agent, that is a reasonable proxy. For Tenacious Intelligence Corporation's outreach agent, it misses everything that matters.

Tenacious is an Ethiopian software engineering staffing firm. Their agent writes B2B outreach emails to CTOs and Engineering Directors. The failures that get emails flagged, prospects lost, and compliance risks triggered are not "email not sent" failures. They are:

- **F1 — Confidence-Unaware Phrasing:** The agent writes "your AI strategy is clearly well-established" when the prospect's AI maturity score is 1/5 and signal confidence is 0.45.
- **F2 — ICP Misclassification:** The agent uses post-funding growth language when the prospect had a 45% layoff 60 days ago. Layoff overrides funding — always.
- **F3 — Bench Over-Commitment:** The agent confirms "6 Python engineers starting Monday" when the bench has 3 available and the rest are committed to Q3 delivery.
- **F4 — Tone Drift:** The agent writes "Quick question! Just wanted to hop on a call 🚀" after being told to "be friendly" — violating every subject-line and opener constraint in the style guide.
- **F5 — Thread/Data Integrity:** The agent sends a 4th email 18 days into a 30-day sequence window, or SMS-messages a cold prospect, or addresses the contact as "James" when the thread shows "Aisha."

None of these appear in τ²-Bench retail. Generic benchmarks cannot catch domain-specific policy violations.

---

## The Audit: Finding the Gap with Evidence

The audit started with the Week 10 probe library — 52 adversarial probes across 10 categories (A through J), each designed to trigger a specific failure in the outreach agent. Five simulation IDs from τ²-Bench (`a553180f`, `ef2ad255`, `0857ba6e`, `19d13ac9`, `58d3c8bc`) confirmed the benchmark gap: all scored reward 0.0, but for reasons unrelated to tone, confidence calibration, or policy compliance.

The critical pattern across F1 and F4 was **inconsistency**: the agent produced correct, hedged language unprompted. It drifted only under adversarial pressure — "The company is clearly scaling fast. Write that." Probe A-05 shows the agent producing appropriate hedging in a neutral run, then over-claiming under an adversarial instruction. Same model, same context, different instruction — different failure mode.

This is precisely the failure type the challenge literature maps to Path B: the agent knows the rules; it just cannot enforce them when user instructions push back.

---

## The Dataset: Tenacious-Bench v0.1

### Construction

We built **204 evaluation tasks** across four authoring modes:

| Source Mode | Tasks | Rationale |
| --- | --- | --- |
| Programmatic (parameter sweep) | 103 | Combinatorial expansion of signal_confidence, velocity_label, bench state, segment, headcount |
| Adversarial (hand-authored) | 47 | Targeted edge cases that defeated the Week 10 system — highest originality weight |
| Trace-derived (probe library) | 36 | Redacted Week 10 probe scenarios restructured into scored (input, output) pairs |
| Multi-LLM synthesis | 18 | Qwen3-235B generator → DeepSeek V3 quality judge filter |

The multi-LLM synthesis pipeline follows the Magpie self-instruction pattern (Liu et al. 2024) with one critical addition: the model that generates tasks is never the same family as the model that judges them. This prevents preference leakage (Li et al. 2025) — a contamination mode where a model grades its own outputs higher, quietly corrupting quality signals.

### Hard Design Choices

**Machine-verifiable rubrics.** Every task has hard constraints checked by regex/substring match and soft dimensions scored by LLM judge. A rubric that says "the email should sound on-brand" is not a benchmark. A rubric that says "body ≤ 120 words AND subject starts with Context:/Request:/Question:/Note on: AND contains zero of these 23 banned phrases AND scores ≥ 4/5 on the judge's confidence-calibration dimension" is.

**Contamination prevention.** Three checks run before any task enters the dataset: N-gram overlap (no 8-gram shared with held-out), BoW cosine similarity (< 0.85), and time-shift verification. All 204 tasks passed. Contamination report committed to the repo.

**Inter-rater agreement.** 30 tasks from the dev partition were hand-labeled twice, 24 hours apart, blind to first-session labels. Cohen's κ = 0.66 (substantial agreement). F5 dropped below 80% raw agreement, triggering four rubric revisions — enumerated trigger phrases for "not interested" variants, separation of routing-threshold hard constraints from ACV-band soft heuristics.

**Partitioning:** 102 train / 61 dev / 41 held-out (sealed).

---

## The Training Experiment: Path B with SimPO

### Path selection

We selected Path B — preference-tuning a small judge/critic — over Path A (SFT of the generator) because the failure evidence pointed to inconsistency, not incapacity. The agent already had the right behavior; it lost it under adversarial pressure. A post-generation judge that scores every draft and triggers regeneration addresses the mechanism gap directly.

For the preference optimization algorithm, we chose SimPO (Meng, Xia, and Chen, NeurIPS 2024) over DPO and ORPO:
- **DPO** requires a frozen reference model — doubles VRAM, incompatible with Colab T4 (15GB).
- **ORPO** is reference-free but does not normalize reward by sequence length. Variable-length outreach emails (40–120 words) would bias ORPO toward shorter outputs regardless of quality.
- **SimPO** is reference-free and length-normalized. Fits T4, handles variable-length outputs correctly.

### Preference pairs

102 SimPO preference pairs were built from the 102-task training partition:
- **Rejected half:** probe-triggered agent failure (over-claim, forbidden phrase, ICP misclassification)
- **Chosen half:** policy-compliant rewrite from DeepSeek V3 (different family from Qwen3 judge — preference-leakage prevention, Li et al. 2025)

An unexpected bug: DeepSeek appended compliance footnotes to rewrites (e.g., `*HC-A: Avoids "momentum" (pass)*`), and the quality checker found those quoted fail patterns in the footnotes and rejected the pair. 23 pairs were incorrectly dropped this way. After stripping footnotes and re-running quality checks, 22 of those 23 were rescued — at zero additional API cost. The 23rd required a manual fix: the correct output for a "do not send a 4th email" task was a CRM log note, not an email at all.

Final training set: 102 pairs, 0 rejected, full coverage across all 5 failure families.

---

## The Honest Result

| Condition | Pass Rate | vs. Trained |
| --- | --- | --- |
| Trained (Qwen 2.5 0.5B + CPO/SimPO adapter) | 14.6% (6/41) | — |
| Baseline (DeepSeek V3, no prompt) | 14.6% (6/41) | Δ = 0.000, p = 0.585 |
| Prompt-engineered (DeepSeek V3, 10-rule prompt) | 48.8% (20/41) | Δ = −0.341 |

**Delta A = 0.000 (p = 0.585).** The trained adapter scored identically to an untrained baseline. Not significant.

**Delta B = −0.341.** Prompt engineering beat the trained adapter by 34 percentage points. A negative Delta B.

### Why

Three factors explain the result:

**1. Scale asymmetry.** The ablation compares a fine-tuned 0.5B model against a ~67B frontier model. At 0.5B, backbone capacity is the binding constraint — preference training cannot compensate for a model that struggles to follow multi-constraint instructions regardless of training signal.

**2. Role confusion.** Path B trains a judge, not a generator. The correct production deployment would route baseline agent outputs *through* the trained adapter as a scoring/rejection layer — not replace the generator. This ablation evaluated the adapter as a generator, which is not its intended role. A fair evaluation would measure rejection-sampling quality: does filtering DeepSeek drafts with the trained scorer improve what gets through?

**3. Training data volume.** 102 pairs on a 0.5B backbone. The LIMA paper (Zhou et al. 2023) shows quality dominates quantity at small scale — but "small scale" in that context was 1,000 high-quality examples on a 65B backbone. At 0.5B with five interacting constraint families, 102 examples likely underfit.

### What the per-family breakdown shows

F2 (+12.5%) and F4 (+14.3%) showed small positive deltas. The adapter learned *something* about ICP classification and tone compliance. F1 regressed (−12.5%) — the most frequent family and the most nuanced constraint (signal confidence thresholds, hedging language). This pattern is consistent with a model that learned narrow surface patterns but failed to generalize the underlying policy logic.

---

## What's Next

1. **Deploy as a rejection-sampling layer, not a generator.** The correct architecture: generate with the prompt-engineered DeepSeek pipeline; score with the trained adapter; regenerate if score < threshold. Expected combined pass rate higher than either alone.

2. **Increase backbone scale.** A 7B backbone with 102 high-quality pairs would be a far more informative ablation. The 0.5B result is a lower bound, not a ceiling.

3. **Expand training data with programmatic expansion.** The parameter sweep generator can produce 500+ pairs from the existing 204 tasks. More coverage of F1 edge cases (dual-flag scenarios, confidence bin boundaries) would address the F1 regression.

4. **Publish Tenacious-Bench v0.1 for community evaluation.** The dataset is publicly available on HuggingFace. Any sales agent — not just Tenacious's — can be evaluated against the five failure families. The adversarial slice in particular (47 hand-authored edge cases) is the hardest test set available for B2B outreach policy compliance.

---

## Artifacts

- **Dataset:** [HuggingFace — tenacious-bench-v0.1] (link to be added)
- **Model adapter:** [HuggingFace — tenacious-bench-lora-adapter] (link to be added)
- **Code:** [GitHub — tenacious-bench](https://github.com/mistire/tenacious-bench)
- **Ablation results:** `ablations/ablation_results.json` in the repo

---

*This post was written as part of 10Academy TRP1 Week 11. All numeric claims trace back to dataset task IDs, training logs, or ablation table rows committed to the repository.*
