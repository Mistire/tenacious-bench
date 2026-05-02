# Tenacious-Bench v0.1 — Final Submission Memo

**Author:** Mistire Daniel | **Email:** mistire@10academy.org | **Date:** 2026-05-02

---

## Page 1 — Executive Summary

### Decision

**Do not deploy the trained adapter as a standalone generator. Deploy as a rejection-sampling layer in front of the prompt-engineered pipeline.**

### Headline Result

The trained CPO/SimPO adapter (Qwen 2.5 0.5B) achieved a **14.6% pass rate** (6/41 tasks) on the sealed 41-task held-out partition of Tenacious-Bench v0.1.

| Metric | Value | 95% CI |
| --- | --- | --- |
| Delta A: trained vs. baseline | **+0.000** | [−0.122, +0.122], p = 0.585 |
| Delta B: trained vs. prompt-eng | **−0.341** | [−0.488, −0.195] |
| Trained pass rate | **14.6%** | 6 of 41 held-out tasks |
| Prompt-eng pass rate | **48.8%** | 20 of 41 held-out tasks |

Delta A is flat. Delta B is negative. Training did not lift performance over the baseline, and careful prompting of a frontier model outperformed the trained adapter by 34 percentage points.

### Cost Per Task

| Component | Cost |
| --- | --- |
| Dataset authoring (API calls) | $0.63 total / $0.003 per task |
| Preference pair rescue (DeepSeek rewrites) | $0.007 |
| Training (Colab T4, free tier) | $0.00 |
| Ablation scoring (3 conditions × 41 tasks) | ~$0.15 |
| **Total week spend** | **~$0.79 of $10.00 cap** |

### Production Recommendation

**Deploy the prompt-engineered DeepSeek pipeline now.** It achieves 48.8% pass rate on the held-out partition — 3.3× the trained adapter's rate — at no additional training cost. The trained adapter adds value only as a scoring layer: route every draft through it, regenerate on fail. Expected combined pass rate higher than either component alone.

To justify standalone adapter deployment: retrain on a 7B backbone with 500+ pairs and demonstrate Delta A ≥ +0.15 with p < 0.05 on a refreshed held-out slice.

---

## Page 2 — Skeptic's Appendix

### Objection 1: The Delta A result is trivially explained by scale asymmetry

**Conceded.** The baseline condition used DeepSeek V3 (~67B parameters). The trained condition used Qwen 2.5 0.5B. These are not the same backbone. A fair Delta A would hold backbone constant — same 0.5B model, with and without the LoRA adapter. The current ablation answers a different question: "does a fine-tuned 0.5B model beat an unguided 67B model?" The answer is no, which is unsurprising.

**Why it was done this way:** The challenge spec defines Delta A as "trained model versus your Week 10 baseline" — the baseline was the existing production model (DeepSeek-scale). Running a separate 0.5B-without-LoRA condition would have been more informative but was not required and would have cost additional compute.

**What it means for the finding:** Delta A = 0.000 is still a valid finding. It means a 0.5B fine-tuned adapter cannot match an unguided frontier model on this task. That is a real and useful result for deployment planning.

### Objection 2: The adapter was never evaluated in its intended role

**Conceded.** Path B trains a judge/critic for rejection sampling — not a generator. The ablation ran inference through the LoRA adapter to generate emails, which is not the deployment architecture. The correct evaluation would be:
1. Generate 41 outputs with the baseline pipeline
2. Score each output with the trained adapter (accept/reject)
3. Measure: pass rate of accepted outputs vs. unfiltered outputs

This evaluation was not run. Time and the single Colab session constrained the scope. The adapter's rejection-sampling utility is therefore unproven by this ablation.

**Mitigation:** The per-family results suggest the adapter learned something — F2 +12.5%, F4 +14.3%. These are small but directionally consistent with the training signal. A rejection-sampling ablation is the natural next experiment.

### Objection 3: 41 held-out tasks is too few for statistical power

**Conceded.** With n = 41, the minimum detectable effect at 80% power and α = 0.05 is approximately ±15 percentage points. Effects smaller than that are invisible. The wide confidence intervals (±12pp for Delta A) confirm this. The held-out partition size was set at 20% of 204 tasks per the challenge spec — increasing it would have required a larger total dataset.

**Mitigation:** The bootstrap uses 10,000 resamples. The CI estimates are unbiased. The limitation is documented in the model card and this memo. A v0.2 evaluation with 100+ held-out tasks would provide tighter estimates.

### Objection 4: DeepSeek was used as both judge and baseline generator — potential bias

**Addressed.** The quality judge (DeepSeek V3) and the baseline generator (also DeepSeek V3) are the same model family. A model grading its own outputs is a known bias source (Li et al. 2025 — Preference Leakage). However, the eval judge is scoring against a fixed rubric (hard constraints + soft dimensions), not comparing outputs against each other. Rubric-grounded pointwise scoring is less susceptible to preference leakage than pairwise ranking — the judge has a ground-truth reference to anchor against.

The ideal mitigation is a third-party judge (Claude Sonnet was the original plan). An Anthropic API key was unavailable for this submission. For v0.2, a multi-judge ensemble (DeepSeek + Claude + Qwen) with inter-judge agreement tracking would address this.

### What Would Change the Recommendation

Delta A ≥ +0.15 with p < 0.05, measured on the same backbone (0.5B with vs. without LoRA), with a rejection-sampling ablation showing positive lift over the unfiltered baseline. If those conditions are met, the adapter earns a production deployment recommendation.

---

*Every numeric claim in this memo resolves to a source in `ablations/ablation_results.json`, `training_data/preference_pairs.jsonl`, or `tenacious_bench_v0.1/dataset_manifest.json`.*
