# Synthesis Memo — Prometheus 2: An Open-Source Language Model Specialized in Evaluating Other Language Models

**Paper:** Kim, S., et al. "Prometheus 2: An Open-Source Language Model Specialized in Evaluating Other Language Models." *arXiv 2024*.
**Author:** Mistire Daniel | **Date:** 2026-05-01

---

## Core Claim

Prometheus 2 is a 7B open judge model trained on ~100K preference pairs to produce absolute and relative evaluations that correlate with human judgment better than GPT-4 on domain-specific rubrics. The key contribution over Prometheus 1 is the addition of pairwise evaluation capability (not just pointwise), and the mixture training that combines absolute and relative feedback signals into a single judge.

The central architectural decision is training on human-written rubrics, not on generic helpfulness criteria. This produces a judge that can follow novel domain-specific rubrics reliably — the paper shows this generalizes to held-out rubric categories not seen during training.

## Relevance to Tenacious-Bench Path B

Path B trains a small judge/critic to score Tenacious outreach drafts against the five-constraint rubric. Prometheus 2 is the canonical prior work for this design: train a small open model as a domain-specific judge using preference pairs (chosen=rubric-compliant, rejected=rubric-violating).

Three specific design choices in Prometheus 2 that directly inform our training setup:

**1. Preference pairs over single-pass SFT.** Prometheus 2 found that preference-tuned judges generalize better to novel rubric dimensions than SFT judges trained on correct-output-only examples. This supports choosing Path B over Path A: the agent already produces correct output in neutral conditions (Probes A-05, A-06), so SFT would overfit to the training rubric without addressing the adversarial-instruction override failure.

**2. Rubric-grounded feedback.** Prometheus 2 requires the judge to cite which rubric dimension a score applies to. Our scoring evaluator uses the same design: the judge prompt explicitly names the soft dimension (confidence calibration, ICP routing, etc.) and the threshold. This prevents the judge from rewarding surface plausibility rather than constraint satisfaction.

**3. Reference judge for calibration.** The paper uses GPT-4 outputs as calibration targets for the preference labels. We use DeepSeek V3-0324 in the same role — different family from the training data generator (which also used DeepSeek) but with preference-leakage prevention enforced by using DeepSeek only for *labeling*, not for generating the candidates being labeled.

## Specific Disagreement

Prometheus 2's training corpus is 100K pairs, and the backbone is 7B. The paper shows that 7B is approximately the minimum scale at which a judge model reliably follows novel rubrics — below this, judges degrade to surface-form heuristics rather than rubric compliance.

Our Tenacious-Bench judge is trained on 102 pairs at 0.5B. This is below both the scale and data thresholds the paper identifies as necessary for reliable judge behavior. The fact that Delta A = 0.000 is consistent with this: the trained adapter is at the scale where it cannot yet reliably distinguish rubric compliance from rubric-adjacent surface features.

The paper does not test judges below 7B or with fewer than ~10K pairs. Our experiment extends the curve in the direction Prometheus 2 does not evaluate, and confirms the paper's implicit lower bound: 0.5B with 102 pairs is insufficient for a reliable domain-specific judge on a five-constraint rubric.

**Implication:** The correct Prometheus 2-aligned deployment for Tenacious is a 7B backbone with 500–1,000 high-quality pairs. The 0.5B experiment is a proof-of-concept that establishes the lower bound, not the production recommendation.

## Design Principle Applied

Following Prometheus 2's rubric-grounding recommendation, the scoring evaluator's judge prompts are structured as: "(dimension name): rate 1–5 on whether the output satisfies (specific constraint). Score ≥ 4 required to pass." This prevents the LLM judge from grading on holistic impression rather than the stated constraint — the same distinction Prometheus 2 identifies as the key gap between generic and rubric-specialized judges.
