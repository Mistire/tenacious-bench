# Community Engagement Artifact

**Author:** Mistire Daniel
**Date:** 2026-05-02

---

## HuggingFace Dataset Discussion Post

**Posted to:** https://huggingface.co/datasets/mistire37/tenacious-bench-v0.1/discussions

**Title:** Tenacious-Bench v0.1 — a domain-specific B2B sales agent evaluation dataset (feedback welcome)

**Body:**

Hi HuggingFace community,

I'm sharing Tenacious-Bench v0.1, a 204-task evaluation dataset for B2B sales outreach agents, built as part of the 10Academy TRP1 program (Week 11).

**What it measures:**
Most general-purpose agentic benchmarks measure functional task completion (did the email send?). Tenacious-Bench measures five failure dimensions that matter in production B2B sales:
- F1: Confidence-calibrated phrasing (no velocity over-claims when signal confidence < 0.6)
- F2: ICP segment routing (layoff overrides funding — always)
- F3: Bench capacity honesty (never confirm headcount you don't have)
- F4: Tone adherence (no forbidden openers, no emoji, single CTA)
- F5: Thread/data integrity (no 4th email in 30 days, no SMS to cold prospects)

**What's in the dataset:**
- 204 tasks across 4 authoring modes (programmatic, adversarial, trace-derived, multi-LLM synthesis)
- Machine-verifiable rubrics: hard constraints via regex + soft dimensions via LLM judge
- Full datasheet (Gebru et al. 2021 format)
- Contamination check: CLEAN (N-gram, BoW cosine, time-shift)
- Inter-rater agreement: Cohen's κ = 0.66

**Training experiment (Path B — SimPO judge):**
Trained a CPO/SimPO adapter on Qwen 2.5 0.5B using 102 preference pairs. Honest result: Delta A = 0.000 (flat vs. baseline), Delta B = −0.341 (prompt engineering beat training by 34pp). Scale asymmetry and backbone size are the likely causes — documented fully in the model card.

**Links:**
- Dataset: https://huggingface.co/datasets/mistire37/tenacious-bench-v0.1
- Adapter: https://huggingface.co/mistire37/tenacious-bench-lora-adapter

Happy to answer questions on the rubric design, contamination protocol, or the honest negative Delta B result. If you work on sales AI or agentic evaluation, I'd love feedback on the failure taxonomy.

---

## 10Academy Community Post

**Posted to:** 10Academy TRP1 cohort channel

**Title:** Week 11 submission — Tenacious-Bench v0.1 published

Sharing my Week 11 work: built Tenacious-Bench v0.1, a 204-task evaluation dataset for the Tenacious outreach agent, trained a SimPO preference adapter on 102 pairs, and ran full ablations.

Key numbers:
- 204 tasks, 4 source modes, 5 failure families, κ=0.66 inter-rater agreement
- $0.79 total API spend (under $10 cap)
- Honest finding: Delta A = 0.000, Delta B = −0.341 — prompt engineering beat training by 34pp on the held-out partition

Dataset + adapter live on HuggingFace:
- https://huggingface.co/datasets/mistire37/tenacious-bench-v0.1
- https://huggingface.co/mistire37/tenacious-bench-lora-adapter

Full write-up in blog_post.md in the repo. Happy to discuss the negative Delta B — it's documented honestly in the model card skeptic's appendix.
