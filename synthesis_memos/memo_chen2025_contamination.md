# Common-Reading Memo: Chen et al. EMNLP 2025

**Paper:** *Recent Advances in Large Language Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation*
**Authors:** Chen et al.
**Venue:** EMNLP 2025
**Read by:** Mistire Daniel — 2026-04-30

---

## Summary

Chen et al. survey contamination-prevention techniques across 80+ benchmark papers, distinguishing between *static* benchmarks (fixed task sets that can be memorized) and *dynamic* benchmarks (task sets that change over time or are generated on-demand). Their central finding is that static benchmarks degrade in discriminative power within 6–18 months of public release because frontier models are trained on web-scraped data that increasingly includes benchmark tasks.

Their taxonomy of contamination types is the paper's most useful contribution:
1. **Direct contamination**: the exact task appears in training data
2. **Paraphrase contamination**: a semantically equivalent task appears in training data
3. **Template contamination**: the task template (not the specific instance) appears in training data, allowing models to recognize the task type and apply memorized strategies

Their recommended defenses, in order of effectiveness:
1. **Dynamic generation**: generate tasks at evaluation time from a seed that is not public
2. **N-gram overlap checks**: catch direct contamination (necessary but not sufficient)
3. **Embedding similarity checks**: catch paraphrase contamination (more expensive but catches more)
4. **Time-shift verification**: ensure tasks reference data from a window that postdates the model's training cutoff

---

## Where Tenacious-Bench Agrees

**Three-check contamination pipeline — implemented.** `contamination_check.py` runs all three checks Chen et al. recommend: N-gram overlap (8-gram threshold), embedding similarity (BoW cosine < 0.85), and time-shift verification. The contamination report is committed alongside the dataset.

**Held-out partition sealed separately — implemented.** The held-out partition is gitignored from training scripts and stored in a separate file. Chen et al. document cases where held-out tasks leaked into training data through shared file paths — our partition structure prevents this.

**Template contamination awareness — partially addressed.** The adversarial task slice (47 tasks) was specifically designed to defeat template-recognition strategies: the adversarial instructions use framing tricks ("just for planning purposes," "the prospect expects confidence") that a model trained on standard probe templates would not recognize as the same task type.

---

## Where Tenacious-Bench Disagrees

**Disagreement: dynamic generation is not the right defense for a domain-specific benchmark.**

Chen et al.'s strongest recommendation is dynamic generation — generating tasks at evaluation time from a private seed so that no fixed task set can be memorized. They demonstrate that dynamic benchmarks maintain discriminative power 3× longer than static benchmarks on their meta-analysis.

For Tenacious-Bench, dynamic generation would undermine the benchmark's primary value: **reproducibility**. The benchmark's purpose is to measure whether a trained component lifts a specific agent on specific Tenacious failure modes. If the task set changes at every evaluation, the Delta A measurement (trained vs. baseline on held-out) cannot be compared across runs, and the ablation table loses its meaning.

Dynamic generation is the right defense for general-capability benchmarks (MMLU, HumanEval) where the goal is to measure a model's general ability without memorization. For a domain-specific benchmark measuring compliance with a fixed rubric, the contamination risk is different: the risk is not that a model memorizes the exact task, but that it memorizes the rubric's fail patterns. This is a different threat model that dynamic generation does not address.

**Evidence from our data:** The contamination check found 0 N-gram violations and 0 embedding violations after the instruction diversity fix. The threat model for Tenacious-Bench is not memorization of specific tasks — it is that a model trained on the training partition learns to avoid the exact fail_patterns in the rubric. This is addressed by the adversarial task design (which uses novel framing tricks not present in the training partition) rather than by dynamic generation.

**Conclusion:** Dynamic generation is appropriate for general-capability benchmarks. For domain-specific rubric-compliance benchmarks, the correct defense is adversarial task design that tests the underlying behavior rather than the surface pattern. Tenacious-Bench v0.2 should expand the adversarial slice to 80+ tasks (from 47) to increase coverage of novel framing tricks.

---

## Application to Tenacious-Bench Contamination Protocol

The Chen et al. taxonomy of contamination types directly informed the three-check pipeline design:
- N-gram check → catches direct contamination (Type 1)
- Embedding check → catches paraphrase contamination (Type 2)
- Time-shift check → catches template contamination via public data references (Type 3)

One Chen et al. finding we have not yet implemented is **cross-benchmark contamination checking**: verifying that Tenacious-Bench tasks do not overlap with τ²-Bench retail tasks. Since τ²-Bench retail is a public benchmark, a model trained on τ²-Bench data could potentially recognize Tenacious-Bench task structures if they share template patterns. Tenacious-Bench v0.2 should add a cross-benchmark N-gram check against the τ²-Bench retail task set.

The paper's finding that embedding similarity checks catch 2.3× more contamination than N-gram checks alone motivates upgrading the BoW cosine check to `sentence-transformers` in v0.2 — a known limitation documented in `interim_report.md`.
