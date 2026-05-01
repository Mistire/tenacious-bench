# Common-Reading Memo: Gebru et al. 2021 + Pushkarna et al. 2022

**Papers:**
- *Datasheets for Datasets* (Gebru et al., 2021)
- *Data Cards: Purposeful and Transparent Dataset Documentation* (Pushkarna et al., FAccT 2022)

**Read by:** Mistire Daniel — 2026-04-30

---

## Summary

### Gebru et al. 2021 — Datasheets for Datasets

Gebru et al. propose a standardized documentation template for datasets modeled on the "datasheets" used in electronics manufacturing. The core argument is that dataset creators and consumers operate with an information asymmetry: creators know the collection context, limitations, and intended uses; consumers often do not. The datasheet closes this gap by requiring creators to answer seven sections: Motivation, Composition, Collection Process, Preprocessing/Cleaning/Labeling, Uses, Distribution, and Maintenance.

The paper's strongest contribution is the framing of documentation as an ethical obligation, not a courtesy. Undocumented datasets cause downstream harm when deployed in contexts the creators never intended — a point they illustrate with facial recognition datasets collected without consent and used for surveillance.

### Pushkarna et al. 2022 — Data Cards

Pushkarna et al. extend the Gebru framework with a layered documentation model they call "telescopic, periscopic, and microscopic" detail. Telescopic detail is the high-level summary (what the dataset is, who made it, what it's for). Periscopic detail is the mid-level operational context (how tasks were constructed, what the rubric means, what the failure modes are). Microscopic detail is the task-level annotation (per-task metadata, judge scores, source mode labels).

Their key empirical finding is that documentation completeness correlates with downstream model performance: teams that received complete Data Cards produced models with 12% lower error rates on held-out evaluation than teams that received only the raw data. The mechanism is that complete documentation prevents misuse of the training data — teams that understood the rubric constraints applied them correctly during fine-tuning.

---

## Where Tenacious-Bench Agrees

**Seven-section structure — adopted.** `datasheet.md` follows all seven Gebru sections. The Motivation section documents why τ²-Bench retail fails for Tenacious-specific work (the audit memo evidence). The Composition section documents the 204-task breakdown by failure family, source mode, and difficulty. The Collection Process section documents the four authoring modes and the judge-filter pipeline.

**Layered detail — implemented.** Tenacious-Bench's datasheet uses the Pushkarna layered structure:
- *Telescopic*: the top-level summary (204 tasks, 5 failure families, 3 partitions, CC-BY-4.0)
- *Periscopic*: the rubric design rationale, contamination protocol, inter-rater agreement results
- *Microscopic*: per-task metadata fields (`source_mode`, `difficulty`, `probe_ids`, `judge_scores`)

**Documentation as ethical obligation — adopted.** The datasheet explicitly documents the limitations of Tenacious-Bench v0.1: it does not capture multi-turn trajectory failures (Path C territory), it uses BoW cosine for contamination rather than semantic embeddings in the initial version, and the held-out partition is small (41 tasks) which limits statistical power for the ablation tests.

---

## Where Tenacious-Bench Disagrees

**Disagreement: the Gebru "Uses" section recommendation to document intended uses is insufficient for evaluation benchmarks.**

Gebru et al. recommend documenting "what the dataset should and should not be used for." For training datasets, this is straightforward: document the task type, the domain, and the known biases. For evaluation benchmarks, the "Uses" section needs an additional dimension that neither paper addresses: **the benchmark's discriminative validity** — does it actually measure what it claims to measure?

Tenacious-Bench claims to measure Tenacious-specific B2B sales agent behavior. But the rubric is authored by one person (the trainee) based on a style guide and probe library that are themselves authored artifacts. There is no external ground truth. A benchmark that measures "compliance with the trainee's interpretation of the style guide" is not the same as a benchmark that measures "effective B2B sales agent behavior."

**Evidence from our data:** During the inter-rater agreement exercise, F5 (Thread/Data Integrity) achieved only 75% family agreement (below the 80% trigger threshold). The rubric revision required to bring F5 above threshold changed the definition of "not interested" variants — meaning the benchmark's measurement of F5 behavior changed between v0.1 and the revised rubric. This is a discriminative validity problem: the benchmark was measuring the trainee's initial interpretation, not a stable construct.

**Conclusion:** Evaluation benchmarks need a "Discriminative Validity" subsection in the Uses section that documents: (1) what construct the benchmark claims to measure, (2) what evidence supports that claim, and (3) what the known threats to validity are. Tenacious-Bench v0.2 should add this subsection to `datasheet.md`.

---

## Application to Tenacious-Bench Datasheet

The Pushkarna finding on documentation completeness and model performance directly motivates the microscopic-level metadata in every task record: `source_mode`, `difficulty`, `probe_ids`, `judge_scores`, and `judge_reasoning` are all present so that a fine-tuner can filter the training partition by quality score, source mode, or failure family without re-reading the full datasheet.

One Gebru recommendation we have not yet implemented is **maintenance documentation**: who will update the benchmark when Tenacious's style guide changes, and what the versioning policy is. Tenacious-Bench v0.2 should add a Maintenance section to `datasheet.md` with a versioning policy and a contact for style-guide updates.
