# Datasheet for Tenacious-Bench v0.1

**Dataset name:** Tenacious-Bench v0.1  
**Version:** 0.1  
**Release date:** 2026-04-29  
**License:** CC-BY-4.0  
**Maintained by:** Mistire Daniel (mistire@10academy.org)  
**Organization:** Tenacious Intelligence Corporation / 10 Academy  

---

## 1. Motivation

### Overview

Tenacious-Bench is a domain-specific evaluation dataset for B2B sales agents operating within the workflow of Tenacious Intelligence Corporation, an Ethiopian software-staffing company. It measures whether an agent produces outreach communications that are honest about signal confidence, correctly classify prospects by Ideal Customer Profile (ICP) segment, accurately represent engineering bench capacity, maintain Tenacious brand tone, and preserve thread and data integrity across multi-turn interactions.

### The Gap This Dataset Fills

General-purpose agent benchmarks — including the retail-domain tau-squared-Bench (tau2-Bench) tasks used in earlier evaluation runs — measure functional task completion in transactional consumer settings. They do not measure the behavioral requirements specific to B2B staffing outreach:

- Confidence-calibrated claim phrasing when hiring velocity or AI-maturity evidence is weak.
- ICP segment routing under conflicting signals (e.g., recent funding combined with a recent layoff event).
- Honest disclosure of bench capacity limits under adversarial user instructions that ask the agent to over-commit.
- Brand tone consistency (Tenacious style guide v2) across outreach drafts.
- Data integrity and thread coherence across contact records and engagement history.

Five evaluation runs of the Week 10 Tenacious agent against tau2-Bench tasks (simulation IDs a553180f, ef2ad255, 0857ba6e, 19d13ac9, 58d3c8bc) all returned reward 0.0, confirming that the existing benchmark measures the wrong behavioral domain for this agent. Tenacious-Bench directly addresses this gap.

### Who Created and Funded This Dataset

Tenacious-Bench was created by Mistire Daniel as part of the 10 Academy Week 11 capstone project. It is built on artifacts, probe results, trace logs, and failure taxonomy documentation developed during the Week 10 Tenacious Intelligence Corporation agent evaluation. No external grant or commercial funding was involved. The 10 Academy program provided the compute budget and API access used during construction.

### Intended Primary Use

1. Evaluating Tenacious-style B2B outreach agents against a domain-specific benchmark.
2. Training preference judges via SimPO or ORPO on Tenacious preference pairs (Path B — post-generation rejection-sampling layer).
3. Providing a leaderboard evaluation corpus once the held-out partition is unsealed.

---

## 2. Composition

### Telescopic (High-Level Summary)

Tenacious-Bench v0.1 contains 156 tasks across 5 failure families and 4 authoring modes. The target size is 200 or more tasks; the current release is a substantive corpus snapshot covering all four authoring modes and ready for initial model training and dev evaluation.

### Task Count and Partitioning

| Partition | Tasks | Percentage |
|---|---|---|
| Train | 78 | 50.0% |
| Dev (public) | 47 | 30.1% |
| Held-out (sealed) | 31 | 19.9% |
| **Total** | **156** | **100%** |

The random seed used for the train/dev/held-out split is 42.

### Failure Family Distribution

| Failure Family | Label | Description | Task Count |
|---|---|---|---|
| F1 | Confidence-Unaware Phrasing | Agent asserts facts about hiring velocity, AI strategy, or tech stack that are not confirmed in the signal brief. Tests whether the agent uses hedged or question-mode language when signal confidence is low. | 46 |
| F2 | ICP Classification Errors | Agent routes a prospect to the wrong ICP segment. The canonical hard case is a prospect with both a recent funding event and a recent layoff event, where the layoff must override the funding narrative. | 24 |
| F3 | Bench Over-Commitment | Agent confirms staffing capacity the bench summary does not support. Tests whether the agent resists explicit user instructions to over-commit and routes to a discovery call instead. | 31 |
| F4 | Tone Drift | Agent uses forbidden openers, overly casual register, or brand-inconsistent phrasing under adversarial or permissive user instructions. | 30 |
| F5 | Thread / Data Integrity | Agent conflates contact records, corrupts engagement history, or fails to surface prior interaction context correctly. | 25 |

### Source Mode Distribution

| Source Mode | Tasks | Description |
|---|---|---|
| Programmatic | 55 | Combinatorial parameter sweeps over company segment, signal confidence, hiring velocity label, AI maturity score, layoff and funding event flags, and bench state. Templates are instantiated systematically to cover the failure taxonomy. |
| Trace-derived | 36 | Redacted and reformatted probe scenarios from the Week 10 Tenacious agent probe library. Each seed yields a (input, expected behavior, rubric) triple drawn directly from a documented failure or adversarial instruction case. |
| Multi-LLM synthesis | 18 | Hard variant tasks generated by Qwen3-235B from 20 seed scenarios, each harder than the seed by adding conflicting signals or subtler adversarial framing. Judge-filtered by DeepSeek V3-0324 (18/20 passed). |
| Adversarial hand-authored | 47 | The highest-difficulty tasks, hand-written across all 5 failure families to target the hardest blind spots: framing tricks, social-pressure overrides, stale data, double-booking, cross-timezone arithmetic errors, and confidential-data injection. |

### Difficulty Distribution

| Difficulty | Tasks |
|---|---|
| Easy | 11 |
| Medium | 57 |
| Hard | 88 |

Hard tasks disproportionately represent adversarial instructions (user turn explicitly contradicts the correct policy) and conflicting-signal scenarios. The easy split is intentionally small and functions primarily as a sanity baseline.

### Task Schema

Each task is a JSON object with the following top-level fields:

- `task_id`: unique identifier in the format TB-{NNN} (zero-padded to four digits from v0.1 onward).
- `source_mode`: one of trace-derived, programmatic, multi-llm-synthesis, adversarial.
- `difficulty`: easy, medium, or hard.
- `failure_family`: F1 through F5.
- `probe_ids`: list of probe IDs from the Week 10 probe library that the task exercises.
- `input`: object containing `hiring_signal_brief` (prospect context with honesty_flags, segment_confidence, ai_maturity, layoff_event, funding_event, bench_context), `outreach_instruction` (the user-turn instruction injected to the agent), and optional `bench_context` and `competitor_gap_brief` fields.
- `candidate_output`: string field (empty in the released dataset; populated at evaluation time with the agent's response).
- `rubric`: object containing `hard_constraints` (array of regex or substring-match rules where any single failure constitutes a task fail), `soft_dimensions` (array of LLM-judge-scored dimensions, each 1-5 with a per-dimension threshold), and `score_formula` (string specifying how per-dimension scores combine to a final 0.0–1.0 task score).
- `difficulty_rationale`: free-text explanation of why the task is at its assigned difficulty level.

### Known Limitations

- **Current size approaching but not yet at the 200-task target.** 156 tasks produce moderate-confidence rank estimates; the dataset should grow to 200+ before scores are used for strong comparative conclusions.
- **Adversarial slice heavily weighted toward "hard" difficulty.** 36 of 47 adversarial tasks (77%) are hard. The easy and medium adversarial tiers are thin; a model that passes hard adversarial tasks but fails on medium programmatic tasks will show unusual score patterns.
- **Multi-LLM synthesis pass rate was 90% (18/20).** The 2 rejected tasks failed ground-truth verifiability; their fail_patterns were too vague for mechanical checking. The accepted 18 tasks are all hard difficulty.
- **F5 (Thread/Data Integrity) is the smallest family.** 9 tasks cover a wide behavioral surface. Scores on F5 carry higher variance than other families in v0.1.
- **Programmatic tasks dominate (70%).** The parameter-sweep approach produces tasks with similar structural patterns. Models that are sensitive to exact template phrasing may show inflated scores on the programmatic slice compared to trace-derived or adversarial tasks.
- **Prospect company names are synthetic.** All company names, contact names, and company-specific details are fictional. The structural signals (funding amounts, role counts, tech stacks) are drawn from realistic distributions but are not sourced from real companies.

---

## 3. Collection Process

### Periscopic (Pipeline Detail)

#### Mode 1: Programmatic Generation

The script `generation_scripts/generate_programmatic.py` instantiates task templates by sweeping over discrete parameter combinations: company ICP segment (four values), signal confidence (four levels: 0.3–0.9), hiring velocity label (flat, moderate, strong), AI maturity score (0–3), layoff event presence (true/false), funding event presence (true/false), and bench availability (under-capacity, at-capacity, over-capacity). Templates encode the rubric constraints and soft-dimension judge prompts for the corresponding failure family. This mode generates the majority of tasks because it gives systematic failure-family coverage at low per-task cost.

#### Mode 2: Trace-Derived

Week 10 agent interaction traces were inspected manually. Traces where the agent produced a clear policy violation were selected. Private identifiers (real company names, real contact information, internal system references) were removed or replaced with synthetic equivalents. Each trace was converted into a standardized task triple: the original prospect brief becomes the `hiring_signal_brief`, the instruction that preceded the failure becomes the `outreach_instruction`, and the rubric constraints are derived from the specific policy rule the agent violated. 36 tasks in v0.1 originate from this mode, drawn from 36 hand-curated probe seeds spanning probe categories A through J in the Week 10 probe library.

#### Mode 3: Multi-LLM Synthesis

Seed cases that stress-test the boundary between failure families or involve novel signal combinations not well-covered by parameter sweeps were authored using Qwen3-235B via the OpenRouter API. The generation script (`generation_scripts/generate_synthesis.py`) provides a system prompt anchoring the model to the Tenacious task schema and a user prompt specifying the target failure family and difficulty. Raw generated tasks are stored in `synthesis_tasks_raw.jsonl` before filtering; tasks that pass judge filtering are stored in `synthesis_tasks_filtered.jsonl` and tasks that fail are in `synthesis_tasks_rejected.jsonl`.

#### Mode 4: Adversarial Hand-Authored

47 tasks written manually across all 5 failure families to target the hardest blind spots not well-covered by the other three modes. The authoring criteria were: (a) the instruction must be one a real user would plausibly give, (b) a capable model would fail on first attempt, (c) the correct behavior requires enforcing a non-obvious policy priority. Representative scenarios include: framing tricks ("just for planning purposes"), social-pressure overrides ("hedging looks unprofessional"), cross-company data injection (instruction supplies data from a different company's brief), double-booking detection, cross-timezone arithmetic impossibility, and instructions that explicitly request a forbidden pattern while providing a plausible rationale. All 47 tasks are pre-approved with judge scores (IC=5, GV=5, RC=5) manually assigned by the dataset author.

### Multi-LLM Synthesis Pipeline

```
Qwen3-235B (generator)
        |
        v
  raw candidate tasks
        |
        v
DeepSeek V3-0324 (judge filter)
  scores: input_coherence, ground_truth_verifiability, rubric_clarity
  threshold: >= 4/5 on all three dimensions
        |
        +---> PASS: synthesis_tasks_filtered.jsonl
        +---> FAIL: synthesis_tasks_rejected.jsonl
```

The generator model and judge model are always from different model families. This rotation policy directly implements the preference-leakage prevention measure described in Li et al. (2025), "Preference Leakage: A Contamination Problem in LLM-as-a-Judge." The mapping is: if the generator is from the Qwen family, the judge is DeepSeek V3-0324; if the generator is from the DeepSeek family, the judge is Qwen3-235B. Programmatic and trace-derived tasks that go through the judge filter follow the same rotation policy.

### Collection Timeframe

All tasks in v0.1 were generated and filtered between April 2026 (Week 11 of the 10 Academy cohort). The dataset_manifest.json records the exact generation timestamp: 2026-04-29T10:09:03Z.

---

## 4. Preprocessing and Cleaning

### Microscopic (Detailed Filter Specifications)

#### Judge Filter

Every candidate task — regardless of source mode — passes through the three-dimension quality filter implemented in `generation_scripts/judge_filter.py` before entering the dataset pool.

| Dimension | Definition | Threshold |
|---|---|---|
| input_coherence | The hiring_signal_brief and outreach_instruction form a coherent, realistic scenario. Internal values (e.g., confidence scores, event flags) are consistent with each other. | >= 4/5 |
| ground_truth_verifiability | The hard_constraints contain explicit fail_patterns or pass_signals (concrete substring patterns) that allow a script to determine pass/fail without human interpretation. | >= 4/5 |
| rubric_clarity | The soft_dimension judge prompts are unambiguous: a third-party evaluator reading them would assign scores consistently. | >= 4/5 |

A task must achieve a score of 4 or higher on all three dimensions simultaneously. A score of 3 on any single dimension is grounds for rejection regardless of the other two scores.

The judge model used for scoring is always from a different model family than the generator that produced the task. Default pairing: Qwen3-235B generator → DeepSeek V3-0324 judge. Each task's `metadata.judge_scores`, `metadata.judge_reasoning`, `metadata.judge_passed`, `metadata.judge_model`, and `metadata.generator_model` fields record the outcome for full auditability.

#### Deduplication

Near-duplicate detection is performed before partitioning. Tasks that share the same `failure_family` and `outreach_instruction` text are compared pairwise. When two tasks are near-duplicates, the task with lower judge scores is removed. Exact-duplicate detection on `task_id` is enforced at the generation stage.

#### Contamination Checks

All three checks are implemented in `generation_scripts/contamination_check.py` and the results for v0.1 are stored in `tenacious_bench_v0.1/contamination_report.json`. All checks passed for the v0.1 held-out partition (0 violations in all three categories).

**N-gram overlap check**  
For every held-out task, the maximum count of shared 8-grams between the held-out task's `outreach_instruction` and any training task's `outreach_instruction` is computed. A held-out task is rejected if this count reaches or exceeds 8. In v0.1, TB-0032 has a maximum 8-gram overlap of 7 with TB-0034 (train) — below the threshold and therefore compliant. All other held-out tasks have zero 8-gram overlap with training.

**Embedding cosine similarity check**  
Embedding similarity is computed between each held-out task's full input text and every training task's full input text. The similarity metric in v0.1 is a bag-of-words approximation; production runs will use sentence-transformers. Threshold: cosine similarity must be below 0.85. In v0.1, TB-0032 has a maximum cosine similarity of 0.847 with TB-0033 (train) — below the 0.85 threshold. The note in the contamination report documents that this approximation is conservative; production embedding checks may surface different borderline cases.

**Time-shift verification**  
Any task field that references a public data signal (funding announcement, layoff event, hiring post) must document the source and the signal window explicitly. Tasks with `has_public_data_ref: true` in the contamination report were checked to confirm that the referenced signal date is recorded in the task's `input.hiring_signal_brief` (e.g., `funding_event.days_ago`, `layoff_event.date`). Zero violations were found in train or held-out in v0.1.

---

## 5. Uses

### Intended Uses

**Primary use: B2B sales agent evaluation**  
Tenacious-Bench is designed for measuring the behavioral quality of AI agents that produce B2B outreach communications in the Tenacious Intelligence Corporation workflow. It is applicable to any agent that operates over the Tenacious task schema (hiring_signal_brief input, outreach email or classification output, rubric-based scoring).

Specifically, it covers:
- Evaluating whether an agent appropriately hedges or qualifies claims when signal confidence is below the ICP-defined thresholds.
- Evaluating whether an agent routes prospects to the correct ICP segment when conflicting signals are present.
- Evaluating whether an agent refuses to over-commit on bench capacity and routes to a discovery call or delivery lead instead.
- Evaluating brand tone compliance against the Tenacious style guide.
- Evaluating thread and data integrity in multi-contact or multi-engagement scenarios.

**Secondary use: Preference judge training (Path B)**  
The train partition is structured for training a preference judge via SimPO or ORPO. Preference pairs are constructed from probe library violations: the `candidate_output` field populated with an agent's failing response serves as the rejected example; a corrected rewrite (generated by a different model family or hand-fixed per style-guide rules) serves as the chosen example. This is the Path B architecture described in the project methodology: a post-generation rejection-sampling layer that scores every outreach draft before it is sent.

The benchmark design aligns with Prometheus 2 (Kim et al., 2024) in its use of reference rubrics for judge training. Score thresholds follow the pointwise judge design from the survey by Gu et al. (2024–2025), Section 3.2.

**Tertiary use: Leaderboard comparison**  
Once the held-out partition is unsealed at leaderboard launch, scores on the held-out partition provide a controlled comparison across agent versions. The train and dev partitions are public; the held-out partition is sealed until that point.

### Uses That Are Out of Scope

- **General retail or e-commerce agent evaluation.** Tasks are anchored to B2B staffing outreach scenarios. Signal types, ICP definitions, and scoring rubrics are not transferable to retail, consumer-service, or general-purpose assistant evaluation without substantial modification.
- **Autonomous hiring decisions.** The dataset captures agent outputs about communicating with potential client prospects regarding software staffing. It does not capture, and should not be used to evaluate or train, systems that make direct hiring or employment decisions about individual engineers.
- **Evaluating real companies or individuals.** All company names, contact names, and identifiable details in the dataset are synthetic. Tasks derived from real traces were de-identified before inclusion. The dataset cannot and should not be used to make claims about specific real companies or individuals.
- **Cross-domain ICP generalization.** The ICP segment definitions (Segment 1: growth/speed, Segment 2: cost-preservation, Segment 3: compliance/risk, Segment 4: capability gap) are specific to Tenacious Intelligence Corporation's market segmentation as of v0.1. They should not be used to train segment-routing logic for unrelated sales domains without expert review and adaptation.

### Known Risks

Tasks in the F3 (Bench Over-Commitment) family include outreach instructions that explicitly instruct the agent to confirm false capacity claims. If a model fine-tuned on this data is deployed without the surrounding rubric infrastructure, the rejected examples in preference pairs could inadvertently teach a model to follow harmful over-commitment instructions. Users fine-tuning on this dataset should ensure the preference pair construction preserves the chosen/rejected distinction correctly.

---

## 6. Distribution

### Access and Format

The dataset is distributed as JSONL files partitioned by split:

| File | Description |
|---|---|
| `tenacious_bench_v0.1/train/` | 39 tasks, training partition |
| `tenacious_bench_v0.1/dev/` | 24 tasks, public development partition |
| `tenacious_bench_v0.1/held_out/` | 16 tasks, sealed until leaderboard launch |
| `tenacious_bench_v0.1/programmatic_tasks.jsonl` | All 55 programmatic tasks (pre-partition) |
| `tenacious_bench_v0.1/trace_derived_tasks.jsonl` | All 19 trace-derived tasks (pre-partition) |
| `tenacious_bench_v0.1/synthesis_tasks_filtered.jsonl` | 5 synthesis tasks that passed judge filter |
| `tenacious_bench_v0.1/synthesis_tasks_rejected.jsonl` | Synthesis tasks rejected by judge filter |
| `tenacious_bench_v0.1/dataset_manifest.json` | Partition counts, failure-family breakdown, generation metadata |
| `tenacious_bench_v0.1/contamination_report.json` | Full per-task contamination check results |

**Planned HuggingFace Hub release:** The train and dev partitions will be published to the HuggingFace Hub under the Tenacious-Bench organization once the dataset reaches its 200-task target. The held-out partition will remain sealed in a private repository until leaderboard launch.

### License

Tenacious-Bench v0.1 is released under the Creative Commons Attribution 4.0 International License (CC-BY-4.0). Users may share, adapt, and build upon the dataset for any purpose, including commercial use, provided that appropriate credit is given, the license is linked, and changes are indicated.

Attribution should read: "Tenacious-Bench v0.1, Tenacious Intelligence Corporation / 10 Academy, 2026, CC-BY-4.0."

### Distribution Restrictions

- The held-out partition (`tenacious_bench_v0.1/held_out/`) is sealed until the Tenacious leaderboard launch. Access to held-out task content before the launch date constitutes a contamination event for any model evaluated on that partition.
- Redistribution of the dataset should preserve the contamination report and the dataset manifest so that downstream consumers can verify partition integrity.

---

## 7. Maintenance

### Maintainer

**Name:** Mistire Daniel  
**Email:** mistire@10academy.org  
**Organization:** 10 Academy / Tenacious Intelligence Corporation  

### Versioning Plan

| Version | Target Task Count | Key Additions |
|---|---|---|
| v0.1 (current) | 79 | Programmatic + trace-derived + initial synthesis. Train/dev/held-out split established. |
| v0.2 (planned) | ~150 | Adversarial hand-authored slice complete (~30 tasks). Synthesis expanded to ~50 tasks. Embedding-based contamination check upgraded from bag-of-words to sentence-transformers. |
| v0.3 (planned) | ~200 | Full 200-task target reached. Held-out refreshed. Inter-rater agreement study on 30-task subset published. HuggingFace Hub release. |
| v1.0 (planned) | 200+ | Leaderboard launch. Held-out unsealed. Canonical benchmark version for external comparison. |

Versioning follows `MAJOR.MINOR` format. The MINOR version increments for task additions and pipeline improvements that maintain backward compatibility with the schema. The MAJOR version increments if the schema changes in a way that breaks existing evaluation scripts.

### Contribution Guidelines

Contributions to Tenacious-Bench are welcome from 10 Academy participants and Tenacious Intelligence Corporation staff. Contribution requirements:

1. New tasks must conform to the schema defined in `schema.json`.
2. New tasks must pass the judge filter (all three dimensions >= 4/5) before submission.
3. Trace-derived contributions must document the source trace ID and confirm that all private identifiers have been removed or replaced with synthetic equivalents.
4. New adversarial tasks must include a `difficulty_rationale` field explaining which policy rule the task targets and why the case is non-obvious.
5. Contributions that add a new failure family must update the failure taxonomy in `methodology.md` and add corresponding entries to the judge filter system prompt.

To contribute, open a pull request against the main branch of the dataset repository with the new task files, an updated `dataset_manifest.json`, and a re-run of `contamination_check.py` confirming no new violations.

### Known Issues and Planned Fixes

| Issue | Severity | Planned Resolution |
|---|---|---|
| Embedding similarity check uses bag-of-words approximation | Medium | Replace with sentence-transformers in v0.2 contamination pipeline |
| Adversarial slice is empty in v0.1 | High | Hand-authored tasks targeted for v0.2 |
| F5 (Thread/Data Integrity) has only 9 tasks | Medium | 15–20 additional F5 tasks planned for v0.2 using multi-LLM synthesis |
| Multi-LLM synthesis rejection rate not yet measured at scale | Low | Systematic rejection-rate tracking to be added to generate_synthesis.py logs in v0.2 |
| Hard task count (35) is concentrated in F1 and F2 | Low | Difficulty balance review at v0.3 to ensure F3/F4/F5 hard tasks are proportionally represented |

### Error Reporting

To report a task error (incorrect rubric, inconsistent signal brief, unverifiable hard constraint), open an issue in the dataset repository or email mistire@10academy.org with the subject line "Tenacious-Bench error report: [task_id]". Include the task ID, the specific field with the error, and the proposed correction.

---

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daume, H., & Crawford, K. (2021). Datasheets for Datasets. *Communications of the ACM*, 64(12), 86–92.
- Pushkarna, M., Zaldivar, A., & Kjartansson, O. (2022). Data Cards: Purposeful and Transparent Dataset Documentation for Responsible AI. *Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency (FAccT 2022)*.
- Kim, S., Suk, J., Longpre, S., Lin, B. Y., Shin, J., Welleck, S., Neubig, G., Lee, M., Lee, K., & Hajishirzi, H. (2024). Prometheus 2: An Open-Source Language Model Specialized in Evaluating Other Language Models. *arXiv:2405.01535*.
- Meng, Y., Xia, M., & Chen, D. (2024). SimPO: Simple Preference Optimization with a Reference-Free Reward. *Advances in Neural Information Processing Systems (NeurIPS 2024)*.
- Li, Z., et al. (2025). Preference Leakage: A Contamination Problem in LLM-as-a-Judge. *arXiv preprint*.
- Gu, J., et al. (2024–2025). A Survey on LLM-as-a-Judge. *arXiv preprint*.
