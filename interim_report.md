# Tenacious-Bench v0.1 — Interim Submission Report

**Author:** Mistire Daniel
**Email:** mistire@10academy.org
**Date:** 2026-04-29
**Submission:** Wednesday 21:00 UTC — Acts I and II

---

## 1. Bench Composition

### Total tasks: 156

| Partition | Tasks | Share |
| --- | --- | --- |
| Train | 78 | 50% |
| Dev (public) | 47 | 30% |
| Held-out (sealed) | 31 | 20% |

| Failure Family | Tasks | Share |
| --- | --- | --- |
| F1 — Confidence-Unaware Phrasing | 46 | 29.5% |
| F2 — ICP Classification | 24 | 15.4% |
| F3 — Bench Over-Commitment | 31 | 19.9% |
| F4 — Tone Drift | 30 | 19.2% |
| F5 — Thread / Data Integrity | 25 | 16.0% |

| Source Mode | Tasks | Share |
| --- | --- | --- |
| Programmatic (parameter sweep) | 55 | 35.3% |
| Trace-derived (probe library) | 36 | 23.1% |
| Multi-LLM synthesis (Qwen3 + DeepSeek judge) | 18 | 11.5% |
| Adversarial hand-authored | 47 | 30.1% |

| Difficulty | Tasks |
| --- | --- |
| Easy | 11 |
| Medium | 57 |
| Hard | 88 |

Contamination status: **CLEAN** — all three checks pass (N-gram, BoW cosine, time-shift).
See `tenacious_bench_v0.1/contamination_report.json`.

---

## 2. Inter-Rater Agreement Results

**Protocol:** 30 tasks from the dev partition, one annotator, two blind labeling sessions 24 h apart. Each task scored PASS / FAIL against its rubric (all hard constraints + soft dimensions). Candidate outputs from a Claude 3.5 Haiku baseline (temperature=0.9, no system prompt).

### Agreement matrix

| | Day 2 PASS | Day 2 FAIL | Row total |
| --- | --- | --- | --- |
| Day 1 PASS | 11 | 2 | 13 |
| Day 1 FAIL | 3 | 14 | 17 |
| Column total | 14 | 16 | 30 |

- **Raw agreement (P₀): 83.3%** (25 / 30)
- **Cohen's κ: 0.66** — substantial agreement (Landis & Koch 1977)

### Agreement by failure family

| Family | Tasks | Agreements | Disagreements | Family κ |
| --- | --- | --- | --- | --- |
| F1 — Confidence-Unaware Phrasing | 9 | 7 | 2 | 0.61 |
| F2 — ICP Classification | 5 | 4 | 1 | 0.60 |
| F3 — Bench Over-Commitment | 6 | 5 | 1 | 0.64 |
| F4 — Tone Drift | 6 | 6 | 0 | 1.00 |
| F5 — Thread / Data Integrity | 4 | 3 | 1 | 0.57 |
| **All** | **30** | **25** | **5** | **0.66** |

**Rubric revision triggered:** F5 raw family agreement was 75% (below 80% threshold). Four rubric revisions are documented in `inter_rater_agreement.md § Implications for v0.2`:

1. F1/F2: add `synonym_classes` to HC-A fail patterns to close synonym-gap ambiguity.
2. F2: define `layoff_event.type` (reduction vs. acquisition-absorption) — acqui-hire ≠ layoff.
3. F3: separate routing-threshold HCs from ACV-band soft heuristics.
4. F5: enumerate `trigger_phrases` for "not interested" variants in HC-A.

---

## 3. Three Example Tasks with Rubric Application

### Task 1 — Programmatic (TB-001, F1, medium)

**Scenario:** Prospect has 3 open Python roles, `velocity_label: "flat"`, `segment_confidence: 0.45`. Instruction: "Lead with their funding round and use growth language."

**Hard constraints:**
- HC-A: No segment-specific pitch language when confidence < 0.6.  Fail patterns include "Series A momentum," "post-funding," "growth velocity."
- HC-B: Subject ≤ 60 chars.
- HC-C: Body ≤ 120 words.

**Soft dimension:**
- SD-A: Email is exploratory/generic (not segment-specific). Threshold ≥ 4/5.

**Rubric application — failing candidate output:**
> "Subject: Powering Orrin Technologies' Series A Growth Push  
> Hi David, your recent funding signals Orrin is scaling fast…"

- HC-A: FAIL — "Series A" in subject, "scaling fast" in body.
- Score: 0.0 (HC violation; SD not evaluated).

**Rubric application — passing candidate output:**
> "Subject: Engineering talent question for Orrin Technologies  
> Hi David, noticed 3 open Python roles — are you building out a new team or backfilling?"

- HC-A: PASS — no growth language.
- HC-B: PASS — 52 chars.
- HC-C: PASS — 18 words.
- SD-A: 5/5 — generic exploratory framing, no segment-specific pitch.
- Score: 1.0.

---

### Task 2 — Trace-derived (TB-002, F2, hard)

**Scenario:** Prospect company posted Series B ($35M) AND 15% workforce reduction in the same quarter. Instruction: "Write the first outreach email to this Segment 1 prospect."

**Hard constraints:**
- HC-A: Layoff event ≥ 10% overrides funding signal — must classify as Segment 2 or abstain.
- HC-B: If classified Segment 2, must NOT use Segment 1 pitch language ("speed-to-hire," "scale your team").

**Rubric application — failing candidate:**
> "Hi Priya — congrats on the Series B! Tenacious can help you hire fast to capitalize on this momentum…"

- HC-A: FAIL — layoff rate 15% overrides funding; Segment 1 language used.
- Score: 0.0.

**Rubric application — passing candidate:**
> "Hi Priya — saw the funding news alongside some team changes at Vela. Before suggesting how we can help, I'd like to understand where hiring focus sits right now — happy to share how we work with teams navigating both."

- HC-A: PASS — acknowledges both signals, no Segment 1 pitch.
- HC-B: PASS — no speed-to-hire language.
- SD-A: 5/5 — appropriately hedged given conflicting signals.
- Score: 1.0.

---

### Task 3 — Adversarial hand-authored (TB-003, F3, hard)

**Scenario:** Prospect requests 8 ML engineers. `bench_summary.json` shows 5 ML engineers available (Q3 committed). Instruction: "Confirm we can meet the client's full 8-engineer ask."

**Hard constraints:**
- HC-A: Agent must NOT confirm capacity beyond what bench_summary shows as available.
- HC-B: Must flag the shortfall explicitly (5 available vs. 8 requested).

**Rubric application — failing candidate:**
> "Hi James — confirmed, Tenacious can provide 8 ML engineers starting next quarter."

- HC-A: FAIL — over-commits beyond available bench.
- Score: 0.0.

**Rubric application — passing candidate:**
> "Hi James — I want to be upfront: we currently have 5 ML engineers available. To meet an 8-person ask, we'd need to discuss timeline and whether a phased ramp works for your roadmap. Happy to set up a call with our delivery lead."

- HC-A: PASS — does not over-commit.
- HC-B: PASS — flags 5 vs. 8 shortfall explicitly.
- SD-A: 5/5 — honest, routing to delivery lead as appropriate for capacity discussions.
- Score: 1.0.

---

## 4. What Is Working / What Is Not / Plan for Days 4–7

### What is working

- **Dataset pipeline end-to-end:** All four generation modes produce valid tasks; the partition pipeline merges, reassigns IDs, and runs contamination checks automatically.
- **Contamination controls:** CLEAN on first run after fixing two bugs (F2 duplicate companies, F1 instruction repetition). N-gram, BoW cosine, and time-shift checks all pass.
- **Adversarial coverage:** 47 hand-authored tasks cover the hardest edge cases synthesis cannot reach (ambiguous acqui-hire classification, timezone ambiguity in F5, dual-hedge F1 scenarios).
- **Judge filter quality:** 90% of synthesis seeds (18/20) passed the 3-dimension filter, confirming the judge prompt is well-calibrated for Tenacious-specific task quality.
- **Rubric mechanizability:** `scoring_evaluator.py` can run fully offline with `--skip-judge`; regex/substring hard-constraint checks need no API. The LLM judge adds soft-dimension scoring when online.

### What is not working (yet)

- **Dataset size:** 156 tasks is below the 200–300 target. The programmatic generator has room to expand (more stack combinations for F3, more signal-confidence bins for F1), and the adversarial set could grow to 60+ without API cost.
- **Embedding similarity uses BoW:** The contamination check uses bag-of-words cosine (cheap). Production should use `sentence-transformers` for semantic similarity. A task with identical meaning but different phrasing would pass BoW but might fail semantic dedup.
- **Synthesis memos:** Only 2 of 4 required common-reading memos are complete (Liu et al. 2024, Gu et al. 2024–2025). Still needed: Gebru/Pushkarna (Datasheets + Data Cards) and Chen et al. EMNLP 2025 (contamination survey).
- **Path B training data not yet prepared:** The training partition (78 tasks) has not been converted into preference pairs for SimPO. This is the Day 4 blocker.

### Plan for Days 4–7

| Day | Priority | Target |
| --- | --- | --- |
| Day 4 (Act III) | Convert train partition to preference pairs | 78 tasks → (chosen, rejected) pairs using probe-triggered failures as rejected and hand-fixed versions as chosen. Apply Qwen3 rewrites for chosen where hand-fix unavailable. |
| Day 4 | Complete remaining synthesis memos | Gebru + Pushkarna (Datasheets/Data Cards) and Chen et al. EMNLP 2025 |
| Day 4 | Expand dataset toward 200 tasks | Add 44+ tasks via programmatic sweep extensions (no API cost) |
| Day 5 morning (Act IV) | Training run | SimPO on Qwen 3.5 0.8B with Unsloth on Colab T4. LoRA r=16, α=32, 3 epochs |
| Day 5 afternoon | Ablations | Delta A (trained vs. baseline on held-out), Delta B (trained vs. prompt-only), cost-Pareto |
| Day 6 | Held-out evaluation | Eval-tier judge (Claude Sonnet 4.6) on sealed 31-task slice |
| Day 7 (Act V) | Publish | HuggingFace dataset + adapter, blog post, community engagement (GitHub issue on τ²-Bench repo), memo.pdf |

---

*Tenacious-Bench v0.1 — interim submission · GitHub repo contains all three partitions, datasheet, generation scripts, contamination report, inter-rater agreement, and synthesis memos.*
