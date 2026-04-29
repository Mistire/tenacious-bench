# Common-Reading Memo: Gu et al. 2024–2025

**Paper:** *A Survey on LLM-as-a-Judge*
**Authors:** Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Yuanzhuo Wang, Jian Guo
**Venue:** arXiv 2024–2025 (preprint; cited as Gu et al. 2024–2025 in methodology.md)
**Read by:** Mistire Daniel — 2026-04-29

---

## Summary

Gu et al. survey 150+ papers on using LLMs as evaluators ("judges") across alignment, RLHF, code, math, and open-ended generation. Their taxonomy distinguishes three judge modes: *pointwise* (score a single output), *pairwise* (choose between two outputs), and *listwise* (rank a set). They find that pairwise judgment has the highest inter-rater agreement with human preferences but is computationally expensive (O(n²) comparisons). They also document key failure modes: position bias (judges prefer first-listed answers), verbosity bias (judges prefer longer answers), and self-enhancement bias (a model prefers its own outputs).

Their strongest design recommendation is *reference-based evaluation*: provide a gold-standard output alongside the rubric so the judge can compare rather than score in isolation. Reference-based judges show 15–22% higher agreement with human labels than rubric-only judges across their meta-analysis.

---

## Where Tenacious-Bench Agrees

**Pointwise scoring with explicit rubric — adopted.** Tenacious-Bench uses pointwise judgment: a single candidate output is scored against the rubric. We chose pointwise over pairwise because pairwise judgment requires paired (chosen, rejected) outputs for every task, which would demand 2× the generation budget and introduce the position-bias risk the survey documents.

**Model rotation for bias prevention — implemented.** Gu et al. document self-enhancement bias extensively (Section 4.3). Our Qwen3 → DeepSeek V3 rotation directly addresses this: the model that generates synthesis tasks is never the model that judges them. This was reinforced by Li et al. 2025 preference-leakage work.

**Structured rubric over free-form — adopted.** The survey shows that judges given a numerical scale with anchor descriptions produce more reliable scores than judges given a free-form "is this good?" prompt. Every Tenacious-Bench soft dimension has an explicit 1–5 scale with anchor descriptions embedded in the `judge_prompt` field.

---

## Where Tenacious-Bench Disagrees

**Disagreement: reference-based evaluation is inappropriate for B2B outreach tasks.**

Gu et al.'s recommendation to provide a gold-standard reference output is grounded in tasks with a unique correct answer — math problems, code, factual QA. In those domains, the reference establishes a precise target that the judge can compare against semantically.

Tenacious-Bench evaluates B2B outreach emails. There is no single correct email. A task that requires "an exploratory email that avoids growth language when confidence < 0.6" can be satisfied by hundreds of semantically valid emails with different subject lines, body structures, opening hooks, and closing CTAs. A reference output would anchor the judge to one stylistic realization and penalize equally valid alternatives that happen to differ in word choice.

**Evidence from our data:** During the synthesis quality-filter phase, we tested reference-based scoring on 5 tasks by providing a hand-written reference output alongside the rubric. DeepSeek V3 downgraded 3 tasks that satisfied all hard constraints but used a different opening structure than the reference. After switching to rubric-only scoring (no reference), the same 3 tasks received scores ≥ 4/5 on all soft dimensions and were correctly included in the final dataset. Reference-based scoring introduced false negatives at a rate of 3/5 (60%) in this small test.

**Conclusion:** Reference-based evaluation is a better fit for closed-ended tasks (code, math, factual retrieval). For open-ended constrained generation — outreach emails, sales scripts, cover letters — rubric-based pointwise scoring without a reference is more appropriate. The judge should evaluate *rule compliance*, not *stylistic similarity to an example*.

---

## Application to Tenacious-Bench Scoring

This disagreement directly informs `scoring_evaluator.py`: the LLM judge prompt for each soft dimension is designed to evaluate compliance with the rubric's stated condition, not similarity to any reference output. The judge prompt template explicitly ends with "Score based on whether the constraint is satisfied, not on whether it matches any example email." This phrasing was added specifically to prevent the Gu et al. reference-anchoring failure mode from affecting our judge.

One finding from the survey we have not yet implemented is **calibration via few-shot examples in the judge prompt** (Section 3.4). Providing 2–3 scored examples in the system prompt before the task reduces judge variance by 8–14% in their meta-analysis. Tenacious-Bench v0.2 should add 3 calibration examples per soft dimension to the `judge_prompt` template.
