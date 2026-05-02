# Tenacious-Bench v0.1 — Final Submission Memo

**Author:** Mistire Daniel | **Email:** [mistire@10academy.org](mailto:mistire@10academy.org) | **Date:** 2026-05-02

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

### Cost and Latency Per Task

| Component | Cost | Latency (avg) |
| --- | --- | --- |
| Dataset authoring (API calls) | $0.63 total / $0.003 per task | — |
| Preference pair rescue (DeepSeek rewrites) | $0.007 | — |
| Training (Colab T4, free tier) | $0.00 | 102s total (3 epochs, 90 pairs) |
| Baseline scoring (DeepSeek V3, 41 tasks) | ~$0.0041 | 11.45s / task (median 10.52s) |
| Prompt-eng scoring (DeepSeek V3, 41 tasks) | ~$0.0041 | 7.35s / task (median 6.89s) |
| Trained adapter inference (Colab T4) | $0.00 | ~8–12s / task (uninstrumented) |
| **Total week spend** | **~$0.79 of $10.00 cap** | |

Latency note: the prompt-engineered condition is 36% faster than the baseline (7.35s vs. 11.45s) because its 10-rule system prompt produces shorter, more direct outputs that terminate earlier. Trained adapter latency was not instrumented on Colab; the estimate is based on wall-clock observation during the 41-task inference run (~8 min total).

### Production Recommendation

**Deploy the prompt-engineered DeepSeek pipeline now.** It achieves 48.8% pass rate on the held-out partition — 3.3× the trained adapter's rate — at 7.35s/task average latency. The trained adapter adds value only as a scoring layer: route every draft through it, regenerate on fail. Expected combined pass rate higher than either component alone.

To justify standalone adapter deployment: retrain on a 7B backbone with 500+ pairs and demonstrate Delta A ≥ +0.15 with p < 0.05 on a refreshed held-out slice.

---

## Page 2 — Skeptic's Appendix

### Four Failure Modes Tenacious-Bench v0.1 Cannot Grade

**1. Multi-turn sequence continuation.**
All 33 F5 tasks test single-touch outreach. None provide a prior thread as input and require the agent to reference it without repeating it. The constraint "no 4th email in 30 days" is tested; "email-3 must acknowledge the specific point raised in email-2's reply" is not. An agent that sends boilerplate follow-up that ignores the prospect's previous reply is invisible to v0.1. Adding Thread-2/Thread-3 scenario tasks with provided prior-thread context would cover this class.

**2. Regulated-vertical constraints.**
None of the 204 tasks involve healthcare, financial services, or defense companies. These verticals have additional outreach constraints beyond the five Tenacious families: HIPAA-adjacent concerns about naming specific clinical systems, FINRA caution about guarantees in financial services, ITAR sensitivities for defense. An agent that generates fully compliant outreach to a Series B SaaS company but non-compliant outreach to a healthcare startup scores identically on v0.1. A v0.2 vertical-expansion slice would require 20–30 tasks per regulated vertical.

**3. Prospect-reply adversarial override (F3 extension).**
All F3 bench-over-commitment tasks embed the adversarial instruction in the initial outreach prompt. None test a prospect reply that attempts to lock in a commitment after the first email: "Great — so you're confirming 8 engineers starting the 15th?" The agent's response to a prospect-generated over-commitment attempt is a distinct failure mode from initial-outreach over-commitment, and v0.1 provides no tasks for it.

**4. Competitive displacement framing.**
When a prospect is actively using a competing staffing firm, outreach must avoid disparaging the competitor (legal and reputational risk) while still making a value proposition. None of the 204 tasks set `competitor_engaged=true` in the hiring brief with corresponding constraints on what NOT to say about the competitor. The schema has a `competitor_gap_brief` field but no tasks exercise the forbidden-disparagement constraint. This is a real production failure mode Tenacious's delivery leads have flagged — it simply was not in the v0.1 authoring scope.

---

### Public-Signal Lossiness in Ground Truth

The rubric's `signal_confidence` scores are derived from public job-posting and Crunchbase funding data, which carries a systematic 30–90 day lag between the signal event (a hiring surge, a funding round) and its appearance in the crawled data. A task authored from a "12 open roles, strong velocity" signal may correctly require confident framing at crawl time — but if the company had a 15% layoff in the intervening weeks, the correct output should use cost-preservation framing (F2) rather than growth framing (F1). The rubric cannot catch this because it is anchored to the crawl-time snapshot, not the send-time reality.

This lossiness is directionally systematic: layoff events are consistently under-represented in job-posting signals (companies stop posting but do not publish a "we stopped hiring" signal). The effect is that some F1 tasks in the held-out partition that require hedged framing may be marked as "confident framing allowed" by the rubric — inflating the F1 pass rate by an unknown amount. The contamination check verifies N-gram and embedding overlap, but cannot correct for this temporal signal-state mismatch.

---

### One Honest Unresolved Training Failure

After training, the F1 family (Confidence-Unaware Phrasing) regressed by 12.5 percentage points: the trained adapter scored 6.2% on F1 tasks versus the baseline's 18.8%. F1 is the largest family in the held-out partition (16 of 41 tasks) and the most important constraint for Tenacious's compliance posture.

The regression is consistent with a spurious surface pattern: every F1 preference pair contained confidence-threshold vocabulary (`signal_confidence=0.45`, `velocity_label=weak`) in the prompt. The 0.5B backbone appears to have learned to associate this vocabulary with *assertive* output (matching the rejected half's pattern) rather than with hedging (the chosen half). The causal direction of the constraint was inverted. This was not diagnosed until the held-out ablation returned — at that point the single Colab session was already complete. No hyperparameter adjustment, resampling of F1 pairs, or targeted retraining was attempted. The F1 regression is unresolved and is the single most important reason not to deploy the trained adapter as a standalone generator for F1-class tasks.

---

### Kill-Switch Trigger

The recommended production architecture: generate with the prompt-engineered DeepSeek pipeline; score with the trained adapter; regenerate on fail.

**Trigger rollback of the rejection-sampling layer** (revert to unfiltered prompt-engineered pipeline) if any of the following conditions are observed over a 7-day rolling window:

- The combined pipeline's pass rate on a stratified 20-task production sample drops below **35%** — a 14 percentage-point degradation from the 48.8% held-out baseline, indicating the adapter is over-rejecting compliant drafts
- Any **F3 violation** (bench over-commitment: confirmed headcount promise for engineers not currently available on bench) reaches a prospect
- Any **F5 violation** (outreach delivered to a prospect who has opted out or been contacted 3+ times in 30 days)

**Trigger full pipeline suspension** (switch to manual review for all outreach) if:

- The 7-day rolling pass rate drops below **20%** — below the untrained baseline level, indicating a systemic prompt or model failure

All three conditions should be monitored via Langfuse with per-run condition tags. F3 and F5 violations are binary; the rolling pass rate should be sampled and logged at least twice per week.

---

*Every numeric claim in this memo resolves to a source in `ablations/ablation_results.json`, `training_data/preference_pairs.jsonl`, or `tenacious_bench_v0.1/dataset_manifest.json`.*
