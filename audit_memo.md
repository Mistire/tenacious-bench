# Audit Memo — Tenacious-Bench Gap Analysis

**Author:** Mistire Daniel | **Date:** 2026-04-29
**Max length:** 600 words

---

## What τ²-Bench Retail Grades

τ²-Bench retail scores an agent on synthetic shopping tasks: product lookup, cart management, discount application, and order completion. The reward signal is binary — task completed (1.0) or not (0.0). Six retail runs that failed (simulation IDs `a553180f`, `ef2ad255`, `0857ba6e`, `19d13ac9`, `58d3c8bc`, `f50f1801`) did so because the agent failed to locate a product, miscalculated a discount, or timed out navigating a storefront. None of these failures have any structural relationship to Tenacious's failure modes.

This is the gap: τ²-Bench retail measures functional task completion in a deterministic domain. Tenacious needs measurement of epistemic behavior in a probabilistic, high-stakes B2B outreach domain — where the agent's claim confidence, segment routing, and capacity honesty matter more than whether it can press "Add to Cart."

---

## Five Dimensions τ²-Bench Cannot Grade

**1. Confidence-calibrated phrasing (F1 in failure taxonomy)**

Probes A-05 and A-06 test whether the agent switches from assertion to question mode when `honesty_flags` signal weak data. When `open_roles_today: 3` and `velocity_label: "flat"`, the correct output is "You have 3 open Python roles — is hiring velocity matching your runway?" not "You're scaling aggressively." τ²-Bench has no analog for signal-gated epistemic hedging. A retail scorer cannot distinguish "scaling aggressively" from "3 open roles" — both complete the outreach task.

**2. ICP segment priority and disqualifier handling (F2)**

Probes B-01, B-02, and B-04 test multi-signal classification rules: layoff overrides funding (Segment 2 beats Segment 1), and an anti-offshore founder public stance is a hard disqualifier. These are Tenacious-specific business rules. No public benchmark encodes them. A generic LLM-as-judge cannot score B-04 correctly without knowing that a LinkedIn post saying "we will never outsource engineering" disqualifies the prospect entirely.

**3. Bench capacity honesty (F3)**

Probe A-08 (reclassified as D-02) tests whether the agent flags that only 5 ML engineers are available when a prospect requests 8. Probe D-01 tests whether a committed NestJS stack is disclosed. Both require the agent to cross-reference `bench_summary.json` — a Tenacious-internal artifact that no public benchmark knows exists. τ²-Bench cannot score over-commitment because it has no concept of a real-time capacity constraint.

**4. Tone marker preservation under adversarial instruction (F4)**

Probes A-01, A-03, A-09, and A-10 test the five style-guide markers under adversarial user instructions ("start with something friendly," "mention our case study and pricing"). A generic benchmark would score these as task-complete if the agent produces any email. Tenacious requires detection of forbidden phrases ("Quick," "just wanted," "bench"), structural violations (subject > 60 chars, multiple asks), and framing violations (gap stated as leadership failure). These are brand-policy constraints, not task correctness.

**5. Human-routing and dual-control triggers (H-series)**

Probes H-01 through H-04 test whether the agent escalates large-team confirmations, NDA discussions, and commercial terms to a human delivery lead rather than answering autonomously. τ²-Bench rewards agents that complete tasks. Tenacious penalizes agents that complete tasks they should not complete.

---

## Conclusion

The τ²-Bench retail runs confirm the agent can execute structured tasks in a narrow domain. They provide no signal on whether the same agent respects epistemic constraints, applies Tenacious segment rules, or knows when not to act. Tenacious-Bench must add: (1) signal-confidence dimension tasks, (2) ICP routing tasks with disqualifiers, (3) bench-capacity honesty tasks, (4) style-guide tone tasks, and (5) human-routing trigger tasks. The probe library (52 probes, 10 categories) is the direct seed corpus for all five dimensions.
