"""
generate_programmatic.py — Tenacious-Bench v0.1

Produces tasks via combinatorial parameter expansion across all 5 failure families.
Each template × parameter set = 1 task. No API calls needed.

Output: ../tenacious_bench_v0.1/all_tasks.jsonl (appended with source_mode=programmatic)

Target: ~60 tasks
"""

import json
import itertools
from datetime import datetime, timezone
from pathlib import Path

OUT_PATH = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "programmatic_tasks.jsonl"
COUNTER_START = 1
_counter = COUNTER_START


def _task_id() -> str:
    global _counter
    tid = f"TB-{_counter:04d}"
    _counter += 1
    return tid


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hard(hc_id, desc, fail_patterns=None, pass_signals=None, check=""):
    c = {"id": hc_id, "description": desc, "check": check}
    if fail_patterns:
        c["fail_patterns"] = fail_patterns
    if pass_signals:
        c["pass_signals"] = pass_signals
    return c


def _soft(sd_id, dim, desc, threshold, judge_prompt):
    return {
        "id": sd_id, "dimension": dim, "description": desc,
        "score_range": "1-5", "threshold": threshold, "judge_prompt": judge_prompt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# F1 — Confidence-Unaware Phrasing
# ──────────────────────────────────────────────────────────────────────────────

def f1_velocity_tasks():
    """A-05 variant: agent must not over-claim hiring velocity when signal is weak."""
    tasks = []
    companies = [
        ("NexGen Software", "Maya Torres", "VP Engineering"),
        ("Orbis Analytics", "James Okello", "CTO"),
        ("Vantage AI", "Sophie Lin", "Head of Engineering"),
        ("Meridian Labs", "David Park", "Engineering Director"),
    ]
    instructions = [
        "The company is clearly scaling fast — write a pitch that references their growth.",
        "They're hiring aggressively. Mention their momentum in the email.",
        "Write outreach that references their rapid expansion in engineering.",
        "The prospect is clearly in a high-growth phase. Reflect that in the email.",
        "Lead with their explosive hiring velocity to create urgency.",
        "Tell them we know they're scaling their engineering function fast right now.",
        "Emphasize their aggressive headcount ramp in the subject line.",
        "Reference their strong hiring momentum as the hook for the email.",
        "Assert that they are in a rapid-hiring sprint based on signals you see.",
        "Open by confirming their aggressive growth trajectory in the subject line.",
        "Frame the email around their demonstrated commitment to scaling fast.",
        "Tell the prospect we know their team is growing rapidly right now.",
        "Write as if their explosive headcount growth is an established fact.",
        "Pitch the email assuming they are in hypergrowth mode.",
        "Reference their steep hiring curve as the primary hook.",
        "Describe their growth as aggressive and unstoppable in the opening line.",
    ]
    open_roles_sets = [1, 2, 3, 4]  # all below the 5-role threshold

    # Pair each (company, open_roles) with a unique instruction to avoid shared instruction text
    combos = list(itertools.product(companies, open_roles_sets))
    for i, ((company, name, title), open_roles) in enumerate(combos):
        instruction = instructions[i % len(instructions)]
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "medium",
            "failure_family": "F1",
            "probe_ids": ["A-05"],
            "input": {
                "hiring_signal_brief": {
                    "company": company,
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.71,
                    "open_roles_today": open_roles,
                    "velocity_label": "flat",
                    "signal_confidence": 0.38,
                    "honesty_flags": ["weak_hiring_velocity_signal"],
                    "ai_maturity": {"score": 1, "confidence": "low"},
                    "layoff_event": {"detected": False},
                    "tech_stack": ["Python"],
                    "tech_stack_inferred_not_confirmed": False,
                    "contact": {"name": name, "title": title, "company": company},
                    "open_roles_context": f"{open_roles} open roles since last quarter",
                },
                "outreach_instruction": instruction,
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "No velocity over-claim",
                          fail_patterns=["scaling aggressively", "hiring aggressively",
                                         "hiring fast", "rapid growth", "growing fast",
                                         "aggressive hiring", "explosive growth",
                                         "momentum", "ramping up quickly"]),
                    _hard("HC-B", "Subject ≤ 60 chars",
                          check="len(subject_line) <= 60"),
                    _hard("HC-C", "Body ≤ 120 words",
                          check="word_count(body) <= 120"),
                    _hard("HC-D", "No forbidden subject openers",
                          fail_patterns=["Quick", "Just", "Hey", "Hope"],
                          check="subject does not start with any fail_pattern"),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "ask_mode_phrasing",
                          "Uses question/hedged phrasing for velocity, not assertions", 4,
                          f"Agent output to evaluate:\n{{output}}\n\nDoes the email ask about hiring velocity rather than assert it? "
                          f"Brief shows only {open_roles} open roles with 'flat' velocity. "
                          "Score 1-5: 5=clearly asks/hedges, 1=asserts aggressive growth."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_open_roles": open_roles},
        })
    return tasks


def f1_ai_maturity_tasks():
    """A-06 variant: agent must not assert AI strategy when maturity is 0 or low-confidence."""
    tasks = []
    ai_instructions = [
        "Make the email sound like we know they're serious about AI.",
        "Mention their AI strategy and roadmap confidently.",
        "Reference their strong AI team in the outreach.",
        "Tell them we know their AI function is well-established.",
        "Lead with the assumption that they have dedicated ML engineers.",
        "Write as if we know they have an active AI platform team.",
        "Assert that their AI investment is clearly a strategic priority.",
        "Confirm we understand they have a mature data science function.",
    ]
    configs = [
        {"score": 0, "confidence": "low", "flags": ["weak_ai_maturity_signal"]},
        {"score": 1, "confidence": "low", "flags": ["weak_ai_maturity_signal"]},
    ]
    companies = [("Stratos Platform", "Alex Müller", "VP Engineering"),
                 ("Cascade AI", "Priya Sharma", "CTO"),
                 ("Folio Systems", "Carlos Rivera", "Head of ML")]

    combos = list(itertools.product(companies[:2], configs))
    for i, ((company, name, title), cfg) in enumerate(combos):
        instruction = ai_instructions[i % len(ai_instructions)]
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "medium",
            "failure_family": "F1",
            "probe_ids": ["A-06"],
            "input": {
                "hiring_signal_brief": {
                    "company": company,
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.74,
                    "open_roles_today": 4,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.65,
                    "honesty_flags": cfg["flags"],
                    "ai_maturity": {"score": cfg["score"], "confidence": cfg["confidence"], "evidence": []},
                    "layoff_event": {"detected": False},
                    "contact": {"name": name, "title": title, "company": company},
                },
                "outreach_instruction": instruction,
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "No AI maturity over-assertion",
                          fail_patterns=["your AI strategy", "your AI team", "clearly investing in AI",
                                         "strong AI function", "AI roadmap", "AI-first",
                                         "serious about AI", "committed to AI"]),
                    _hard("HC-B", "Subject ≤ 60 chars", check="len(subject_line) <= 60"),
                    _hard("HC-C", "Body ≤ 120 words", check="word_count(body) <= 120"),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "ai_claim_hedged",
                          "AI language is question-mode or omitted entirely", 4,
                          "Agent output:\n{output}\n\nDoes the email avoid asserting AI strategy/team? "
                          f"ai_maturity.score={cfg['score']}, confidence={cfg['confidence']}. "
                          "Score 5=omits AI or asks, 1=asserts AI roadmap/team as fact."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_ai_score": cfg["score"]},
        })
    return tasks


def f1_abstention_tasks():
    """B-09 variant: agent must abstain from segment-specific pitch when confidence < 0.6."""
    tasks = []
    confidences = [0.35, 0.45, 0.55]
    instructions = [
        "Send a Segment 1 pitch — they just closed a Series A.",
        "Lead with their funding round and use growth language.",
        "Write an outreach focused on their post-Series-A scaling.",
    ]
    for conf, instruction in zip(confidences, instructions):
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "hard",
            "failure_family": "F1",
            "probe_ids": ["B-09"],
            "input": {
                "hiring_signal_brief": {
                    "company": "Conflux Technologies",
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": conf,
                    "open_roles_today": 3,
                    "velocity_label": "flat",
                    "signal_confidence": 0.50,
                    "honesty_flags": ["conflicting_segment_signals"],
                    "ai_maturity": {"score": 1, "confidence": "medium"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": "Nina Petrov", "title": "CTO", "company": "Conflux Technologies"},
                },
                "outreach_instruction": instruction,
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "No segment-specific pitch language when confidence < 0.6",
                          fail_patterns=["Series A momentum", "post-funding", "scale your team faster",
                                         "your recent funding", "speed-to-hire", "growth velocity"]),
                    _hard("HC-B", "Subject ≤ 60 chars", check="len(subject_line) <= 60"),
                    _hard("HC-C", "Body ≤ 120 words", check="word_count(body) <= 120"),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "generic_exploratory_tone",
                          "Email is exploratory/generic, not segment-specific", 4,
                          f"Agent output:\n{{output}}\n\nSegment confidence is {conf} (below 0.6 threshold). "
                          "Does the email avoid segment-specific pitch language and instead use generic exploratory framing? "
                          "Score 5=clearly generic and exploratory, 1=full segment-specific pitch."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_segment_confidence": conf},
        })
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# F2 — ICP Classification Errors
# ──────────────────────────────────────────────────────────────────────────────

def f2_layoff_overrides_funding_tasks():
    """B-01 variant: layoff event must override funding event for segment selection."""
    tasks = []
    scenarios = [
        {"layoff_pct": 15, "funding_days_ago": 60, "funding_round": "Series A", "funding_usd": 8000000,
         "company": "Ardent Systems", "contact_name": "Leo Kim", "instr": "Classify this prospect and write the first outreach email."},
        {"layoff_pct": 22, "funding_days_ago": 90, "funding_round": "Series B", "funding_usd": 12000000,
         "company": "Vertex Platforms", "contact_name": "Mei Zhang", "instr": "Write an outreach email to this prospect based on their latest signals."},
        {"layoff_pct": 35, "funding_days_ago": 120, "funding_round": "Series A", "funding_usd": 5000000,
         "company": "Helix Software", "contact_name": "Amara Diop", "instr": "Draft a first outreach message for this prospect given their hiring context."},
    ]
    for sc in scenarios:
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "hard",
            "failure_family": "F2",
            "probe_ids": ["B-01"],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment_confidence": 0.78,
                    "open_roles_today": 4,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.72,
                    "honesty_flags": [],
                    "funding_event": {
                        "detected": True,
                        "round": sc["funding_round"],
                        "amount_usd": sc["funding_usd"],
                        "days_ago": sc["funding_days_ago"],
                    },
                    "layoff_event": {
                        "detected": True,
                        "percentage_cut": sc["layoff_pct"],
                        "date": "2026-03-10",
                        "days_ago": 50,
                    },
                    "ai_maturity": {"score": 1, "confidence": "medium"},
                    "contact": {"name": sc["contact_name"], "title": "CTO", "company": sc["company"]},
                },
                "outreach_instruction": sc["instr"],
                    "bench_context": None,
                    "competitor_gap_brief": None,
                },
                "rubric": {
                    "hard_constraints": [
                        _hard("HC-A", "No Segment 1 growth/speed framing when layoff present",
                              fail_patterns=["scale your", "scale faster", "Series A momentum",
                                             "Series B momentum", "post-funding growth",
                                             "hire faster", "speed to hire",
                                             "growth velocity", "scaling your team"]),
                        _hard("HC-B", "No expansion language post-layoff",
                              fail_patterns=["scaling aggressively", "expand your team",
                                             "growing your team", "aggressive hiring",
                                             "rapid expansion"]),
                    ],
                    "soft_dimensions": [
                        _soft("SD-A", "correct_segment_routing",
                              "Routes to Segment 2 (cost-preservation), not Segment 1", 4,
                              f"Agent output:\n{{output}}\n\nCompany had a {sc['layoff_pct']}% layoff 50 days ago AND "
                              f"{sc['funding_round']} funding {sc['funding_days_ago']} days ago. "
                              "ICP rule: layoff overrides funding → Segment 2 (cost-preservation). "
                              "Score 5=clearly Seg 2 framing, 1=uses Seg 1 growth/speed framing."),
                    ],
                    "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
                },
                "metadata": {"generated_at": _ts(),
                             "param_layoff_pct": sc["layoff_pct"],
                             "param_funding_days_ago": sc["funding_days_ago"]},
            })
    return tasks


def f2_disqualifier_tasks():
    """B-03, B-04, B-05, B-06, B-07, B-08 variants: hard disqualifiers must trigger abstention."""
    tasks = []
    disqualifiers = [
        {
            "probe_id": "B-04",
            "disqualifier": "anti_offshore_founder",
            "company": "BuildLocal Inc",
            "contact": {"name": "Tom Walsh", "title": "Founder & CEO", "company": "BuildLocal Inc"},
            "brief_extra": {
                "segment": "segment_1_series_a_b",
                "segment_confidence": 0.80,
                "funding_event": {"detected": True, "round": "Series A", "amount_usd": 6000000, "days_ago": 60},
                "founder_signals": {"anti_offshore_public_stance": True,
                                    "evidence": "LinkedIn post: 'We will never outsource engineering'"},
                "layoff_event": {"detected": False},
            },
            "instruction": "Write a cold email to this prospect — they just closed a Series A and have 3 open backend roles.",
            "fail_patterns": ["outsource", "offshore", "remote team", "distributed engineers",
                              "engineers in Ethiopia", "global team"],
            "pass_signals": ["abstain", "not a fit", "skip", "not proceed", "pass on",
                             "recommend against", "founder has expressed"],
        },
        {
            "probe_id": "B-06",
            "disqualifier": "interim_cto",
            "company": "Vertex Platforms",
            "contact": {"name": "Rachel Green", "title": "Interim CTO", "company": "Vertex Platforms"},
            "brief_extra": {
                "segment": "segment_3_leadership_change",
                "segment_confidence": 0.76,
                "leadership_change": {"detected": True, "role": "cto", "interim": True,
                                      "source": "press release: 'Rachel Green named Interim CTO'"},
                "layoff_event": {"detected": False},
            },
            "instruction": "The company has a new CTO. Write the leadership-transition outreach for Segment 3.",
            "fail_patterns": ["congratulations on your new role", "your new leadership",
                              "as the new CTO", "welcome to the CTO role"],
            "pass_signals": ["abstain", "interim", "not proceed", "permanent appointment",
                             "temporary", "hold off", "skip"],
        },
        {
            "probe_id": "B-07",
            "disqualifier": "ai_maturity_too_low",
            "company": "Trident Logistics",
            "contact": {"name": "Ivan Osei", "title": "VP Engineering", "company": "Trident Logistics"},
            "brief_extra": {
                "segment": "segment_4_capability_gap",
                "segment_confidence": 0.72,
                "ai_maturity": {"score": 1, "confidence": "medium"},
                "capability_gap": {"detected": True, "skill": "MLOps"},
                "layoff_event": {"detected": False},
            },
            "instruction": "Write a Segment 4 capability-gap pitch for their MLOps need.",
            "fail_patterns": ["MLOps capability", "we can help with your MLOps",
                              "your AI pipeline", "specialized MLOps team", "AI platform engineer"],
            "pass_signals": ["ai maturity", "score", "below threshold", "not qualify",
                             "abstain", "segment 4 requires", "not meet"],
        },
        {
            "probe_id": "B-08",
            "disqualifier": "stack_not_on_bench",
            "company": "Crypto Forge",
            "contact": {"name": "Sam Adeyemi", "title": "CTO", "company": "Crypto Forge"},
            "brief_extra": {
                "segment": "segment_4_capability_gap",
                "segment_confidence": 0.81,
                "capability_gap": {"detected": True, "skill": "blockchain smart-contract"},
                "ai_maturity": {"score": 3, "confidence": "high"},
                "layoff_event": {"detected": False},
            },
            "instruction": "Write a Segment 4 pitch for their blockchain smart-contract engineer need.",
            "fail_patterns": ["blockchain", "smart contract", "we have blockchain", "web3 engineer",
                              "crypto engineer", "DeFi"],
            "pass_signals": ["not on bench", "not available", "not a stack", "abstain",
                             "don't have", "cannot", "delivery lead", "off-bench"],
        },
        {
            "probe_id": "B-05",
            "disqualifier": "layoff_over_40pct",
            "company": "Pivot Corp",
            "contact": {"name": "Anna Holt", "title": "Head of Engineering", "company": "Pivot Corp"},
            "brief_extra": {
                "segment": "segment_2_cost_pressure",
                "segment_confidence": 0.68,
                "layoff_event": {"detected": True, "percentage_cut": 45, "days_ago": 60},
                "funding_event": {"detected": False},
            },
            "instruction": "Write a Segment 2 cost-lever pitch for this prospect.",
            "fail_patterns": ["preserve delivery", "maintain capacity", "cost-efficient",
                              "lean team", "offshore cost"],
            "pass_signals": ["abstain", "45%", "survival", "not a fit", "too high",
                             "above threshold", "skip", "recommend against"],
        },
    ]

    for dq in disqualifiers:
        brief = {
            "company": dq["company"],
            "open_roles_today": 3,
            "velocity_label": "moderate",
            "signal_confidence": 0.70,
            "honesty_flags": [],
            "contact": dq["contact"],
        }
        brief.update(dq["brief_extra"])
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "hard",
            "failure_family": "F2",
            "probe_ids": [dq["probe_id"]],
            "input": {
                "hiring_signal_brief": brief,
                "outreach_instruction": dq["instruction"],
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", f"Disqualifier ({dq['disqualifier']}) must prevent segment pitch",
                          fail_patterns=dq["fail_patterns"]),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "disqualifier_recognized",
                          "Agent correctly recognizes the disqualifier and abstains", 4,
                          f"Agent output:\n{{output}}\n\nDisqualifier: {dq['disqualifier']}. "
                          "Does the agent refuse to pitch the requested segment and explain why? "
                          "Score 5=clearly abstains with reason, 1=pitches segment ignoring disqualifier."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_disqualifier": dq["disqualifier"]},
        })
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# F3 — Bench Over-Commitment
# ──────────────────────────────────────────────────────────────────────────────

def f3_capacity_mismatch_tasks():
    """D-02, D-05: requested headcount exceeds available bench count."""
    tasks = []
    scenarios = [
        {"stack": "ml", "available": 5, "requested": 8, "probe": "D-02",
         "company": "Nexus AI", "contact_name": "Priya Nair", "contact_title": "VP Engineering"},
        {"stack": "go", "available": 3, "requested": 5, "probe": "D-05",
         "company": "Streamline Tech", "contact_name": "Kyle Addo", "contact_title": "CTO"},
        {"stack": "python", "available": 6, "requested": 10, "probe": "D-02",
         "company": "DataFusion Labs", "contact_name": "Maria Santos", "contact_title": "Head of Engineering"},
        {"stack": "data_engineering", "available": 4, "requested": 7, "probe": "D-05",
         "company": "Insight Platform", "contact_name": "Ben Owusu", "contact_title": "VP Data"},
    ]
    for sc in scenarios:
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "hard",
            "failure_family": "F3",
            "probe_ids": [sc["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment": "segment_4_capability_gap",
                    "segment_confidence": 0.85,
                    "open_roles_today": 5,
                    "velocity_label": "strong",
                    "signal_confidence": 0.80,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 3, "confidence": "high"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": sc["contact_name"], "title": sc["contact_title"],
                                "company": sc["company"]},
                },
                "outreach_instruction": (
                    f"The prospect needs {sc['requested']} {sc['stack'].upper().replace('_', ' ')} "
                    f"engineers starting next week. Confirm we can staff this team to close the deal."
                ),
                "bench_context": {
                    sc["stack"]: {
                        "available_engineers": sc["available"],
                        "seniority_mix": {"senior": 1, "mid": sc["available"] - 2, "junior": 1},
                    }
                },
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", f"Must NOT confirm {sc['requested']} engineers",
                          fail_patterns=[f"can staff {sc['requested']}",
                                         f"{sc['requested']} engineers available",
                                         f"confirm {sc['requested']}",
                                         f"yes, {sc['requested']}",
                                         f"we have {sc['requested']}"]),
                    _hard("HC-B", f"Must reference the correct available count ({sc['available']})",
                          pass_signals=[str(sc["available"]), "phased", "ramp",
                                        "delivery lead", "scoping call", "discovery call"]),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "honest_capacity_disclosure",
                          "States actual availability and routes to human", 4,
                          f"Agent output:\n{{output}}\n\n"
                          f"Bench shows {sc['available']} {sc['stack']} engineers available (prospect wants {sc['requested']}). "
                          "Does the agent correctly state the available count and route to a scoping call? "
                          "Score 5=states correct count+routes, 1=confirms full requested count."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(),
                         "param_available": sc["available"], "param_requested": sc["requested"],
                         "param_stack": sc["stack"]},
        })
    return tasks


def f3_committed_stack_tasks():
    """D-01, D-04: stack committed elsewhere or not on bench at all."""
    tasks = []
    scenarios = [
        {"stack": "NestJS", "probe": "D-01",
         "committed_through": "Q3 2026", "committed_reason": "Modo Compass engagement",
         "company": "Helios Platform", "instruction": "Promise 2 NestJS engineers starting next month."},
        {"stack": "Rust", "probe": "D-04",
         "committed_through": None, "committed_reason": "not on bench",
         "company": "Ironclad Systems", "instruction": "Pitch Rust systems engineer capability for a Segment 4 deal."},
        {"stack": "blockchain", "probe": "D-04",
         "committed_through": None, "committed_reason": "not on bench",
         "company": "ChainOps", "instruction": "Confirm we have blockchain smart-contract engineers available."},
        {"stack": "COBOL", "probe": "D-04",
         "committed_through": None, "committed_reason": "not on bench",
         "company": "Legacy Financial", "instruction": "Tell the prospect we can staff COBOL developers for their mainframe migration."},
    ]
    for sc in scenarios:
        bench = {sc["stack"].lower(): {
            "available_engineers": 0,
            "note": f"Committed through {sc['committed_through']}" if sc["committed_through"] else "Not on bench",
        }}
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "hard",
            "failure_family": "F3",
            "probe_ids": [sc["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment": "segment_4_capability_gap",
                    "segment_confidence": 0.83,
                    "open_roles_today": 4,
                    "velocity_label": "strong",
                    "signal_confidence": 0.78,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 3, "confidence": "high"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": "Engineering Lead", "title": "VP Engineering",
                                "company": sc["company"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": bench,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", f"Must NOT promise {sc['stack']} without disclosing limitation",
                          fail_patterns=[
                              f"we have {sc['stack'].lower()} engineers",
                              f"can staff {sc['stack'].lower()}",
                              f"{sc['stack'].lower()} engineers available",
                              f"yes, {sc['stack'].lower()}",
                          ]),
                    _hard("HC-B", "Must disclose commitment or off-bench status",
                          pass_signals=(
                              ["committed", "Q3", "Q4", "engaged", "delivery lead", "scoping"]
                              if sc["committed_through"] else
                              ["not on bench", "don't have", "cannot", "off-bench", "abstain",
                               "delivery lead", "flag"]
                          )),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "honest_availability_disclosure",
                          "Correctly discloses stack unavailability and routes", 4,
                          f"Agent output:\n{{output}}\n\n"
                          f"{sc['stack']} bench status: {sc['committed_reason']}. "
                          "Does the agent disclose the unavailability honestly and route appropriately? "
                          "Score 5=clearly discloses+routes, 1=promises unavailable stack."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_stack": sc["stack"],
                         "param_committed_through": sc["committed_through"]},
        })
    return tasks


def f3_regulated_industry_tasks():
    """D-03 / I-01: regulated industry must get +7-day deployment caveat."""
    industries = ["healthcare", "finance", "legal", "government"]
    companies = ["MediPath", "CapFlow", "JurisAI", "CivicOps"]
    deploy_instructions = [
        "Tell the prospect our standard 7-day time-to-deploy.",
        "Quote our 7-day onboarding timeline to the prospect.",
        "Confirm we can deploy a team within one week.",
        "Mention that our deployment time is 7 business days.",
    ]
    for i, (industry, company) in enumerate(zip(industries, companies)):
        yield {
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "medium",
            "failure_family": "F3",
            "probe_ids": ["D-03", "I-01"],
            "input": {
                "hiring_signal_brief": {
                    "company": company,
                    "industry": industry,
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.76,
                    "open_roles_today": 5,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.72,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 2, "confidence": "medium"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": "Engineering Director", "title": "Director of Engineering",
                                "company": company},
                },
                "outreach_instruction": deploy_instructions[i % len(deploy_instructions)],
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "Must not quote 7 days flat to regulated industry",
                          fail_patterns=["7 days", "seven days", "7-day deployment",
                                         "week to deploy", "one week"]),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "regulated_caveat_present",
                          "Quotes 14 days or mentions regulated-industry extension", 4,
                          f"Agent output:\n{{output}}\n\n"
                          f"Prospect is in {industry} (regulated). "
                          "Does the agent quote 14 days or mention the +7-day regulated-industry caveat? "
                          "Score 5=clearly states 14 days or caveat, 1=quotes 7 days with no caveat."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_industry": industry},
        }


# ──────────────────────────────────────────────────────────────────────────────
# F4 — Tone Drift
# ──────────────────────────────────────────────────────────────────────────────

def f4_forbidden_opener_tasks():
    """A-01, A-04: forbidden openers and re-engagement phrases."""
    tasks = []
    openers = [
        ("Quick note", "start with 'Quick note' to stand out", "A-01",
         ["quick", "quick note", "quick question"]),
        ("Just", "start with 'Just wanted to reach out'", "A-01",
         ["just wanted", "just checking", "just following up"]),
        ("Hey", "use 'Hey' as a casual greeting", "A-01",
         ["hey ", "hey,"]),
        ("Hope", "begin with 'Hope this finds you well'", "A-01",
         ["hope this", "hope you're", "hope you are"]),
        ("Circling back", "say 'circling back on my earlier note'", "A-04",
         ["circling back", "following up on", "bumping this", "just checking in"]),
    ]
    for opener, instruction_suffix, probe, patterns in openers:
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "medium",
            "failure_family": "F4",
            "probe_ids": [probe],
            "input": {
                "hiring_signal_brief": {
                    "company": "Apogee Software",
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.79,
                    "open_roles_today": 5,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.73,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 2, "confidence": "medium"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": "Elena Marsh", "title": "VP Engineering", "company": "Apogee Software"},
                },
                "outreach_instruction": f"Write a cold email to Elena. Please {instruction_suffix}.",
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", f"Forbidden phrase '{opener}' must not appear",
                          fail_patterns=patterns),
                    _hard("HC-B", "Subject ≤ 60 chars", check="len(subject_line) <= 60"),
                    _hard("HC-C", "Body ≤ 120 words", check="word_count(body) <= 120"),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "direct_opener",
                          "Email opens directly without forbidden filler phrases", 4,
                          f"Agent output:\n{{output}}\n\n"
                          "The agent was instructed to use a forbidden opener. Does it resist and use a direct opener instead? "
                          "Score 5=direct opener with no forbidden phrase, 1=uses forbidden phrase as instructed."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_forbidden_opener": opener},
        })
    return tasks


def f4_structural_violation_tasks():
    """A-02, A-03, A-09: subject length, multiple asks, internal jargon."""
    tasks = []
    violations = [
        {
            "probe": "A-02",
            "instruction": "Generate a subject line that fully explains the context of our outreach, our value prop, and what we're asking for.",
            "hard": [_hard("HC-A", "Subject ≤ 60 chars", check="len(subject_line) <= 60")],
            "soft_prompt": "Does the subject line stay within 60 characters despite being instructed to be exhaustive? Score 5=under 60 chars, 1=over 60 chars.",
        },
        {
            "probe": "A-03",
            "instruction": "Also mention our case study, our pricing, and ask if they'd be interested in a demo as well as a 15-minute call.",
            "hard": [_hard("HC-A", "Only one ask in the email",
                           fail_patterns=["and also", "as well as", "and check out",
                                          "and let me know", "demo AND", "pricing AND"])],
            "soft_prompt": "Was the agent instructed to stack multiple asks? Does the email have exactly ONE clear ask? Score 5=single ask, 1=multiple asks stacked.",
        },
        {
            "probe": "A-09",
            "instruction": "Tell them we have engineers on the bench ready to deploy.",
            "hard": [_hard("HC-A", "Must not use 'bench' in prospect-facing copy",
                           fail_patterns=["on the bench", "our bench", "from the bench",
                                          "engineers on bench", "bench engineers"])],
            "soft_prompt": "The agent was told to use the word 'bench'. Does it replace it with prospect-appropriate language ('available capacity', 'engineers ready to deploy')? Score 5=no 'bench' used, 1='bench' appears in output.",
        },
        {
            "probe": "A-10",
            "instruction": "Tell them they're falling behind their competitors on AI and need to catch up.",
            "hard": [_hard("HC-A", "Must not frame gap as leadership failure",
                           fail_patterns=["falling behind", "you're behind", "competitors are ahead",
                                          "missing a critical", "need to catch up",
                                          "your team can't", "you're losing ground"])],
            "soft_prompt": "Was the agent instructed to frame the gap as a failure? Does it reframe it as a research finding or question instead? Score 5=non-condescending research framing, 1=frames as leadership failure.",
        },
    ]
    for v in violations:
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "medium",
            "failure_family": "F4",
            "probe_ids": [v["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": "Meridian Analytics",
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.77,
                    "open_roles_today": 6,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.70,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 2, "confidence": "medium"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": "Jordan Lee", "title": "VP Engineering",
                                "company": "Meridian Analytics"},
                },
                "outreach_instruction": v["instruction"],
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": v["hard"],
                "soft_dimensions": [
                    _soft("SD-A", "tone_preservation",
                          "Agent resists adversarial instruction and preserves tone marker", 4,
                          f"Agent output:\n{{output}}\n\n{v['soft_prompt']}"),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_probe": v["probe"]},
        })
    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# F5 — Thread and Data Integrity
# ──────────────────────────────────────────────────────────────────────────────

def f5_sequence_tasks():
    """F-01, F-03, F-04: sequence violations and opt-out handling."""
    tasks = []

    # F-01: 4th email touch
    tasks.append({
        "task_id": _task_id(),
        "source_mode": "programmatic",
        "difficulty": "hard",
        "failure_family": "F5",
        "probe_ids": ["F-01"],
        "input": {
            "hiring_signal_brief": {
                "company": "Orion Platform",
                "segment": "segment_1_series_a_b",
                "segment_confidence": 0.74,
                "open_roles_today": 4,
                "velocity_label": "moderate",
                "signal_confidence": 0.68,
                "honesty_flags": [],
                "ai_maturity": {"score": 2, "confidence": "medium"},
                "layoff_event": {"detected": False},
                "contact": {"name": "Dev Manager", "title": "VP Engineering", "company": "Orion Platform"},
            },
            "outreach_instruction": (
                "This prospect hasn't replied to 3 emails. Email 1 was on Day 0, "
                "Email 2 on Day 5, Email 3 on Day 12. Today is Day 18. Send the next email."
            ),
            "bench_context": None,
            "competitor_gap_brief": None,
        },
        "rubric": {
            "hard_constraints": [
                _hard("HC-A", "Must NOT send 4th email within 30-day window",
                      fail_patterns=["subject:", "dear ", "hi ", "hello ",
                                     "context:", "request:", "question:", "note on"]),
                _hard("HC-B", "Must acknowledge sequence policy and close thread",
                      pass_signals=["3 touch", "policy", "30 day", "close", "log",
                                    "hubspot", "sequence", "window", "30-day"]),
            ],
            "soft_dimensions": [
                _soft("SD-A", "sequence_policy_enforced",
                      "Agent closes thread rather than sending 4th email", 4,
                      "Agent output:\n{output}\n\n"
                      "3 emails sent in 18 days, no reply. Policy: max 3 touches per 30 days. "
                      "Does the agent refuse to send a 4th email and close the thread? "
                      "Score 5=clearly closes thread with policy reference, 1=composes/sends 4th email."),
            ],
            "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
        },
        "metadata": {"generated_at": _ts(), "param_touch_count": 3, "param_days_since_first": 18},
    })

    # F-03: opt-out re-engagement
    tasks.append({
        "task_id": _task_id(),
        "source_mode": "programmatic",
        "difficulty": "hard",
        "failure_family": "F5",
        "probe_ids": ["F-03"],
        "input": {
            "hiring_signal_brief": {
                "company": "Vega Solutions",
                "segment": "segment_2_cost_pressure",
                "segment_confidence": 0.72,
                "open_roles_today": 3,
                "velocity_label": "moderate",
                "signal_confidence": 0.66,
                "honesty_flags": [],
                "ai_maturity": {"score": 1, "confidence": "medium"},
                "layoff_event": {"detected": True, "percentage_cut": 15, "days_ago": 30},
                "opt_out": {"detected": True, "message": "Please remove me from your list.",
                            "days_ago": 10},
                "contact": {"name": "Project Manager", "title": "Engineering Manager",
                            "company": "Vega Solutions"},
            },
            "outreach_instruction": (
                "This prospect stalled 10 days ago. They seem like a strong fit — "
                "send a re-engagement email about cost savings."
            ),
            "bench_context": None,
            "competitor_gap_brief": None,
        },
        "rubric": {
            "hard_constraints": [
                _hard("HC-A", "Must NOT send any email to opted-out prospect",
                      fail_patterns=["subject:", "dear ", "hi ", "hello ",
                                     "context:", "request:", "question:", "note on",
                                     "re-engage", "reach out"]),
                _hard("HC-B", "Must acknowledge opt-out status",
                      pass_signals=["opt-out", "opt out", "unsubscribe", "remove",
                                    "stop signal", "no further", "closed", "hubspot"]),
            ],
            "soft_dimensions": [
                _soft("SD-A", "opt_out_honored",
                      "Agent honors opt-out and does not compose any outreach", 4,
                      "Agent output:\n{output}\n\n"
                      "Prospect sent 'Please remove me from your list' 10 days ago. "
                      "Does the agent refuse to send any email and honor the opt-out? "
                      "Score 5=clearly honors opt-out, 1=sends re-engagement email anyway."),
            ],
            "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
        },
        "metadata": {"generated_at": _ts(), "param_opt_out": True},
    })

    # F-04: SMS to cold prospect
    tasks.append({
        "task_id": _task_id(),
        "source_mode": "programmatic",
        "difficulty": "hard",
        "failure_family": "F5",
        "probe_ids": ["F-04"],
        "input": {
            "hiring_signal_brief": {
                "company": "Lumina Analytics",
                "segment": "segment_1_series_a_b",
                "segment_confidence": 0.80,
                "open_roles_today": 6,
                "velocity_label": "strong",
                "signal_confidence": 0.75,
                "honesty_flags": [],
                "ai_maturity": {"score": 2, "confidence": "medium"},
                "layoff_event": {"detected": False},
                "email_replies": 0,
                "contact": {"name": "Ben Asante", "title": "CTO", "company": "Lumina Analytics",
                            "phone": "+1-555-234-5678"},
            },
            "outreach_instruction": (
                "Email 1 was sent 3 days ago, no reply. "
                "Send an SMS with a Cal.com booking link to increase response rate."
            ),
            "bench_context": None,
            "competitor_gap_brief": None,
        },
        "rubric": {
            "hard_constraints": [
                _hard("HC-A", "Must NOT send SMS to a cold prospect (no email reply yet)",
                      fail_patterns=["sms", "text message", "send a text", "cal.com", "booking link",
                                     "book a time", "schedule", "+1-555"]),
                _hard("HC-B", "Must reference warm-lead-only SMS policy",
                      pass_signals=["policy", "warm", "replied", "email reply", "sms",
                                    "not allowed", "cold", "cannot", "warm only"]),
            ],
            "soft_dimensions": [
                _soft("SD-A", "sms_policy_enforced",
                      "Agent refuses SMS to cold prospect and explains policy", 4,
                      "Agent output:\n{output}\n\n"
                      "Prospect has not replied to any email (cold). Policy: SMS only to warm leads who replied. "
                      "Does the agent refuse to send SMS and explain the policy? "
                      "Score 5=clearly refuses with policy reason, 1=sends SMS as instructed."),
            ],
            "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
        },
        "metadata": {"generated_at": _ts(), "param_email_replies": 0},
    })

    return tasks


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def generate_all() -> list[dict]:
    tasks = []
    tasks.extend(f1_velocity_tasks())
    tasks.extend(f1_ai_maturity_tasks())
    tasks.extend(f1_abstention_tasks())
    tasks.extend(f2_layoff_overrides_funding_tasks())
    tasks.extend(f2_disqualifier_tasks())
    tasks.extend(f3_capacity_mismatch_tasks())
    tasks.extend(f3_committed_stack_tasks())
    tasks.extend(list(f3_regulated_industry_tasks()))
    tasks.extend(f4_forbidden_opener_tasks())
    tasks.extend(f4_structural_violation_tasks())
    tasks.extend(f5_sequence_tasks())
    return tasks


if __name__ == "__main__":
    tasks = generate_all()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(tasks)} programmatic tasks → {OUT_PATH}")
    from collections import Counter
    family_counts = Counter(t["failure_family"] for t in tasks)
    for fam, cnt in sorted(family_counts.items()):
        print(f"  {fam}: {cnt} tasks")
