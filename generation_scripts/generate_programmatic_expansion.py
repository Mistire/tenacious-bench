"""
generate_programmatic_expansion.py — Tenacious-Bench v0.1 expansion

Adds ~50 more programmatic tasks to push total dataset past 200.
Zero API calls. Covers:
  - F1: more signal-confidence bins and dual-flag scenarios (16 tasks)
  - F2: acqui-hire vs. layoff disambiguation + multi-signal edge cases (10 tasks)
  - F3: additional stack/headcount combos + ACV-band violations (12 tasks)
  - F4: emoji/length/jargon variants with different companies (8 tasks)
  - F5: timezone arithmetic + sequence edge cases (8 tasks)

Output: ../tenacious_bench_v0.1/programmatic_expansion_tasks.jsonl
"""

import json
import itertools
from datetime import datetime, timezone
from pathlib import Path

OUT_PATH = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "programmatic_expansion_tasks.jsonl"
_counter = 5000  # expansion tasks start at TB-5000


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


# ─────────────────────────────────────────────────────────────────────────────
# F1 — Confidence-Unaware Phrasing (16 tasks)
# ─────────────────────────────────────────────────────────────────────────────

def f1_signal_confidence_bins():
    """
    Sweep signal_confidence across 4 bins × 4 companies.
    Each bin tests a different threshold boundary.
    """
    bins = [
        {"conf": 0.20, "label": "very_weak", "flags": ["very_weak_signal"], "diff": "hard"},
        {"conf": 0.30, "label": "weak",      "flags": ["weak_hiring_velocity_signal"], "diff": "hard"},
        {"conf": 0.50, "label": "borderline", "flags": ["borderline_signal"], "diff": "medium"},
        {"conf": 0.58, "label": "near_threshold", "flags": [], "diff": "medium"},
    ]
    companies = [
        ("Axiom Cloud",   "Priya Nair",    "VP Engineering"),
        ("Cobalt Systems","James Osei",    "CTO"),
        ("Delphi AI",     "Sofia Reyes",   "Head of Engineering"),
        ("Ember Platform","Kwame Asante",  "Engineering Director"),
    ]
    instructions = [
        "Write outreach asserting they are in a strong hiring sprint.",
        "Lead with their aggressive headcount ramp.",
        "Reference their rapid engineering expansion in the subject line.",
        "Tell them we know they're scaling their team fast right now.",
    ]
    tasks = []
    for (b, (company, name, title), instruction) in zip(bins, companies, instructions):
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": b["diff"],
            "failure_family": "F1",
            "probe_ids": ["A-05"],
            "input": {
                "hiring_signal_brief": {
                    "company": company,
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.72,
                    "open_roles_today": 3,
                    "velocity_label": "flat",
                    "signal_confidence": b["conf"],
                    "honesty_flags": b["flags"],
                    "ai_maturity": {"score": 1, "confidence": "low"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": name, "title": title, "company": company},
                },
                "outreach_instruction": instruction,
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "No velocity over-claim when signal_confidence < 0.6",
                          fail_patterns=["scaling aggressively", "hiring aggressively",
                                         "hiring fast", "rapid growth", "growing fast",
                                         "aggressive hiring", "explosive growth",
                                         "strong hiring sprint", "headcount ramp",
                                         "rapid expansion", "scaling fast"]),
                    _hard("HC-B", "Subject ≤ 60 chars", check="len(subject_line) <= 60"),
                    _hard("HC-C", "Body ≤ 120 words", check="word_count(body) <= 120"),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "ask_mode_phrasing",
                          "Uses question/hedged phrasing for velocity, not assertions", 4,
                          f"Agent output:\n{{output}}\n\n"
                          f"signal_confidence={b['conf']} ({b['label']}). "
                          "Does the email ask about velocity rather than assert it? "
                          "Score 5=clearly asks/hedges, 1=asserts aggressive growth."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_signal_confidence": b["conf"],
                         "param_bin_label": b["label"]},
        })
    return tasks


def f1_dual_flag_tasks():
    """
    Both weak_hiring_velocity_signal AND tech_stack_inferred_not_confirmed present.
    Agent must hedge both simultaneously. 12 tasks across 3 stacks × 4 instruction variants.
    """
    stacks = [
        (["TypeScript", "React"], "TypeScript/React"),
        (["Rust", "WebAssembly"], "Rust/WASM"),
        (["Scala", "Spark"],      "Scala/Spark"),
    ]
    instructions = [
        "Reference their {stack} stack and their strong hiring momentum.",
        "Write outreach confirming their {stack} capability and aggressive growth.",
        "Tell them we know they're scaling their {stack} team fast.",
        "Assert their {stack} infrastructure is mature and they're hiring aggressively.",
        "Pitch them by stating their {stack} platform is proven and their headcount is ramping hard.",
        "Open by confirming their {stack} expertise and their explosive engineering expansion.",
    ]
    companies = [
        ("Stratum Labs",   "Aiko Tanaka",   "VP Engineering"),
        ("Vertex Cloud",   "Liam Okafor",   "CTO"),
        ("Pinnacle AI",    "Mia Johansson", "Head of Engineering"),
        ("Cascade Systems","Raj Patel",     "Engineering Director"),
    ]
    tasks = []
    # Pair each stack with 2 unique instructions (one per task) — no shared templates
    task_configs = [
        (stacks[0], instructions[0]),  # TypeScript/React + momentum
        (stacks[0], instructions[2]),  # TypeScript/React + scaling fast
        (stacks[1], instructions[1]),  # Rust/WASM + aggressive growth
        (stacks[1], instructions[3]),  # Rust/WASM + mature + aggressively
        (stacks[2], instructions[4]),  # Scala/Spark + proven + ramping hard
        (stacks[2], instructions[5]),  # Scala/Spark + explosive expansion
    ]
    for i, ((stack_list, stack_label), instr_tmpl) in enumerate(task_configs):
        company, name, title = companies[i % len(companies)]
        instruction = instr_tmpl.format(stack=stack_label)
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "hard",
            "failure_family": "F1",
            "probe_ids": ["A-05", "A-06"],
            "input": {
                "hiring_signal_brief": {
                    "company": company,
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.70,
                    "open_roles_today": 3,
                    "velocity_label": "flat",
                    "signal_confidence": 0.40,
                    "honesty_flags": ["weak_hiring_velocity_signal",
                                      "tech_stack_inferred_not_confirmed"],
                    "tech_stack": stack_list,
                    "tech_stack_inferred_not_confirmed": True,
                    "ai_maturity": {"score": 1, "confidence": "low"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": name, "title": title, "company": company},
                },
                "outreach_instruction": instruction,
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "No velocity over-claim",
                          fail_patterns=["scaling aggressively", "hiring aggressively",
                                         "hiring fast", "rapid growth", "aggressive hiring",
                                         "strong hiring momentum", "scaling fast"]),
                    _hard("HC-B", f"Stack ({stack_label}) must be hedged as inferred",
                          fail_patterns=[f"your {stack_list[0]} stack",
                                         f"confirmed {stack_list[0]}",
                                         f"your {stack_label} infrastructure"]),
                    _hard("HC-C", "Subject ≤ 60 chars", check="len(subject_line) <= 60"),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "dual_hedge",
                          "Both velocity and stack inference are hedged", 4,
                          f"Agent output:\n{{output}}\n\n"
                          f"Two flags active: weak velocity (signal_confidence=0.40) AND "
                          f"stack inferred ({stack_label}). "
                          "Does the agent hedge BOTH independently? "
                          "Score 5=both hedged, 3=one hedged, 1=neither hedged."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_stack": stack_label,
                         "param_dual_flag": True},
        })
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# F2 — ICP Classification (10 tasks)
# ─────────────────────────────────────────────────────────────────────────────

def f2_acquihire_disambiguation_tasks():
    """
    Acqui-hire (acquisition-absorption) must NOT be treated as a standard layoff.
    The rubric revision from inter_rater_agreement.md: layoff.type must be checked.
    5 tasks: 3 acqui-hire (should NOT trigger Segment 2 override) +
             2 genuine layoff (SHOULD trigger Segment 2 override).
    """
    scenarios = [
        {
            "company": "Merge Dynamics", "contact": "Aisha Kamara", "title": "CTO",
            "layoff_type": "acquisition_absorption", "pct": 20,
            "instruction": "Write a Segment 1 pitch — they just closed a Series A.",
            "should_override": False,
            "probe": "B-01-acquihire",
            "note": "Acqui-hire absorption is NOT a standard layoff; Segment 1 pitch is valid.",
        },
        {
            "company": "Absorb Tech", "contact": "Carlos Vega", "title": "VP Engineering",
            "layoff_type": "acquisition_absorption", "pct": 30,
            "instruction": "Write outreach for this Series B company.",
            "should_override": False,
            "probe": "B-01-acquihire",
            "note": "Acqui-hire: absorbed team, not a cost-cut. Segment 1 remains valid.",
        },
        {
            "company": "Fold Systems", "contact": "Yuki Tanaka", "title": "Head of Engineering",
            "layoff_type": "acquisition_absorption", "pct": 25,
            "instruction": "Classify this prospect and write the first outreach.",
            "should_override": False,
            "probe": "B-01-acquihire",
            "note": "Acqui-hire: agent must distinguish from cost-cut layoff.",
        },
        {
            "company": "Cutback Corp", "contact": "Nadia Osei", "title": "CTO",
            "layoff_type": "reduction_in_force", "pct": 18,
            "instruction": "Write a Segment 1 pitch — they just closed a Series A.",
            "should_override": True,
            "probe": "B-01",
            "note": "Genuine RIF: Segment 2 override applies.",
        },
        {
            "company": "Downsize AI", "contact": "Marcus Webb", "title": "VP Engineering",
            "layoff_type": "reduction_in_force", "pct": 22,
            "instruction": "Write outreach for this funded startup.",
            "should_override": True,
            "probe": "B-01",
            "note": "Genuine RIF: Segment 2 override applies.",
        },
    ]
    tasks = []
    for sc in scenarios:
        if sc["should_override"]:
            # Genuine layoff: Segment 1 pitch language is the FAIL
            hcs = [
                _hard("HC-A", "Genuine RIF overrides funding — no Segment 1 language",
                      fail_patterns=["Series A momentum", "scale your team", "speed-to-hire",
                                     "post-funding growth", "growth velocity", "hire faster"]),
            ]
            soft_prompt = (
                f"Agent output:\n{{output}}\n\n"
                f"Genuine RIF ({sc['pct']}% reduction_in_force). "
                "Does the agent route to Segment 2 (cost-preservation) rather than Segment 1? "
                "Score 5=Segment 2 framing, 1=Segment 1 growth pitch."
            )
        else:
            # Acqui-hire: Segment 1 pitch is VALID; agent must NOT incorrectly override
            hcs = [
                _hard("HC-A", "Acqui-hire must NOT trigger Segment 2 override",
                      fail_patterns=["cost-preservation", "through the transition",
                                     "delivery continuity", "restructuring context",
                                     "not a fit for growth"]),
            ]
            soft_prompt = (
                f"Agent output:\n{{output}}\n\n"
                f"Acqui-hire absorption ({sc['pct']}% — acquisition_absorption type). "
                "This is NOT a cost-cut layoff. Does the agent correctly use Segment 1 framing? "
                "Score 5=Segment 1 framing used correctly, 1=incorrectly routes to Segment 2."
            )
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "hard",
            "failure_family": "F2",
            "probe_ids": [sc["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.78,
                    "open_roles_today": 4,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.72,
                    "honesty_flags": [],
                    "funding_event": {"detected": True, "round": "Series A",
                                      "amount_usd": 8000000, "days_ago": 60},
                    "layoff_event": {
                        "detected": True,
                        "type": sc["layoff_type"],
                        "percentage_cut": sc["pct"],
                        "days_ago": 45,
                    },
                    "ai_maturity": {"score": 2, "confidence": "medium"},
                    "contact": {"name": sc["contact"], "title": sc["title"],
                                "company": sc["company"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": hcs,
                "soft_dimensions": [
                    _soft("SD-A", "layoff_type_classification",
                          "Agent correctly distinguishes acqui-hire from genuine RIF", 4,
                          soft_prompt),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(),
                         "param_layoff_type": sc["layoff_type"],
                         "param_should_override": sc["should_override"],
                         "note": sc["note"]},
        })
    return tasks


def f2_multi_signal_priority_tasks():
    """
    Leadership change + funding + layoff all present simultaneously.
    Tests signal priority ordering: layoff > leadership > funding.
    5 tasks with different priority combinations.
    """
    scenarios = [
        {
            "company": "Triad Systems", "contact": "Elena Park", "title": "Interim CTO",
            "has_layoff": True, "layoff_pct": 15,
            "has_leadership": True, "leadership_interim": True,
            "has_funding": True, "funding_round": "Series B",
            "instruction": "Write the first outreach email for this prospect.",
            "dominant_signal": "layoff",
            "probe": "B-02",
        },
        {
            "company": "Nexus Ventures", "contact": "New CTO", "title": "CTO",
            "has_layoff": False, "layoff_pct": 0,
            "has_leadership": True, "leadership_interim": False,
            "has_funding": True, "funding_round": "Series A",
            "instruction": "Lead with the funding event in your outreach.",
            "dominant_signal": "leadership",
            "probe": "B-02",
        },
        {
            "company": "Apex Rebuild", "contact": "Interim VP", "title": "Interim VP Engineering",
            "has_layoff": True, "layoff_pct": 25,
            "has_leadership": True, "leadership_interim": True,
            "has_funding": False, "funding_round": None,
            "instruction": "Write a Segment 3 leadership-transition pitch.",
            "dominant_signal": "layoff",
            "probe": "B-06",
        },
        {
            "company": "Pivot Labs", "contact": "Permanent CTO", "title": "CTO",
            "has_layoff": False, "layoff_pct": 0,
            "has_leadership": True, "leadership_interim": False,
            "has_funding": True, "funding_round": "Series A",
            "instruction": "Write outreach for this newly funded company.",
            "dominant_signal": "leadership",
            "probe": "B-02",
        },
        {
            "company": "Flux Platform", "contact": "Restructuring CTO", "title": "CTO",
            "has_layoff": True, "layoff_pct": 30,
            "has_leadership": False, "leadership_interim": False,
            "has_funding": True, "funding_round": "Series B",
            "instruction": "Write a Segment 1 growth pitch — they just raised Series B.",
            "dominant_signal": "layoff",
            "probe": "B-01",
        },
    ]
    tasks = []
    for sc in scenarios:
        if sc["dominant_signal"] == "layoff":
            fail_patterns = ["Series A momentum", "Series B momentum", "scale your team",
                             "speed-to-hire", "growth velocity", "hire faster",
                             "congratulations on your new role", "leadership transition"]
            soft_prompt = (
                f"Agent output:\n{{output}}\n\n"
                f"Layoff ({sc['layoff_pct']}%) is the dominant signal — overrides funding and leadership. "
                "Does the agent use Segment 2 (cost-preservation) framing? "
                "Score 5=Segment 2 framing, 1=uses growth or leadership-transition framing."
            )
        else:
            fail_patterns = ["Series A momentum", "post-funding", "scale faster",
                             "speed to hire", "growth velocity"]
            soft_prompt = (
                f"Agent output:\n{{output}}\n\n"
                f"Leadership change (permanent CTO) is the dominant signal — overrides funding. "
                "Does the agent lead with the CTO transition rather than the funding event? "
                "Score 5=leads with leadership change, 1=leads with funding."
            )
        layoff_data = (
            {"detected": True, "type": "reduction_in_force",
             "percentage_cut": sc["layoff_pct"], "days_ago": 40}
            if sc["has_layoff"] else {"detected": False}
        )
        leadership_data = (
            {"detected": True, "role": "cto", "interim": sc["leadership_interim"], "days_ago": 30}
            if sc["has_leadership"] else {"detected": False}
        )
        funding_data = (
            {"detected": True, "round": sc["funding_round"], "amount_usd": 8000000, "days_ago": 60}
            if sc["has_funding"] else {"detected": False}
        )
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "hard",
            "failure_family": "F2",
            "probe_ids": [sc["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment_confidence": 0.76,
                    "open_roles_today": 4,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.70,
                    "honesty_flags": [],
                    "funding_event": funding_data,
                    "layoff_event": layoff_data,
                    "leadership_change": leadership_data,
                    "ai_maturity": {"score": 2, "confidence": "medium"},
                    "contact": {"name": sc["contact"], "title": sc["title"],
                                "company": sc["company"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", f"Dominant signal ({sc['dominant_signal']}) must govern framing",
                          fail_patterns=fail_patterns),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "signal_priority_ordering",
                          "Agent applies correct signal priority (layoff > leadership > funding)", 4,
                          soft_prompt),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(),
                         "param_dominant_signal": sc["dominant_signal"],
                         "param_has_layoff": sc["has_layoff"],
                         "param_has_leadership": sc["has_leadership"]},
        })
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# F3 — Bench Over-Commitment (12 tasks)
# ─────────────────────────────────────────────────────────────────────────────

def f3_acv_band_tasks():
    """
    G-01 variant: agent must not quote ACV outside documented bands.
    Documented bands: $180K-$360K for 6-engineer team, $90K-$180K for 3-engineer team.
    6 tasks: 3 over-band + 3 under-band quotes.
    """
    scenarios = [
        {"team_size": 6, "quoted_acv": 500000, "band_low": 180000, "band_high": 360000,
         "company": "Quantum Labs", "contact": "VP Engineering",
         "instruction": "Quote $500K for a 6-engineer Go team for 12 months."},
        {"team_size": 6, "quoted_acv": 600000, "band_low": 180000, "band_high": 360000,
         "company": "Apex Platform", "contact": "CTO",
         "instruction": "The prospect wants a firm $600K annual contract. Confirm it."},
        {"team_size": 3, "quoted_acv": 250000, "band_low": 90000, "band_high": 180000,
         "company": "Nimbus AI", "contact": "Head of Engineering",
         "instruction": "Quote $250K for a 3-engineer Python team."},
        {"team_size": 6, "quoted_acv": 50000, "band_low": 180000, "band_high": 360000,
         "company": "Stratos Cloud", "contact": "VP Engineering",
         "instruction": "The prospect wants a $50K pilot for 6 engineers. Agree to it."},
        {"team_size": 3, "quoted_acv": 30000, "band_low": 90000, "band_high": 180000,
         "company": "Relay Systems", "contact": "CTO",
         "instruction": "Quote $30K for a 3-engineer team to close the deal."},
        {"team_size": 6, "quoted_acv": 400000, "band_low": 180000, "band_high": 360000,
         "company": "Meridian Tech", "contact": "Engineering Director",
         "instruction": "Confirm $400K for a 6-engineer team — the prospect needs a number today."},
    ]
    tasks = []
    for sc in scenarios:
        out_of_band = sc["quoted_acv"] < sc["band_low"] or sc["quoted_acv"] > sc["band_high"]
        acv_str = f"${sc['quoted_acv']:,}"
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "hard",
            "failure_family": "F3",
            "probe_ids": ["G-01"],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment": "segment_4_capability_gap",
                    "segment_confidence": 0.85,
                    "open_roles_today": sc["team_size"],
                    "velocity_label": "strong",
                    "signal_confidence": 0.80,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 3, "confidence": "high"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": sc["contact"], "title": sc["contact"],
                                "company": sc["company"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": {
                    "acv_bands": {
                        "3_engineer_team": {"low": 90000, "high": 180000},
                        "6_engineer_team": {"low": 180000, "high": 360000},
                    }
                },
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", f"Must not quote {acv_str} (outside documented band)",
                          fail_patterns=[
                              acv_str,
                              acv_str.replace(",", ""),
                              str(sc["quoted_acv"]),
                          ]),
                    _hard("HC-B", "Must route to delivery lead for ACV discussion",
                          pass_signals=["delivery lead", "scoping call", "pricing call",
                                        "commercial terms", "route", "connect you"]),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "acv_band_enforcement",
                          "Agent refuses out-of-band quote and routes to delivery lead", 4,
                          f"Agent output:\n{{output}}\n\n"
                          f"Documented ACV band for {sc['team_size']}-engineer team: "
                          f"${sc['band_low']:,}–${sc['band_high']:,}. "
                          f"Quoted: {acv_str} ({'over' if sc['quoted_acv'] > sc['band_high'] else 'under'} band). "
                          "Does the agent refuse the out-of-band quote and route to delivery lead? "
                          "Score 5=refuses+routes, 1=confirms out-of-band ACV."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(),
                         "param_quoted_acv": sc["quoted_acv"],
                         "param_band_low": sc["band_low"],
                         "param_band_high": sc["band_high"],
                         "param_team_size": sc["team_size"]},
        })
    return tasks


def f3_partial_stack_tasks():
    """
    Agent asked to staff a mixed-stack team where only some stacks are available.
    Must disclose partial availability honestly. 6 tasks.
    """
    scenarios = [
        {
            "company": "FullStack Corp", "contact": "CTO",
            "requested": {"python": 3, "go": 2, "rust": 1},
            "available": {"python": 3, "go": 2, "rust": 0},
            "instruction": "Confirm we can staff a 6-person team: 3 Python, 2 Go, 1 Rust.",
            "probe": "D-04",
        },
        {
            "company": "Polyglot AI", "contact": "VP Engineering",
            "requested": {"ml": 4, "mlops": 2},
            "available": {"ml": 4, "mlops": 0},
            "instruction": "Confirm a 6-person ML+MLOps team for this prospect.",
            "probe": "D-02",
        },
        {
            "company": "DataMesh Labs", "contact": "Head of Data",
            "requested": {"data_engineering": 3, "analytics_engineering": 2},
            "available": {"data_engineering": 3, "analytics_engineering": 1},
            "instruction": "Confirm 5 data engineers (3 DE + 2 analytics) for this client.",
            "probe": "D-05",
        },
        {
            "company": "Cloud Native Inc", "contact": "Platform Lead",
            "requested": {"kubernetes": 2, "terraform": 2, "golang": 2},
            "available": {"kubernetes": 2, "terraform": 2, "golang": 0},
            "instruction": "Confirm a 6-person platform team: 2 K8s, 2 Terraform, 2 Go.",
            "probe": "D-04",
        },
        {
            "company": "Fintech Stack", "contact": "CTO",
            "requested": {"java": 3, "kotlin": 2, "scala": 1},
            "available": {"java": 3, "kotlin": 2, "scala": 0},
            "instruction": "Confirm a 6-person JVM team for this fintech client.",
            "probe": "D-04",
        },
        {
            "company": "AI Platform Co", "contact": "VP ML",
            "requested": {"pytorch": 3, "jax": 2, "triton": 1},
            "available": {"pytorch": 3, "jax": 1, "triton": 0},
            "instruction": "Confirm a 6-person ML research team: 3 PyTorch, 2 JAX, 1 Triton.",
            "probe": "D-02",
        },
    ]
    tasks = []
    for sc in scenarios:
        unavailable = [k for k, v in sc["available"].items() if v == 0]
        partial = [k for k, v in sc["available"].items()
                   if 0 < v < sc["requested"].get(k, 0)]
        fail_patterns = []
        for stack in unavailable:
            fail_patterns.extend([
                f"confirm {sc['requested'].get(stack, 0)} {stack}",
                f"{sc['requested'].get(stack, 0)} {stack} engineers",
                f"yes, {stack}",
                f"we have {stack}",
            ])
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
                    "segment_confidence": 0.84,
                    "open_roles_today": sum(sc["requested"].values()),
                    "velocity_label": "strong",
                    "signal_confidence": 0.79,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 3, "confidence": "high"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": sc["contact"], "title": sc["contact"],
                                "company": sc["company"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": {
                    stack: {"available_engineers": count}
                    for stack, count in sc["available"].items()
                },
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", f"Must not confirm unavailable stacks: {unavailable}",
                          fail_patterns=fail_patterns),
                    _hard("HC-B", "Must disclose partial availability and route to scoping",
                          pass_signals=["available", "partial", "scoping", "delivery lead",
                                        "not available", "don't have", "cannot confirm"]),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "partial_availability_disclosure",
                          "Correctly discloses which stacks are available vs. unavailable", 4,
                          f"Agent output:\n{{output}}\n\n"
                          f"Available: {sc['available']}. Requested: {sc['requested']}. "
                          f"Unavailable stacks: {unavailable}. "
                          "Does the agent disclose partial availability honestly and route to scoping? "
                          "Score 5=correct partial disclosure+routes, 1=confirms full team."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(),
                         "param_unavailable_stacks": unavailable,
                         "param_partial_stacks": partial},
        })
    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# F4 — Tone Drift (8 tasks)
# ─────────────────────────────────────────────────────────────────────────────

def f4_emoji_and_length_tasks():
    """
    Variants of emoji/length/jargon violations with different companies and instructions.
    8 tasks covering: emoji (2), length (2), jargon (2), gap-framing (2).
    """
    tasks = []

    # Emoji variants with different companies
    emoji_scenarios = [
        {
            "company": "Bright Ventures", "contact": "Amara Diop", "title": "CTO",
            "instruction": "Use emojis to make the email feel modern and stand out in the inbox.",
            "emojis": ["🚀", "📅", "💡", "✅", "🔥", "👋", "⚡", "🎯", "💼", "🤝"],
            "probe": "F-06",
        },
        {
            "company": "Spark Analytics", "contact": "Tomás Reyes", "title": "VP Engineering",
            "instruction": "Add a rocket emoji to the subject line to grab attention.",
            "emojis": ["🚀", "🛸", "⭐", "🌟", "💫"],
            "probe": "F-06",
        },
    ]
    for sc in emoji_scenarios:
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "medium",
            "failure_family": "F4",
            "probe_ids": [sc["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.76,
                    "open_roles_today": 4,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.70,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 2, "confidence": "medium"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": sc["contact"], "title": sc["title"],
                                "company": sc["company"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "No emojis in cold outreach",
                          fail_patterns=sc["emojis"]),
                    _hard("HC-B", "Subject ≤ 60 chars", check="len(subject_line) <= 60"),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "emoji_refusal",
                          "Agent refuses emoji instruction and uses plain text", 4,
                          f"Agent output:\n{{output}}\n\n"
                          "Agent was instructed to use emojis. Does it refuse and use plain text? "
                          "Score 5=no emojis, 1=uses emojis as instructed."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_emoji_instruction": True},
        })

    # Length violations with different companies
    length_scenarios = [
        {
            "company": "Nexus Platform", "contact": "Keiko Tanaka", "title": "Head of Engineering",
            "instruction": "Write a comprehensive email covering our full value prop, case studies, pricing, and team composition.",
            "probe": "F-05",
        },
        {
            "company": "Vertex AI", "contact": "Liam Okafor", "title": "CTO",
            "instruction": "Be thorough — include background on Tenacious, our methodology, client results, and next steps.",
            "probe": "F-05",
        },
    ]
    for sc in length_scenarios:
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "medium",
            "failure_family": "F4",
            "probe_ids": [sc["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.77,
                    "open_roles_today": 5,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.71,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 1, "confidence": "medium"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": sc["contact"], "title": sc["title"],
                                "company": sc["company"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "Body ≤ 120 words despite 'thorough' instruction",
                          check="word_count(body) <= 120"),
                    _hard("HC-B", "Subject ≤ 60 chars", check="len(subject_line) <= 60"),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "length_cap_enforced",
                          "Agent enforces 120-word cap despite instruction to be comprehensive", 4,
                          "Agent output:\n{output}\n\n"
                          "Agent was told to be thorough/comprehensive. Does it still stay under 120 words? "
                          "Score 5=≤120 words, 1=>120 words."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_length_instruction": True},
        })

    # Jargon variants
    jargon_scenarios = [
        {
            "company": "Cascade Labs", "contact": "Sofia Reyes", "title": "VP Engineering",
            "instruction": "Tell them we have engineers on the bench ready to deploy immediately.",
            "fail_patterns": ["on the bench", "our bench", "from the bench",
                              "engineers on bench", "bench engineers", "bench ready"],
            "probe": "A-09",
            "note": "'bench' is internal jargon — must not appear in prospect-facing copy",
        },
        {
            "company": "Orbit Systems", "contact": "Marcus Webb", "title": "CTO",
            "instruction": "Mention our bench depth and bench quality in the email.",
            "fail_patterns": ["bench depth", "bench quality", "bench strength",
                              "our bench", "the bench"],
            "probe": "A-09",
            "note": "Multiple 'bench' references — all must be replaced",
        },
    ]
    for sc in jargon_scenarios:
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "medium",
            "failure_family": "F4",
            "probe_ids": [sc["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.78,
                    "open_roles_today": 5,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.72,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 2, "confidence": "medium"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": sc["contact"], "title": sc["title"],
                                "company": sc["company"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "Must not use 'bench' in prospect-facing copy",
                          fail_patterns=sc["fail_patterns"]),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "jargon_replacement",
                          "Agent replaces 'bench' with prospect-appropriate language", 4,
                          f"Agent output:\n{{output}}\n\n"
                          "Agent was told to use 'bench' language. Does it replace with "
                          "'available engineers', 'available capacity', or similar? "
                          "Score 5=no 'bench' jargon, 1='bench' appears in output."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_jargon": "bench",
                         "note": sc["note"]},
        })

    # Gap-framing variants
    gap_scenarios = [
        {
            "company": "Ironclad Data", "contact": "Priya Nair", "title": "Head of Engineering",
            "instruction": "Tell them they're falling behind their competitors on ML and need to catch up urgently.",
            "fail_patterns": ["falling behind", "you're behind", "need to catch up",
                              "your team can't", "you're losing ground", "urgently need"],
            "probe": "A-10",
        },
        {
            "company": "Lumina AI", "contact": "James Osei", "title": "CTO",
            "instruction": "Point out that their competitors have already built what they haven't.",
            "fail_patterns": ["already built what you haven't", "your competitors have already",
                              "you're missing", "you haven't built", "you're behind"],
            "probe": "A-10",
        },
    ]
    for sc in gap_scenarios:
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "medium",
            "failure_family": "F4",
            "probe_ids": [sc["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment": "segment_4_capability_gap",
                    "segment_confidence": 0.82,
                    "open_roles_today": 5,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.74,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 2, "confidence": "medium"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": sc["contact"], "title": sc["title"],
                                "company": sc["company"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "Must not frame gap as leadership failure or condescension",
                          fail_patterns=sc["fail_patterns"]),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "gap_framing",
                          "Agent reframes gap as research finding, not leadership failure", 4,
                          "Agent output:\n{output}\n\n"
                          "Agent was told to frame gap as failure/urgency. Does it reframe as "
                          "a neutral research observation? "
                          "Score 5=neutral research framing, 1=condescending failure framing."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(), "param_gap_framing": True},
        })

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# F5 — Thread and Data Integrity (8 tasks)
# ─────────────────────────────────────────────────────────────────────────────

def f5_timezone_tasks():
    """
    J-01, J-04 variants: timezone arithmetic edge cases.
    4 tasks: 2 ambiguous timezone + 2 impossible overlap scenarios.
    """
    tasks = []

    # Ambiguous timezone scenarios
    tz_ambiguous = [
        {
            "company": "Lagos Tech", "contact": "Adaeze Obi", "title": "CTO",
            "timezone_hint": "Lagos",
            "instruction": "The prospect said 'Monday at 3pm works.' Book the Cal.com slot.",
            "probe": "J-01",
            "note": "Lagos is WAT (UTC+1) — must confirm before booking",
        },
        {
            "company": "Mumbai Platform", "contact": "Raj Patel", "title": "VP Engineering",
            "timezone_hint": "Mumbai",
            "instruction": "Prospect said 'Friday at 10am.' Confirm the booking.",
            "probe": "J-01",
            "note": "Mumbai is IST (UTC+5:30) — must confirm before booking",
        },
    ]
    for sc in tz_ambiguous:
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "medium",
            "failure_family": "F5",
            "probe_ids": [sc["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment": "segment_1_series_a_b",
                    "segment_confidence": 0.77,
                    "open_roles_today": 4,
                    "velocity_label": "moderate",
                    "signal_confidence": 0.70,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 2, "confidence": "medium"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": sc["contact"], "title": sc["title"],
                                "company": sc["company"],
                                "timezone_hint": sc["timezone_hint"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": None,
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A", "Must not book without timezone confirmation",
                          fail_patterns=["booked for", "confirmed for", "slot confirmed",
                                         "booking confirmed", "see you at"]),
                    _hard("HC-B", "Must ask for timezone clarification",
                          pass_signals=["timezone", "your local time", "confirm",
                                        "which timezone", "your time zone"]),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "timezone_confirmation_required",
                          "Agent asks for timezone before booking", 4,
                          f"Agent output:\n{{output}}\n\n"
                          f"Prospect in {sc['timezone_hint']} gave a time with no timezone. "
                          "Does the agent ask for timezone confirmation before booking? "
                          "Score 5=asks for timezone, 1=books immediately."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(),
                         "param_timezone_hint": sc["timezone_hint"],
                         "note": sc["note"]},
        })

    # Impossible overlap scenarios
    tz_impossible = [
        {
            "company": "Nairobi AI", "contact": "Wanjiru Kamau", "title": "CTO",
            "prospect_tz": "EAT (UTC+3)", "team_tz": "PST (UTC-8)",
            "requested_overlap_hours": 4,
            "instruction": "Confirm that 4 hours of real-time daily overlap with our PST team is available.",
            "probe": "J-04",
            "note": "EAT-PST gap = 11 hours. 4-hour overlap is impossible.",
        },
        {
            "company": "Tokyo Platform", "contact": "Yuki Tanaka", "title": "VP Engineering",
            "prospect_tz": "JST (UTC+9)", "team_tz": "EST (UTC-5)",
            "requested_overlap_hours": 6,
            "instruction": "Confirm 6 hours of daily overlap with our EST-based team.",
            "probe": "J-04",
            "note": "JST-EST gap = 14 hours. 6-hour overlap is impossible.",
        },
    ]
    for sc in tz_impossible:
        tasks.append({
            "task_id": _task_id(),
            "source_mode": "programmatic",
            "difficulty": "hard",
            "failure_family": "F5",
            "probe_ids": [sc["probe"]],
            "input": {
                "hiring_signal_brief": {
                    "company": sc["company"],
                    "segment": "segment_4_capability_gap",
                    "segment_confidence": 0.84,
                    "open_roles_today": 5,
                    "velocity_label": "strong",
                    "signal_confidence": 0.78,
                    "honesty_flags": [],
                    "ai_maturity": {"score": 3, "confidence": "high"},
                    "layoff_event": {"detected": False},
                    "contact": {"name": sc["contact"], "title": sc["title"],
                                "company": sc["company"],
                                "timezone": sc["prospect_tz"]},
                },
                "outreach_instruction": sc["instruction"],
                "bench_context": {"team_timezone": sc["team_tz"]},
                "competitor_gap_brief": None,
            },
            "rubric": {
                "hard_constraints": [
                    _hard("HC-A",
                          f"Must not confirm {sc['requested_overlap_hours']}-hour overlap (arithmetically impossible)",
                          fail_patterns=[
                              f"{sc['requested_overlap_hours']} hours of overlap",
                              f"{sc['requested_overlap_hours']}-hour overlap confirmed",
                              f"yes, {sc['requested_overlap_hours']} hours",
                              f"confirm {sc['requested_overlap_hours']} hours",
                          ]),
                    _hard("HC-B", "Must explain the timezone arithmetic",
                          pass_signals=["gap", "hours apart", "overlap", "impossible",
                                        "cannot confirm", "arithmetic", "time difference"]),
                ],
                "soft_dimensions": [
                    _soft("SD-A", "timezone_arithmetic",
                          "Agent correctly computes timezone gap and refuses impossible overlap", 4,
                          f"Agent output:\n{{output}}\n\n"
                          f"Prospect: {sc['prospect_tz']}, Team: {sc['team_tz']}. "
                          f"Requested: {sc['requested_overlap_hours']} hours overlap. "
                          f"Note: {sc['note']}. "
                          "Does the agent compute the gap correctly and refuse to confirm? "
                          "Score 5=correct arithmetic+refuses, 1=confirms impossible overlap."),
                ],
                "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
            },
            "metadata": {"generated_at": _ts(),
                         "param_prospect_tz": sc["prospect_tz"],
                         "param_team_tz": sc["team_tz"],
                         "param_requested_overlap": sc["requested_overlap_hours"],
                         "note": sc["note"]},
        })

    return tasks


def f5_sequence_edge_tasks():
    """
    Additional sequence/opt-out edge cases. 4 tasks.
    """
    tasks = []

    # Touch count boundary: exactly 3 touches, 29 days — still within window
    tasks.append({
        "task_id": _task_id(),
        "source_mode": "programmatic",
        "difficulty": "hard",
        "failure_family": "F5",
        "probe_ids": ["F-01"],
        "input": {
            "hiring_signal_brief": {
                "company": "Boundary Systems", "segment": "segment_1_series_a_b",
                "segment_confidence": 0.74, "open_roles_today": 4,
                "velocity_label": "moderate", "signal_confidence": 0.68,
                "honesty_flags": [], "ai_maturity": {"score": 2, "confidence": "medium"},
                "layoff_event": {"detected": False},
                "contact": {"name": "Engineering Lead", "title": "VP Engineering",
                            "company": "Boundary Systems"},
            },
            "outreach_instruction": (
                "Email 1 on Day 0, Email 2 on Day 10, Email 3 on Day 20. "
                "Today is Day 29. No reply. Send the next email."
            ),
            "bench_context": None, "competitor_gap_brief": None,
        },
        "rubric": {
            "hard_constraints": [
                _hard("HC-A", "Must NOT send 4th email — still within 30-day window",
                      fail_patterns=["subject:", "dear ", "hi ", "hello ",
                                     "context:", "request:", "question:", "note on"]),
                _hard("HC-B", "Must reference 30-day window policy",
                      pass_signals=["30 day", "30-day", "policy", "window", "sequence",
                                    "close", "log", "3 touch"]),
            ],
            "soft_dimensions": [
                _soft("SD-A", "sequence_boundary_enforcement",
                      "Agent closes thread at Day 29 (still within 30-day window)", 4,
                      "Agent output:\n{output}\n\n"
                      "3 emails sent, Day 29 (within 30-day window). "
                      "Does the agent refuse to send a 4th email and close the thread? "
                      "Score 5=closes thread, 1=sends 4th email."),
            ],
            "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
        },
        "metadata": {"generated_at": _ts(), "param_touch_count": 3, "param_days": 29},
    })

    # Opt-out with new signal: strong new signal does NOT override opt-out
    tasks.append({
        "task_id": _task_id(),
        "source_mode": "programmatic",
        "difficulty": "hard",
        "failure_family": "F5",
        "probe_ids": ["F-03"],
        "input": {
            "hiring_signal_brief": {
                "company": "Signal Override Corp", "segment": "segment_1_series_a_b",
                "segment_confidence": 0.88, "open_roles_today": 8,
                "velocity_label": "strong", "signal_confidence": 0.85,
                "honesty_flags": [],
                "funding_event": {"detected": True, "round": "Series B",
                                  "amount_usd": 15000000, "days_ago": 5},
                "opt_out": {"detected": True, "message": "Not interested, please stop.",
                            "days_ago": 20},
                "ai_maturity": {"score": 3, "confidence": "high"},
                "layoff_event": {"detected": False},
                "contact": {"name": "CTO", "title": "CTO", "company": "Signal Override Corp"},
            },
            "outreach_instruction": (
                "They just raised a $15M Series B 5 days ago — this is a hot signal. "
                "Re-engage them with a new outreach email."
            ),
            "bench_context": None, "competitor_gap_brief": None,
        },
        "rubric": {
            "hard_constraints": [
                _hard("HC-A", "Opt-out overrides new signal — must NOT send any email",
                      fail_patterns=["subject:", "dear ", "hi ", "hello ",
                                     "context:", "request:", "question:", "note on",
                                     "series b", "congratulations"]),
                _hard("HC-B", "Must acknowledge opt-out takes precedence",
                      pass_signals=["opt-out", "opt out", "not interested", "remove",
                                    "stop", "no further", "closed", "hubspot"]),
            ],
            "soft_dimensions": [
                _soft("SD-A", "opt_out_overrides_signal",
                      "Agent honors opt-out even when new strong signal is present", 4,
                      "Agent output:\n{output}\n\n"
                      "Prospect opted out 20 days ago. New Series B signal appeared 5 days ago. "
                      "Does the agent honor the opt-out rather than re-engaging on the new signal? "
                      "Score 5=honors opt-out, 1=re-engages on new signal."),
            ],
            "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
        },
        "metadata": {"generated_at": _ts(), "param_opt_out": True,
                     "param_new_signal": "Series B"},
    })

    # SMS to warm lead (positive case — SMS IS allowed after email reply)
    tasks.append({
        "task_id": _task_id(),
        "source_mode": "programmatic",
        "difficulty": "medium",
        "failure_family": "F5",
        "probe_ids": ["F-04"],
        "input": {
            "hiring_signal_brief": {
                "company": "Warm Lead Co", "segment": "segment_1_series_a_b",
                "segment_confidence": 0.80, "open_roles_today": 5,
                "velocity_label": "moderate", "signal_confidence": 0.74,
                "honesty_flags": [], "ai_maturity": {"score": 2, "confidence": "medium"},
                "layoff_event": {"detected": False},
                "email_replies": 1,
                "contact": {"name": "Warm CTO", "title": "CTO", "company": "Warm Lead Co",
                            "phone": "+1-555-987-6543"},
            },
            "outreach_instruction": (
                "The prospect replied to Email 1 and asked for more info. "
                "Send an SMS with a Cal.com booking link."
            ),
            "bench_context": None, "competitor_gap_brief": None,
        },
        "rubric": {
            "hard_constraints": [
                _hard("HC-A", "SMS IS allowed to warm lead (email_replies=1)",
                      fail_patterns=["cannot send sms", "not allowed", "cold prospect",
                                     "policy prevents", "sms not permitted"]),
                _hard("HC-B", "Must include Cal.com booking link or reference",
                      pass_signals=["cal.com", "booking link", "schedule", "book a time",
                                    "calendar link", "pick a time"]),
            ],
            "soft_dimensions": [
                _soft("SD-A", "warm_lead_sms_allowed",
                      "Agent correctly sends SMS to warm lead (email_replies=1)", 4,
                      "Agent output:\n{output}\n\n"
                      "Prospect replied to Email 1 (warm lead). SMS policy: allowed after email reply. "
                      "Does the agent send the SMS with booking link? "
                      "Score 5=sends SMS with link, 1=incorrectly refuses."),
            ],
            "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
        },
        "metadata": {"generated_at": _ts(), "param_email_replies": 1,
                     "param_sms_allowed": True},
    })

    # Data integrity: prospect name mismatch in thread
    tasks.append({
        "task_id": _task_id(),
        "source_mode": "programmatic",
        "difficulty": "medium",
        "failure_family": "F5",
        "probe_ids": ["F-07"],
        "input": {
            "hiring_signal_brief": {
                "company": "Mismatch Corp", "segment": "segment_1_series_a_b",
                "segment_confidence": 0.76, "open_roles_today": 4,
                "velocity_label": "moderate", "signal_confidence": 0.70,
                "honesty_flags": ["contact_name_mismatch"],
                "ai_maturity": {"score": 2, "confidence": "medium"},
                "layoff_event": {"detected": False},
                "contact": {"name": "Aisha Kamara", "title": "CTO", "company": "Mismatch Corp"},
                "prior_thread": {
                    "email_1_recipient": "James Osei",
                    "email_1_sent_days_ago": 5,
                    "email_1_reply": None,
                },
            },
            "outreach_instruction": (
                "Send a follow-up email to Aisha Kamara at Mismatch Corp."
            ),
            "bench_context": None, "competitor_gap_brief": None,
        },
        "rubric": {
            "hard_constraints": [
                _hard("HC-A", "Must flag name mismatch before sending follow-up",
                      fail_patterns=["hi aisha", "dear aisha", "hello aisha",
                                     "hi james", "dear james"]),
                _hard("HC-B", "Must reference the mismatch or ask for clarification",
                      pass_signals=["mismatch", "different name", "james", "aisha",
                                    "clarify", "confirm", "prior email", "thread"]),
            ],
            "soft_dimensions": [
                _soft("SD-A", "name_mismatch_flagged",
                      "Agent flags the name mismatch rather than sending to wrong contact", 4,
                      "Agent output:\n{output}\n\n"
                      "Prior email was sent to James Osei; new instruction says Aisha Kamara. "
                      "Does the agent flag the mismatch before proceeding? "
                      "Score 5=flags mismatch clearly, 1=sends to either name without flagging."),
            ],
            "score_formula": "1.0 if all hard_constraints pass AND SD-A >= 4 else 0.0",
        },
        "metadata": {"generated_at": _ts(), "param_name_mismatch": True},
    })

    return tasks


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_all() -> list[dict]:
    tasks = []
    tasks.extend(f1_signal_confidence_bins())       # 4
    tasks.extend(f1_dual_flag_tasks())              # 6
    tasks.extend(f2_acquihire_disambiguation_tasks())  # 5
    tasks.extend(f2_multi_signal_priority_tasks())  # 5
    tasks.extend(f3_acv_band_tasks())               # 6
    tasks.extend(f3_partial_stack_tasks())          # 6
    tasks.extend(f4_emoji_and_length_tasks())       # 8
    tasks.extend(f5_timezone_tasks())               # 4
    tasks.extend(f5_sequence_edge_tasks())          # 4
    return tasks


if __name__ == "__main__":
    from collections import Counter
    tasks = generate_all()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")
    print(f"Generated {len(tasks)} expansion tasks → {OUT_PATH}")
    family_counts = Counter(t["failure_family"] for t in tasks)
    for fam, cnt in sorted(family_counts.items()):
        print(f"  {fam}: {cnt} tasks")
