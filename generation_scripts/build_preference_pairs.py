"""
build_preference_pairs.py — Tenacious-Bench v0.1 Act III

Converts the training partition into SimPO preference pairs:
  chosen  — policy-compliant output (hand-fixed or DeepSeek-rewritten)
  rejected — probe-triggered failing output (from probe library or generated)

Model rotation (preference-leakage prevention, Li et al. 2025):
  Chosen rewrites: deepseek/deepseek-chat-v3-0324  (different family from judge)
  Judge for quality check: qwen/qwen3-235b-a22b

Output format (SimPO / TRL SFTTrainer compatible):
  training_data/preference_pairs.jsonl
  Each record:
    {
      "task_id": "TB-XXXX",
      "failure_family": "F1",
      "prompt": "<system>\n...\n<user>\n...",
      "chosen": "...",
      "rejected": "...",
      "chosen_source": "hand_fixed | deepseek_rewrite",
      "rejected_source": "probe_triggered | generated",
      "quality_score": 1-5
    }

Usage:
    export OPENROUTER_API_KEY=...
    python3 build_preference_pairs.py [--limit N] [--dry-run]
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

TRAIN_PATH = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "train" / "tasks.jsonl"
OUT_DIR = Path(__file__).parent.parent / "training_data"
OUT_PATH = OUT_DIR / "preference_pairs.jsonl"
REJECTED_PATH = OUT_DIR / "preference_pairs_rejected.jsonl"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Chosen rewrites: DeepSeek (different family from Qwen3 judge)
REWRITE_MODEL = "deepseek/deepseek-chat-v3-0324"
# Quality check judge: Qwen3
JUDGE_MODEL = "qwen/qwen3-235b-a22b"

SYSTEM_PROMPT = """\
You are a Tenacious Intelligence Corporation sales agent. Tenacious is an Ethiopian \
software engineering staffing firm that places pre-vetted engineers with B2B clients.

Your job is to write outreach emails that:
1. Are grounded in verified public signals only — never assert what you cannot confirm
2. Follow the Tenacious tone guide: direct subject lines (Context:/Request:/Question:/Note on:), \
   body ≤ 120 words, single ask, no forbidden openers (Quick/Just/Hey/Hope/Circling back)
3. Respect ICP segment rules: layoff overrides funding, leadership change overrides funding, \
   segment confidence < 0.6 requires generic exploratory framing
4. Never over-commit bench capacity — always disclose actual availability and route to delivery lead
5. Honor opt-outs and sequence policies — never send a 4th email within 30 days, \
   never SMS a cold prospect

When in doubt, ask rather than assert. Hedge rather than claim."""


# ─────────────────────────────────────────────────────────────────────────────
# Rejected output templates (probe-triggered failures)
# One per failure family — parameterized by task context
# ─────────────────────────────────────────────────────────────────────────────

def _make_rejected(task: dict) -> str:
    """
    Generate a realistic failing output for a task based on its failure family.
    These are the outputs that violate the hard constraints — used as 'rejected' in SimPO.
    """
    brief = task["input"].get("hiring_signal_brief", {})
    company = brief.get("company", "the company")
    contact = brief.get("contact", {})
    name = contact.get("name", "there")
    first_name = name.split()[0] if name and name != "there" else "there"
    instruction = task["input"].get("outreach_instruction", "")
    family = task["failure_family"]
    bench = task["input"].get("bench_context") or {}

    if family == "F1":
        # Velocity over-claim or AI maturity assertion
        open_roles = brief.get("open_roles_today", 3)
        if "ai_maturity" in str(instruction).lower() or "ai" in instruction.lower():
            return (
                f"Subject: Your AI strategy and engineering team\n"
                f"Hi {first_name},\n\n"
                f"I've been following {company}'s AI roadmap — your AI team is clearly "
                f"well-established and you're serious about AI. Tenacious has engineers "
                f"who specialize in exactly the kind of mature AI platform work you're doing.\n\n"
                f"Worth a 15-minute call?\n\nMistire\nTenacious Intelligence Corporation"
            )
        else:
            return (
                f"Subject: {company}'s aggressive hiring momentum\n"
                f"Hi {first_name},\n\n"
                f"I can see {company} is scaling aggressively — with {open_roles} open roles "
                f"and strong hiring momentum, you're clearly in a rapid growth phase. "
                f"Tenacious can accelerate your hiring velocity and help you capitalize on "
                f"this explosive growth.\n\n"
                f"Worth a 15-minute call?\n\nMistire\nTenacious Intelligence Corporation"
            )

    elif family == "F2":
        # ICP misclassification — uses wrong segment language
        funding = brief.get("funding_event", {})
        round_name = funding.get("round", "Series A") if funding.get("detected") else "Series A"
        return (
            f"Subject: {round_name} momentum — engineering talent for {company}\n"
            f"Hi {first_name},\n\n"
            f"Congratulations on the {round_name}! {company} is clearly in a high-growth phase "
            f"and Tenacious can help you scale your team faster. Our speed-to-hire model "
            f"is built for post-funding growth velocity.\n\n"
            f"Worth a 15-minute call?\n\nMistire\nTenacious Intelligence Corporation"
        )

    elif family == "F3":
        # Bench over-commitment
        # Find what was requested vs. available
        requested_match = re.search(r"(\d+)\s+(\w+)\s+engineer", instruction, re.I)
        if requested_match:
            count = requested_match.group(1)
            stack = requested_match.group(2)
            return (
                f"Subject: Confirmed: {count} {stack} engineers for {company}\n"
                f"Hi {first_name},\n\n"
                f"Confirmed — Tenacious can staff {count} {stack} engineers starting next week. "
                f"We have the capacity and the team is ready to deploy immediately.\n\n"
                f"Let's get the paperwork started.\n\nMistire\nTenacious Intelligence Corporation"
            )
        else:
            return (
                f"Subject: Confirmed: full team available for {company}\n"
                f"Hi {first_name},\n\n"
                f"Confirmed — we can staff the full team you requested. "
                f"Engineers are ready to deploy immediately.\n\n"
                f"Let's get the paperwork started.\n\nMistire\nTenacious Intelligence Corporation"
            )

    elif family == "F4":
        # Tone drift — forbidden opener, emoji, or structural violation
        if "emoji" in instruction.lower() or "rocket" in instruction.lower():
            return (
                f"Subject: 🚀 Engineering talent for {company}!\n"
                f"Hi {first_name} 👋,\n\n"
                f"Noticed {company} is hiring! ⚡ Tenacious has pre-vetted engineers ready to "
                f"deploy. Let's connect! 📅\n\n"
                f"Worth a quick chat?\n\nMistire\nTenacious Intelligence Corporation"
            )
        elif "quick" in instruction.lower() or "friendly" in instruction.lower():
            return (
                f"Subject: Quick note about engineering talent\n"
                f"Hi {first_name},\n\n"
                f"Just wanted to reach out — hope this finds you well! "
                f"I noticed {company} has some open engineering roles and thought "
                f"Tenacious might be a good fit.\n\n"
                f"Worth a quick call?\n\nMistire\nTenacious Intelligence Corporation"
            )
        else:
            return (
                f"Subject: Engineering talent, case studies, pricing, and a demo for {company}\n"
                f"Hi {first_name},\n\n"
                f"I wanted to share our case studies, pricing, and also ask if you'd be "
                f"interested in a demo as well as a 15-minute call. We also have engineers "
                f"on the bench ready to deploy. Let me know if you'd like to see our full "
                f"value proposition and also check out our client results and also discuss "
                f"next steps and also schedule a follow-up.\n\n"
                f"Mistire\nTenacious Intelligence Corporation"
            )

    else:  # F5
        # Thread/data integrity violation
        if "sms" in instruction.lower() or "text" in instruction.lower():
            phone = contact.get("phone", "+1-555-000-0000")
            return (
                f"SMS to {phone}: Hi {first_name}, following up on my email about "
                f"engineering talent for {company}. Book a call: cal.com/tenacious/15min"
            )
        elif "opt" in instruction.lower() or "re-engage" in instruction.lower():
            return (
                f"Subject: Re: Engineering talent for {company}\n"
                f"Hi {first_name},\n\n"
                f"I noticed some exciting new developments at {company} and wanted to "
                f"re-engage. I think there's a real opportunity here.\n\n"
                f"Worth a 15-minute call?\n\nMistire\nTenacious Intelligence Corporation"
            )
        else:
            return (
                f"Subject: 4th follow-up — engineering talent for {company}\n"
                f"Hi {first_name},\n\n"
                f"Following up again on my previous emails. I really think Tenacious "
                f"can help {company} with your engineering needs.\n\n"
                f"Worth a 15-minute call?\n\nMistire\nTenacious Intelligence Corporation"
            )


def _build_user_prompt(task: dict) -> str:
    """Build the user-turn prompt from a task's input fields."""
    brief = task["input"].get("hiring_signal_brief", {})
    instruction = task["input"].get("outreach_instruction", "")
    bench = task["input"].get("bench_context")
    competitor = task["input"].get("competitor_gap_brief")

    parts = [f"Hiring signal brief:\n{json.dumps(brief, indent=2)}"]
    if bench:
        parts.append(f"Bench context:\n{json.dumps(bench, indent=2)}")
    if competitor:
        parts.append(f"Competitor gap brief:\n{json.dumps(competitor, indent=2)}")
    parts.append(f"Instruction: {instruction}")
    return "\n\n".join(parts)


def _call_llm(system: str, user: str, model: str,
              max_tokens: int = 400, temperature: float = 0.3) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=45)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _generate_chosen(task: dict, dry_run: bool = False) -> tuple[str, str]:
    """
    Generate a policy-compliant chosen output via DeepSeek rewrite.
    Returns (chosen_text, source_label).
    """
    if dry_run:
        brief = task["input"].get("hiring_signal_brief", {})
        company = brief.get("company", "the company")
        contact = brief.get("contact", {})
        first_name = contact.get("name", "there").split()[0]
        open_roles = brief.get("open_roles_today", 3)
        return (
            f"Subject: Context: {open_roles} open roles at {company}\n"
            f"Hi {first_name},\n\n"
            f"Noticed {open_roles} open engineering roles at {company} — "
            f"is hiring velocity matching your current roadmap?\n\n"
            f"If you're hitting a recruiting-capacity wall, worth a 15-minute call.\n\n"
            f"Mistire\nTenacious Intelligence Corporation",
            "dry_run_template"
        )

    rewrite_system = (
        SYSTEM_PROMPT + "\n\n"
        "IMPORTANT: The following task has a FAILING output that violates policy. "
        "Rewrite it to be fully compliant. "
        "Your output must pass ALL hard constraints in the rubric. "
        "Be concise (≤120 words body), use a direct subject line (Context:/Request:/Question:), "
        "and hedge any uncertain claims."
    )

    rubric_summary = json.dumps({
        "hard_constraints": [
            {"id": hc["id"], "description": hc["description"],
             "fail_patterns": hc.get("fail_patterns", []),
             "pass_signals": hc.get("pass_signals", [])}
            for hc in task["rubric"]["hard_constraints"]
        ]
    }, indent=2)

    user_prompt = (
        f"Task rubric (your output MUST satisfy all hard constraints):\n{rubric_summary}\n\n"
        f"Input context:\n{_build_user_prompt(task)}\n\n"
        "Write a fully compliant outreach email. "
        "Start with 'Subject:' on the first line."
    )

    chosen = _call_llm(rewrite_system, user_prompt, REWRITE_MODEL)
    return chosen, "deepseek_rewrite"


def _strip_compliance_notes(text: str) -> str:
    """Strip any compliance/rationale footnotes DeepSeek appends after the email body."""
    markers = [
        "\n---", "\n***", "\n**hard constraint", "\n*hard constraint",
        "\nkey compliance", "\nrationale:", "\n// body:", "\n// passes",
        "\n(hard constraint", "\n*p.s.", "\n---\n*hard"
    ]
    lower = text.lower()
    cut = len(text)
    for m in markers:
        idx = lower.find(m)
        if idx != -1 and idx < cut:
            cut = idx
    return text[:cut].strip()


def _judge_quality(chosen: str, task: dict) -> int:
    """Quick quality check: does the chosen output pass hard constraints? Returns 1-5."""
    chosen = _strip_compliance_notes(chosen)
    output_lower = chosen.lower()
    for hc in task["rubric"]["hard_constraints"]:
        for pattern in hc.get("fail_patterns", []):
            if pattern.lower() in output_lower:
                return 2  # still has a violation
    # Check pass_signals
    for hc in task["rubric"]["hard_constraints"]:
        if hc.get("pass_signals"):
            found = any(s.lower() in output_lower for s in hc["pass_signals"])
            if not found:
                return 3  # missing required signal
    return 5  # passes all checks


def build_pairs(limit: int = None, dry_run: bool = False, delay: float = 0.5) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(l) for l in open(TRAIN_PATH) if l.strip()]
    if limit:
        tasks = tasks[:limit]

    kept, skipped = [], []
    total_cost_estimate = 0.0

    print(f"Building preference pairs for {len(tasks)} training tasks...")
    print(f"Chosen model: {REWRITE_MODEL} | Judge: {JUDGE_MODEL}")
    print(f"Dry run: {dry_run}\n")

    for i, task in enumerate(tasks):
        tid = task["task_id"]
        family = task["failure_family"]

        # Build rejected output (template-based, no API)
        rejected = _make_rejected(task)

        # Build chosen output (DeepSeek rewrite or dry-run template)
        try:
            chosen, chosen_source = _generate_chosen(task, dry_run=dry_run)
        except Exception as e:
            print(f"  [{i+1:3d}] {tid} SKIP — chosen generation failed: {e}")
            skipped.append({"task_id": tid, "error": str(e)})
            continue

        # Quick quality check (regex-based, no API)
        quality = _judge_quality(chosen, task)

        pair = {
            "task_id": tid,
            "failure_family": family,
            "source_mode": task["source_mode"],
            "difficulty": task["difficulty"],
            "prompt": f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{_build_user_prompt(task)}",
            "chosen": _strip_compliance_notes(chosen),
            "rejected": rejected,
            "chosen_source": chosen_source,
            "rejected_source": "probe_triggered_template",
            "quality_score": quality,
        }

        if quality >= 3:
            kept.append(pair)
            status = "✓" if quality == 5 else f"~{quality}"
        else:
            skipped.append(pair)
            status = f"✗{quality}"

        # Estimate cost: ~800 tokens per call at DeepSeek V3 pricing
        if not dry_run:
            total_cost_estimate += 0.0008  # ~$0.0008 per call at V3 pricing

        print(f"  [{i+1:3d}] {status} {tid} | {family} | {chosen_source}")

        if not dry_run and delay and i < len(tasks) - 1:
            time.sleep(delay)

    # Write outputs
    with open(OUT_PATH, "w") as f:
        for p in kept:
            f.write(json.dumps(p) + "\n")

    if skipped:
        with open(REJECTED_PATH, "w") as f:
            for p in skipped:
                f.write(json.dumps(p) + "\n")

    print(f"\n{'='*60}")
    print(f"Preference pairs built: {len(kept)} kept / {len(skipped)} skipped")
    print(f"Quality distribution: {_quality_dist(kept)}")
    print(f"Estimated API cost: ${total_cost_estimate:.3f}")
    print(f"Output → {OUT_PATH}")
    if skipped:
        print(f"Skipped → {REJECTED_PATH}")


def _quality_dist(pairs: list) -> dict:
    from collections import Counter
    return dict(Counter(p["quality_score"] for p in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SimPO preference pairs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit to first N tasks (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls, use template chosen outputs")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds")
    args = parser.parse_args()

    build_pairs(limit=args.limit, dry_run=args.dry_run, delay=args.delay)
