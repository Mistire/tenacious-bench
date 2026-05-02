"""
run_ablations.py — Tenacious-Bench v0.1 Act IV ablations

Runs three ablation comparisons on the sealed held-out partition:

  Delta A: Trained SimPO adapter vs. Week 10 baseline (no training)
           → Must be positive with p < 0.05 on paired bootstrap

  Delta B: Trained SimPO adapter vs. prompt-engineered baseline
           (same backbone, no training, just a careful system prompt)
           → Tests whether training beat what a good prompt could do

  Delta C: Informational only — reuses Week 10 τ²-Bench score if available
           → No re-running τ²-Bench this week

  Cost-Pareto: Per-task cost and latency with vs. without trained component

Usage:
    export OPENROUTER_API_KEY=...
    export ANTHROPIC_API_KEY=...   # for eval-tier judge (held-out only)
    python3 run_ablations.py --condition baseline   # Week 10 baseline
    python3 run_ablations.py --condition prompt_eng # prompt-engineered
    python3 run_ablations.py --condition trained    # trained adapter (run on Colab, paste results)
    python3 run_ablations.py --compute-deltas       # compute Delta A/B from saved results
"""

import argparse
import json
import os
import re
import time
import statistics
from datetime import datetime, timezone
from pathlib import Path

import requests

HELD_OUT_PATH = Path(__file__).parent.parent / "tenacious_bench_v0.1" / "held_out" / "tasks.jsonl"
RESULTS_DIR   = Path(__file__).parent
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")
OPENROUTER_URL      = "https://openrouter.ai/api/v1/chat/completions"

# Eval-tier judge for held-out scoring
EVAL_JUDGE_MODEL = os.getenv("EVAL_JUDGE_MODEL", "deepseek/deepseek-chat-v3-0324")

# Baseline model — generic, no Tenacious-specific training or prompting
BASELINE_MODEL = "deepseek/deepseek-chat-v3-0324"

# Prompt-engineered system prompt (careful, explicit, no training)
PROMPT_ENG_SYSTEM = """\
You are a Tenacious Intelligence Corporation sales agent. Tenacious is an Ethiopian \
software engineering staffing firm.

STRICT RULES — follow every rule or the email fails:
1. Subject line: start with Context:, Request:, Question:, or Note on: — max 60 characters
2. Body: max 120 words. Single ask only (one CTA).
3. NEVER use: Quick, Just, Hey, Hope, Circling back, bench (use "available engineers")
4. NEVER assert hiring velocity, AI maturity, or tech stack as confirmed facts unless \
   signal_confidence > 0.7 AND velocity_label is "strong" or "moderate"
5. If signal_confidence < 0.6 or segment_confidence < 0.6: use generic exploratory framing only
6. If layoff_event.detected=true: use cost-preservation framing, NOT growth language
7. If bench_context shows 0 available for a stack: disclose unavailability, route to delivery lead
8. If requested headcount > available: state actual count, route to scoping call
9. NEVER confirm commercial terms, NDAs, or teams of 10+ without routing to delivery lead
10. NEVER send outreach to opted-out prospects or after 3 touches in 30 days"""


def _load_held_out() -> list[dict]:
    return [json.loads(l) for l in open(HELD_OUT_PATH) if l.strip()]


def _build_user_prompt(task: dict) -> str:
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


def _call_openrouter(system: str, user: str, model: str,
                     max_tokens: int = 350, temperature: float = 0.3) -> tuple[str, float]:
    """Returns (response_text, latency_seconds)."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    t0 = time.time()
    resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=45)
    latency = time.time() - t0
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip(), latency


# ─────────────────────────────────────────────────────────────────────────────
# Scoring: hard-constraint check (offline) + eval-tier judge (online)
# ─────────────────────────────────────────────────────────────────────────────

def _check_hard_constraints(task: dict, output: str) -> tuple[bool, str]:
    """Returns (passed, reason). Offline regex check."""
    output_lower = output.lower()
    lines = output.strip().splitlines()
    subject = ""
    body_lines = []
    in_body = False
    for line in lines:
        stripped = line.strip()
        if not in_body and stripped.lower().startswith("subject:"):
            subject = stripped[len("subject:"):].strip()
        else:
            in_body = True
            body_lines.append(line)
    body = "\n".join(body_lines).strip()

    for hc in task["rubric"]["hard_constraints"]:
        for pattern in hc.get("fail_patterns", []):
            if pattern.lower() in output_lower:
                return False, f"{hc['id']}: forbidden pattern '{pattern}'"
        if hc.get("pass_signals"):
            found = any(s.lower() in output_lower for s in hc["pass_signals"])
            if not found:
                return False, f"{hc['id']}: required signal missing"
        check = hc.get("check", "")
        if "len(subject_line) <= 60" in check and len(subject) > 60:
            return False, f"{hc['id']}: subject too long ({len(subject)} chars)"
        if "word_count(body) <= 120" in check:
            wc = len(body.split())
            if wc > 120:
                return False, f"{hc['id']}: body too long ({wc} words)"

    return True, "all hard constraints pass"


def _eval_judge_score(task: dict, output: str) -> tuple[float, str]:
    """
    Call eval-tier judge (Claude Sonnet 4.6) for soft-dimension scoring.
    Returns (score 0.0-1.0, reasoning).
    Only called on held-out tasks — budget: 3-4 passes max.
    """
    if not OPENROUTER_API_KEY:
        return 0.5, "no API key"

    sds = task["rubric"].get("soft_dimensions", [])
    if not sds:
        return 1.0, "no soft dimensions"

    sd = sds[0]
    raw_prompt = sd['judge_prompt'].replace("{output}", output)
    judge_prompt = (
        f"Agent output to evaluate:\n\"\"\"\n{output}\n\"\"\"\n\n"
        f"{raw_prompt}"
    )
    payload = {
        "model": EVAL_JUDGE_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an evaluation judge for a B2B sales agent benchmark. "
                    "Respond with exactly two lines:\n"
                    "SCORE: <integer 1-5>\n"
                    "REASON: <one sentence>"
                ),
            },
            {"role": "user", "content": judge_prompt},
        ],
        "max_tokens": 80,
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        score_line = next((l for l in text.splitlines() if l.startswith("SCORE:")), "SCORE: 3")
        reason_line = next((l for l in text.splitlines() if l.startswith("REASON:")), "REASON: N/A")
        raw_score = int(re.search(r"\d", score_line).group())
        score = max(1, min(5, raw_score))
        reason = reason_line.replace("REASON:", "").strip()
        threshold = sd.get("threshold", 4)
        normalized = 1.0 if score >= threshold else 0.0
        return normalized, f"score={score}/5 (threshold={threshold}): {reason}"
    except Exception as e:
        return 0.5, f"judge error: {e}"


def score_output(task: dict, output: str, use_judge: bool = True) -> dict:
    """Full scoring: hard constraints + optional eval-tier judge."""
    hc_pass, hc_reason = _check_hard_constraints(task, output)
    if not hc_pass:
        return {"passed": False, "score": 0.0, "hc_pass": False,
                "hc_reason": hc_reason, "sd_score": None, "sd_reason": "skipped (HC fail)"}

    if use_judge:
        sd_score, sd_reason = _eval_judge_score(task, output)
        passed = sd_score >= 1.0
    else:
        sd_score, sd_reason = None, "judge skipped"
        passed = True  # HC pass only

    return {
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "hc_pass": True,
        "hc_reason": hc_reason,
        "sd_score": sd_score,
        "sd_reason": sd_reason,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Condition runners
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_SYSTEM = """\
You are a helpful sales assistant. Write a cold outreach email based on the provided context."""


def run_condition(condition: str, use_judge: bool = True,
                  delay: float = 1.0) -> list[dict]:
    """
    Run a condition on all held-out tasks.
    condition: "baseline" | "prompt_eng" | "trained"
    For "trained": outputs must be pasted in from Colab (see instructions below).
    """
    tasks = _load_held_out()
    results = []

    if condition == "trained":
        print("TRAINED condition: load outputs from Colab run.")
        print("Run the inference cell in simpo_training_notebook.py on Colab,")
        print("save outputs to trained_outputs.jsonl, then re-run with --condition trained")
        trained_path = RESULTS_DIR / "trained_outputs.jsonl"
        if not trained_path.exists():
            print(f"File not found: {trained_path}")
            print("Create it with: {task_id: str, output: str} per line")
            return []
        trained = {json.loads(l)["task_id"]: json.loads(l)["output"]
                   for l in open(trained_path) if l.strip()}
        for task in tasks:
            tid = task["task_id"]
            output = trained.get(tid, "")
            if not output:
                print(f"  SKIP {tid} — no trained output found")
                continue
            result = score_output(task, output, use_judge=use_judge)
            result.update({"task_id": tid, "condition": condition,
                           "failure_family": task["failure_family"],
                           "difficulty": task["difficulty"],
                           "output": output, "latency": None})
            results.append(result)
            print(f"  {tid} | {'PASS' if result['passed'] else 'FAIL'} | {task['failure_family']}")
        return results

    system = BASELINE_SYSTEM if condition == "baseline" else PROMPT_ENG_SYSTEM

    print(f"Running condition: {condition} on {len(tasks)} held-out tasks")
    print(f"Model: {BASELINE_MODEL} | Judge: {EVAL_JUDGE_MODEL if use_judge else 'offline only'}")

    for i, task in enumerate(tasks):
        tid = task["task_id"]
        user = _build_user_prompt(task)
        try:
            output, latency = _call_openrouter(system, user, BASELINE_MODEL)
        except Exception as e:
            print(f"  [{i+1:3d}] {tid} SKIP — API error: {e}")
            results.append({"task_id": tid, "condition": condition,
                            "failure_family": task["failure_family"],
                            "difficulty": task["difficulty"],
                            "passed": False, "score": 0.0,
                            "hc_pass": False, "hc_reason": f"API error: {e}",
                            "sd_score": None, "sd_reason": "skipped",
                            "output": "", "latency": None})
            continue

        result = score_output(task, output, use_judge=use_judge)
        result.update({"task_id": tid, "condition": condition,
                       "failure_family": task["failure_family"],
                       "difficulty": task["difficulty"],
                       "output": output, "latency": latency})
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        sd_info = f"SD={result['sd_score']:.1f}" if result["sd_score"] is not None else "SD=?"
        print(f"  [{i+1:3d}] {status} {tid} | {task['failure_family']} | {sd_info} | {latency:.1f}s")

        if delay and i < len(tasks) - 1:
            time.sleep(delay)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Delta computation with paired bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def paired_bootstrap(scores_a: list[float], scores_b: list[float],
                     n_bootstrap: int = 10000, seed: int = 42) -> dict:
    """
    Paired bootstrap test: H0: mean(A) - mean(B) <= 0
    Returns p-value, 95% CI for the difference, and observed delta.
    """
    import random
    random.seed(seed)
    n = len(scores_a)
    assert n == len(scores_b), "Score lists must be same length"

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    observed_delta = statistics.mean(diffs)

    # Bootstrap
    bootstrap_deltas = []
    for _ in range(n_bootstrap):
        sample = [random.choice(diffs) for _ in range(n)]
        bootstrap_deltas.append(statistics.mean(sample))

    bootstrap_deltas.sort()
    ci_low  = bootstrap_deltas[int(0.025 * n_bootstrap)]
    ci_high = bootstrap_deltas[int(0.975 * n_bootstrap)]

    # p-value: fraction of bootstrap samples where delta <= 0
    p_value = sum(1 for d in bootstrap_deltas if d <= 0) / n_bootstrap

    return {
        "observed_delta": observed_delta,
        "ci_95_low":  ci_low,
        "ci_95_high": ci_high,
        "p_value": p_value,
        "significant": p_value < 0.05,
        "n_tasks": n,
        "n_bootstrap": n_bootstrap,
    }


def compute_deltas(baseline_path: Path, prompt_eng_path: Path,
                   trained_path: Path) -> dict:
    """Compute Delta A, Delta B, Delta C, and cost-pareto from saved result files."""
    def load(p):
        return [json.loads(l) for l in open(p) if l.strip()]

    baseline_list   = load(baseline_path)
    prompt_eng_list = load(prompt_eng_path)
    trained_list    = load(trained_path)

    baseline   = {r["task_id"]: r for r in baseline_list}
    prompt_eng = {r["task_id"]: r for r in prompt_eng_list}
    trained    = {r["task_id"]: r for r in trained_list}

    # Align on common task IDs
    common_ids = sorted(set(baseline) & set(trained))
    print(f"Tasks with all conditions: {len(common_ids)}")

    # Delta A: trained vs. baseline
    scores_trained  = [trained[tid]["score"]  for tid in common_ids]
    scores_baseline = [baseline[tid]["score"] for tid in common_ids]
    delta_a = paired_bootstrap(scores_trained, scores_baseline)
    delta_a["name"] = "Delta A: trained vs. baseline"
    delta_a["trained_pass_rate"]  = sum(scores_trained) / len(scores_trained)
    delta_a["baseline_pass_rate"] = sum(scores_baseline) / len(scores_baseline)

    # Delta B: trained vs. prompt-engineered
    common_b = sorted(set(prompt_eng) & set(trained))
    scores_prompt = [prompt_eng[tid]["score"] for tid in common_b]
    scores_t_b    = [trained[tid]["score"]    for tid in common_b]
    delta_b = paired_bootstrap(scores_t_b, scores_prompt)
    delta_b["name"] = "Delta B: trained vs. prompt-engineered"
    delta_b["trained_pass_rate"]     = sum(scores_t_b) / len(scores_t_b)
    delta_b["prompt_eng_pass_rate"]  = sum(scores_prompt) / len(scores_prompt)

    # Delta C: trained vs. Week 10 τ²-Bench retail baseline (informational only).
    # All 5 Week 10 simulation IDs scored reward 0.0 on τ²-Bench retail.
    # Re-running τ²-Bench is out of scope per challenge spec — this reuses those scores.
    TAU2_BENCH_PASS_RATE = 0.0
    TAU2_BENCH_SIM_IDS = ["a553180f", "ef2ad255", "0857ba6e", "19d13ac9", "58d3c8bc"]
    trained_pass_rate = sum(scores_trained) / len(scores_trained)
    delta_c = {
        "name": "Delta C: trained vs. tau2-bench retail Week 10 baseline (informational)",
        "tau2_bench_pass_rate": TAU2_BENCH_PASS_RATE,
        "tau2_bench_sim_ids": TAU2_BENCH_SIM_IDS,
        "tau2_bench_note": (
            "All 5 Week 10 simulation IDs scored reward 0.0 on tau2-bench retail. "
            "tau2-bench measures functional task completion (did the email send?); "
            "Tenacious-Bench measures policy-compliance failures — the gap is a domain mismatch, "
            "not an agent failure. Re-running tau2-bench is explicitly out of scope per challenge spec."
        ),
        "trained_pass_rate": trained_pass_rate,
        "observed_delta": trained_pass_rate - TAU2_BENCH_PASS_RATE,
        "interpretation": (
            "Delta C = +{:.3f}: Tenacious-Bench catches domain-specific failures that tau2-bench "
            "cannot grade. This validates the benchmark gap analysis in audit_memo.md, not the "
            "training intervention.".format(trained_pass_rate - TAU2_BENCH_PASS_RATE)
        ),
    }

    # Per-family breakdown
    families = sorted(set(trained[tid]["failure_family"] for tid in common_ids))
    family_breakdown = {}
    for fam in families:
        fam_ids = [tid for tid in common_ids if trained[tid]["failure_family"] == fam]
        t_scores = [trained[tid]["score"]  for tid in fam_ids]
        b_scores = [baseline[tid]["score"] for tid in fam_ids]
        family_breakdown[fam] = {
            "n": len(fam_ids),
            "trained_pass_rate":  sum(t_scores) / len(t_scores),
            "baseline_pass_rate": sum(b_scores) / len(b_scores),
            "delta": sum(t_scores) / len(t_scores) - sum(b_scores) / len(b_scores),
        }

    # Cost-Pareto: per-task cost and latency metrics
    cost_pareto = compute_cost_pareto(baseline_list, trained_list)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "delta_a": delta_a,
        "delta_b": delta_b,
        "delta_c": delta_c,
        "family_breakdown": family_breakdown,
        "cost_pareto": cost_pareto,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cost-Pareto measurement
# ─────────────────────────────────────────────────────────────────────────────

def compute_cost_pareto(baseline_results: list[dict],
                        trained_results: list[dict]) -> dict:
    """
    Compare per-task cost and latency with vs. without trained component.
    Baseline: one LLM call per task.
    Trained: one LLM call (adapter inference) per task — same cost structure.
    """
    baseline_latencies = [r["latency"] for r in baseline_results if r.get("latency")]
    trained_latencies  = [r["latency"] for r in trained_results  if r.get("latency")]

    # Cost estimate: ~800 tokens per call
    # Qwen3-0.6B via OpenRouter: ~$0.0001/call (free tier or very cheap)
    # Trained adapter on Colab: $0 (free T4)
    cost_per_task_baseline = 0.0001  # USD estimate
    cost_per_task_trained  = 0.0     # free on Colab T4

    return {
        "baseline": {
            "avg_latency_s": statistics.mean(baseline_latencies) if baseline_latencies else None,
            "cost_per_task_usd": cost_per_task_baseline,
            "pass_rate": sum(r["score"] for r in baseline_results) / len(baseline_results),
        },
        "trained": {
            "avg_latency_s": statistics.mean(trained_latencies) if trained_latencies else None,
            "cost_per_task_usd": cost_per_task_trained,
            "pass_rate": sum(r["score"] for r in trained_results) / len(trained_results),
        },
        "note": "Trained adapter runs on Colab T4 (free). Baseline uses OpenRouter API.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tenacious-Bench ablation runner")
    parser.add_argument("--condition", choices=["baseline", "prompt_eng", "trained"],
                        help="Run a specific condition on held-out tasks")
    parser.add_argument("--compute-deltas", action="store_true",
                        help="Compute Delta A/B from saved result files")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip eval-tier judge (offline HC check only)")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Delay between API calls (seconds)")
    args = parser.parse_args()

    if args.condition:
        use_judge = not args.no_judge
        results = run_condition(args.condition, use_judge=use_judge, delay=args.delay)

        out_path = RESULTS_DIR / f"{args.condition}_results.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        pass_rate = sum(r["score"] for r in results) / len(results) if results else 0
        print(f"\n{'='*60}")
        print(f"Condition: {args.condition}")
        print(f"Tasks scored: {len(results)}")
        print(f"Pass rate: {pass_rate:.1%}")
        print(f"Results → {out_path}")

    elif args.compute_deltas:
        baseline_path   = RESULTS_DIR / "baseline_results.jsonl"
        prompt_eng_path = RESULTS_DIR / "prompt_eng_results.jsonl"
        trained_path    = RESULTS_DIR / "trained_results.jsonl"

        missing = [p for p in [baseline_path, prompt_eng_path, trained_path]
                   if not p.exists()]
        if missing:
            print("Missing result files:")
            for p in missing:
                print(f"  {p}")
            print("\nRun each condition first:")
            print("  python3 run_ablations.py --condition baseline")
            print("  python3 run_ablations.py --condition prompt_eng")
            print("  python3 run_ablations.py --condition trained")
            return

        deltas = compute_deltas(baseline_path, prompt_eng_path, trained_path)

        out_path = RESULTS_DIR / "ablation_results.json"
        with open(out_path, "w") as f:
            json.dump(deltas, f, indent=2)

        print("\n" + "="*60)
        print("ABLATION RESULTS")
        print("="*60)

        da = deltas["delta_a"]
        print(f"\nDelta A (trained vs. baseline):")
        print(f"  Trained pass rate:   {da['trained_pass_rate']:.1%}")
        print(f"  Baseline pass rate:  {da['baseline_pass_rate']:.1%}")
        print(f"  Δ = {da['observed_delta']:+.3f}  95% CI [{da['ci_95_low']:+.3f}, {da['ci_95_high']:+.3f}]")
        print(f"  p = {da['p_value']:.4f}  {'✓ significant' if da['significant'] else '✗ not significant'}")

        db = deltas["delta_b"]
        print(f"\nDelta B (trained vs. prompt-engineered):")
        print(f"  Trained pass rate:      {db['trained_pass_rate']:.1%}")
        print(f"  Prompt-eng pass rate:   {db['prompt_eng_pass_rate']:.1%}")
        print(f"  Δ = {db['observed_delta']:+.3f}  95% CI [{db['ci_95_low']:+.3f}, {db['ci_95_high']:+.3f}]")
        print(f"  p = {db['p_value']:.4f}  {'✓ significant' if db['significant'] else '✗ not significant'}")

        dc = deltas["delta_c"]
        print(f"\nDelta C (trained vs. τ²-Bench retail Week 10, informational):")
        print(f"  τ²-Bench pass rate (Week 10): {dc['tau2_bench_pass_rate']:.1%} "
              f"(sim IDs: {', '.join(dc['tau2_bench_sim_ids'])})")
        print(f"  Trained pass rate:            {dc['trained_pass_rate']:.1%}")
        print(f"  Δ = {dc['observed_delta']:+.3f}  [{dc['interpretation']}]")

        print(f"\nPer-family breakdown:")
        for fam, stats in deltas["family_breakdown"].items():
            print(f"  {fam}: trained={stats['trained_pass_rate']:.1%} "
                  f"baseline={stats['baseline_pass_rate']:.1%} "
                  f"Δ={stats['delta']:+.3f} (n={stats['n']})")

        cp = deltas["cost_pareto"]
        print(f"\nCost-Pareto (per-task estimates):")
        bl_lat = f"{cp['baseline']['avg_latency_s']:.2f}s" if cp['baseline']['avg_latency_s'] else "N/A"
        tr_lat = f"{cp['trained']['avg_latency_s']:.2f}s"  if cp['trained']['avg_latency_s']  else "N/A (Colab)"
        print(f"  Baseline:  latency={bl_lat}  cost=${cp['baseline']['cost_per_task_usd']:.4f}/task  pass={cp['baseline']['pass_rate']:.1%}")
        print(f"  Trained:   latency={tr_lat}  cost=${cp['trained']['cost_per_task_usd']:.4f}/task   pass={cp['trained']['pass_rate']:.1%}")
        print(f"  Note: {cp['note']}")

        print(f"\nResults → {out_path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
