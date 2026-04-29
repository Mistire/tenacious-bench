# Budget Log — Week 11 Tenacious-Bench

All costs in USD. API calls routed through OpenRouter.

| Date | Bucket | Resource | Calls | Tokens (in/out) | Cost (USD) | Purpose | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-04-27 | Synthesis generation | Qwen3-235B-A22B (OpenRouter) | 20 | ~60 K / ~18 K | $0.42 | Generate 20 synthesis seed tasks | 2 rejected by judge filter |
| 2026-04-27 | Synthesis judging | DeepSeek V3-0324 (OpenRouter) | 20 | ~80 K / ~6 K | $0.11 | 3-dimension quality filter (IC, GV, RC) | Model rotation — never same family as generator |
| 2026-04-29 | Scorer validation | DeepSeek V3-0324 (OpenRouter) | 5 | ~12 K / ~3 K | $0.02 | Manual spot-check of scoring_evaluator.py soft-dimension judge | 5 dev tasks |
| **Total** | | | **45** | **~152 K / ~27 K** | **$0.55** | | Well inside $5 budget cap |

## Notes

- Programmatic (55 tasks), trace-derived (36 tasks), and adversarial (47 tasks) were generated locally with no API calls.
- OpenRouter pricing used: Qwen3-235B-A22B at $0.13/M input + $0.60/M output; DeepSeek V3-0324 at $0.27/M input + $1.10/M output.
- Token counts are estimates from the synthesis script's `usage` field logging; exact figures in `synthesis_tasks_raw.jsonl` metadata.
- Held-out partition is sealed and excluded from all API-based post-processing.
