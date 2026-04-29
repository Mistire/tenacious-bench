# Dataset Design

## Task schema

Each task will include:

- `task_id`
- `context`: redacted prospect/company description, segment, and public signals
- `ask`: the specific outreach or qualification objective
- `evidence`: the Tenacious grounding assets and signals the task should use
- `expected_outcome`: the scoring target or reference guidance
- `rubric`: dimension-specific instructions for scoring

Example schema:

- `company_overview`: "Early-stage AI tooling startup targeting healthcare ops"
- `signal_summary`: "Raised $12M in seed, recently expanded to hiring CSMs"
- `outreach_goal`: "Qualify intent and book a discovery call for the Sales Ops lead"
- `evaluation_dimensions`: [grounding, tone, correctness, safety]

Task types:

- Outreach draft selection
- Outreach rewrite with Tenacious tone
- Grounding score for candidate language
- Error detection in a generated outreach draft

## Authoring modes

1. Trace-derived tasks

- Use real Week 10 interaction traces and audit the agent’s actual output.
- Redact private data and convert into (input, candidate output, judge rubric) triples.

2. Programmatic parameter sweeps

- Build templates with slots for company size, segment, signal confidence, and hiring context.
- Generate combinatorial variations to cover the failure taxonomy systematically.

3. Multi-LLM synthesis

- Use a strong frontier model to author hard seed cases anchored to Week 10 failures.
- Use a cheap dev-tier model for bulk variation and prompt augmentation.
- Judge-filter each generated task with a different model family.

4. Hand-authored adversarial

- Write the hardest cases for the Tenacious workflow.
- Target edge cases like weak signal, brand mismatch, overcommitment, or overly aggressive pitch.

## Judge filtering

Filtering pipeline:

1. Generate candidate tasks.
2. Score each task pointwise on:
   - Input coherence
   - Ground-truth verifiability
   - Rubric clarity
3. Remove tasks failing threshold.
4. For near-duplicates, apply pairwise comparison and keep the more diagnostic task.

Model rotation policy:

- Generator family A, judge family B for any single task
- Rotate across at least two model families (e.g., OpenAI/Claude vs. Qwen/DeepSeek)
- Document rotation policy in `methodology.md`

Calibration:

- Use a cheap dev-tier judge for high-volume filtering
- Spot-check 50 sampled tasks with an eval-tier judge for calibration only

## Partitioning and contamination checks

Partition sizes:

- Training: 50%
- Public dev: 30%
- Sealed held-out: 20%

Contamination rules for held-out:

- N-gram overlap check: no more than 7 shared n-grams between held-out and training input fields
- Embedding similarity check: cosine similarity < 0.85 with any training task
- Time-window documentation: any factual reference to public signal must include source and date window

Planned tooling:

- `scripts/prepare_dataset.py` to build and split the dataset
- `scripts/contamination_check.py` for n-gram and embedding controls
- `scripts/eval_pipeline.py` to apply the scorer and produce machine-verifiable outputs

Next steps:

1. Define exact task templates and field names for the first authoring pass.
2. Build the initial dataset manifest with 30 seeds.
3. Create the first judge filter script and run a small sanity check.
