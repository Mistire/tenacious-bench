"""
colab_inference_cell.py
=======================
Paste this into a Colab cell AFTER running the training notebook.
It runs the trained adapter on all held-out tasks and saves outputs
to trained_outputs.jsonl — which run_ablations.py then scores.

Upload held_out_tasks.jsonl to Colab alongside preference_pairs.jsonl.
"""

COLAB_INFERENCE_CELL = """
import json, torch
from pathlib import Path
from unsloth import FastLanguageModel

# ── Load trained adapter ──────────────────────────────────────────────────────
OUTPUT_DIR   = "/content/tenacious_simpo_adapter"
HELD_OUT_PATH = "/content/held_out_tasks.jsonl"
OUT_PATH      = "/content/trained_outputs.jsonl"
MAX_NEW_TOKENS = 350
TEMPERATURE    = 0.3

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=OUTPUT_DIR,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=False,
)
FastLanguageModel.for_inference(model)

SYSTEM = (
    "You are a Tenacious Intelligence Corporation sales agent. "
    "Write policy-compliant outreach emails."
)

def run_task(task: dict) -> str:
    brief = task["input"].get("hiring_signal_brief", {})
    instruction = task["input"].get("outreach_instruction", "")
    bench = task["input"].get("bench_context")

    user = f"Hiring signal brief:\\n{json.dumps(brief, indent=2)}"
    if bench:
        user += f"\\n\\nBench context:\\n{json.dumps(bench, indent=2)}"
    user += f"\\n\\nInstruction: {instruction}"

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": user},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

# ── Run all held-out tasks ────────────────────────────────────────────────────
tasks = [json.loads(l) for l in open(HELD_OUT_PATH) if l.strip()]
print(f"Running inference on {len(tasks)} held-out tasks...")

outputs = []
for i, task in enumerate(tasks):
    output = run_task(task)
    outputs.append({"task_id": task["task_id"], "output": output})
    print(f"  [{i+1:3d}] {task['task_id']} | {task['failure_family']}")

with open(OUT_PATH, "w") as f:
    for o in outputs:
        f.write(json.dumps(o) + "\\n")

print(f"\\nOutputs saved → {OUT_PATH}")
print("Download this file and place it in tenacious-bench/ablations/trained_outputs.jsonl")
print("Then run: python3 run_ablations.py --condition trained")
print("Then run: python3 run_ablations.py --compute-deltas")
"""

if __name__ == "__main__":
    print("Colab Inference Cell")
    print("=" * 60)
    print("Paste the following into a Colab cell after training:")
    print()
    print(COLAB_INFERENCE_CELL)
