"""
Tenacious-Bench v0.1 — SimPO Training Notebook
================================================
Path B: Preference-tuned judge / critic on Qwen 2.5 0.5B via Unsloth

HARDWARE REQUIREMENTS:
  GPU:       NVIDIA T4 (15 GB VRAM) — Google Colab free tier
  RAM:       12 GB minimum
  Precision: fp16 (T4 does not support bf16)
  Disk:      ~3 GB for model weights + adapter

PINNED LIBRARY VERSIONS (tested 2026-04-30 on Colab T4):
  unsloth_zoo>=0.0.3   (must install first — Unsloth kernel)
  unsloth[colab-new]   (auto-pins compatible transformers/peft)
  trl==0.22.0          (CPOConfig/CPOTrainer — SimPOConfig removed in TRL 1.x)
  datasets==3.5.0

NOTE ON TRL VERSION:
  TRL >= 1.0 removed SimPOConfig. We use CPOConfig(loss_type="simpo", cpo_alpha=0.0)
  which is the correct SimPO implementation in trl==0.22.0.

EXPECTED RUNTIMES (free Colab T4, fp16):
  Cell 0 (install):                   6-10 min  (kernel compile, first run only)
  Cell 2 (load model + LoRA):         2-3 min
  Cell 4 (training, 3 epochs/90 pairs): ~2-4 min  (actual: ~102 s observed)
  Cell 6 (sanity inference, 3 tasks): ~30 s
  Cell 7 (full held-out, 41 tasks):   ~5-8 min
  Total wall time:                    ~20-30 min

INSTRUCTIONS FOR COLAB:
1. Runtime → Change runtime type → T4 GPU
2. Run Cell 0 (install) — takes 6-10 min on first run; restart runtime when prompted
3. Upload preference_pairs.jsonl to /content/
4. Run Cells 1 → 2 → 3 → 4 → 5 → 6 → 7 in order
5. Download trained_outputs.jsonl from /content/ after Cell 7
6. Adapter is pushed to HuggingFace at end of Cell 5

Cost: $0 on free Colab T4
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 0 — Install (run once, ~6-10 min; restart runtime after)
# ─────────────────────────────────────────────────────────────────────────────
CELL_0 = """
%%capture
# Install order matters: unsloth_zoo must precede unsloth
!pip install "unsloth_zoo>=0.0.3"
!pip install "unsloth[colab-new]"
!pip install "trl==0.22.0"
!pip install "datasets==3.5.0"

# Verify versions
import trl, peft, unsloth, datasets as ds
print(f"trl={trl.__version__}  peft={peft.__version__}  unsloth={unsloth.__version__}  datasets={ds.__version__}")
# Expected: trl=0.22.0  datasets=3.5.0
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Imports and config
# ─────────────────────────────────────────────────────────────────────────────
CELL_1 = """
import json, os, random, torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel

# TRL >= 1.x removed SimPOConfig; use CPOConfig(loss_type="simpo", cpo_alpha=0.0) instead
from trl import CPOConfig, CPOTrainer

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ── Model ────────────────────────────────────────────────────────────────────
# IMPORTANT: Use Qwen hub path, NOT the unsloth/ hub path (unsloth/Qwen2.5-0.5B-Instruct
# does not exist). Pin MODEL_REVISION to a specific commit hash for reproducibility;
# "main" resolves to whatever HEAD is at training time.
#
# To pin to the exact checkpoint used in the published run:
#   1. Visit https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/commits/main
#   2. Copy the 40-char commit SHA
#   3. Set MODEL_REVISION = "<that SHA>"
MODEL_NAME     = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "main"   # pin to a specific commit SHA for full reproducibility

# ── LoRA ─────────────────────────────────────────────────────────────────────
MAX_SEQ_LEN  = 1024
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODS  = ["q_proj", "v_proj"]

# ── Training ─────────────────────────────────────────────────────────────────
EPOCHS     = 3
BATCH_SIZE = 2       # T4-safe (15 GB VRAM, fp16)
GRAD_ACCUM = 4       # effective batch = 8
LR         = 5e-5
BETA       = 2.0     # SimPO temperature
GAMMA      = 0.5     # SimPO margin

# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "/content/tenacious_simpo_adapter"
HF_REPO    = os.getenv("HF_MODEL_REPO", "mistire37/tenacious-bench-lora-adapter")
HF_TOKEN   = os.getenv("HF_TOKEN", "")
DATA_PATH  = "/content/preference_pairs.jsonl"

print("Config loaded.")
print(f"  Model:    {MODEL_NAME} (revision={MODEL_REVISION})")
print(f"  LoRA:     r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}")
print(f"  Training: epochs={EPOCHS} lr={LR} beta={BETA} gamma={GAMMA}")
print(f"  Hardware: T4 GPU required (fp16, 15GB VRAM)")
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Load model with Unsloth
# ─────────────────────────────────────────────────────────────────────────────
CELL_2 = """
# Expected: ~2-3 min on T4 (model download + LoRA setup)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    revision=MODEL_REVISION,   # pinned revision for reproducibility
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,          # auto: fp16 on T4, bf16 on A100/L4
    load_in_4bit=False,  # 16-bit LoRA — required for SimPO gradient quality
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=TARGET_MODS,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

print("Model loaded.")
model.print_trainable_parameters()
# Expected: ~819,200 trainable params (0.16% of 514M total)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load and format dataset
# ─────────────────────────────────────────────────────────────────────────────
CELL_3 = """
def format_pair(row):
    \"\"\"Convert a preference pair into CPOTrainer format (prompt, chosen, rejected).\"\"\"
    prompt_text = row.get("prompt") or ""   # guard against None prompt field

    if "<|system|>" in prompt_text and "<|user|>" in prompt_text:
        sys_part = prompt_text.split("<|system|>")[1].split("<|user|>")[0].strip()
        usr_part = prompt_text.split("<|user|>")[1].strip()
    else:
        sys_part = ""
        usr_part = prompt_text.strip()

    messages_prompt = [
        {"role": "system", "content": sys_part},
        {"role": "user",   "content": usr_part},
    ]
    messages_chosen   = messages_prompt + [{"role": "assistant", "content": row["chosen"]}]
    messages_rejected = messages_prompt + [{"role": "assistant", "content": row["rejected"]}]

    prompt_str   = tokenizer.apply_chat_template(messages_prompt,   tokenize=False, add_generation_prompt=True)
    chosen_str   = tokenizer.apply_chat_template(messages_chosen,   tokenize=False)
    rejected_str = tokenizer.apply_chat_template(messages_rejected, tokenize=False)

    return {
        "prompt":         prompt_str,
        "chosen":         chosen_str[len(prompt_str):],
        "rejected":       rejected_str[len(prompt_str):],
        "task_id":        row["task_id"],
        "failure_family": row["failure_family"],
    }

raw = [r for r in (json.loads(l) for l in open(DATA_PATH) if l.strip())
       if r.get("prompt")]   # pre-filter rows with missing prompt
print(f"Loaded {len(raw)} preference pairs (None-prompt rows dropped)")

dataset = Dataset.from_list(raw)
dataset = dataset.map(format_pair, remove_columns=dataset.column_names)

split    = dataset.train_test_split(test_size=0.1, seed=SEED)
train_ds = split["train"]
val_ds   = split["test"]
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
# Expected with 102 pairs: Train: 91 | Val: 11  (actual run: 90 | 11)
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Training
# ─────────────────────────────────────────────────────────────────────────────
CELL_4 = """
# CPOConfig with loss_type="simpo" is the correct SimPO implementation in TRL 0.22.
# cpo_alpha=0.0 disables the BC-regularisation term, giving pure SimPO.
training_args = CPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    loss_type="simpo",
    beta=BETA,
    simpo_gamma=GAMMA,
    cpo_alpha=0.0,
    max_length=MAX_SEQ_LEN,
    max_prompt_length=768,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=not torch.cuda.is_bf16_supported(),   # fp16=True on T4
    bf16=torch.cuda.is_bf16_supported(),       # bf16=True on A100/L4
    seed=SEED,
    report_to="none",
)

trainer = CPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=tokenizer,
)

print("Starting SimPO training (CPOTrainer, loss_type=simpo)...")
print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM} (eff batch=8)")
print(f"  LR: {LR} | Beta: {BETA} | Gamma: {GAMMA} | cpo_alpha: 0.0")
print(f"  Expected runtime on T4: ~2-4 min for 90 pairs x 3 epochs")

train_result = trainer.train()

print(f"\\nTraining complete!")
print(f"  Train loss: {train_result.training_loss:.4f}")
print(f"  Runtime:    {train_result.metrics['train_runtime']:.0f}s")
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Save loss log and push adapter to HuggingFace
# ─────────────────────────────────────────────────────────────────────────────
CELL_5 = """
import json
from pathlib import Path

log = {
    "seed": SEED,
    "model": MODEL_NAME,
    "model_revision": MODEL_REVISION,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT,
    "target_modules": TARGET_MODS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "lr": LR,
    "beta": BETA,
    "gamma": GAMMA,
    "cpo_alpha": 0.0,
    "loss_type": "simpo",
    "train_pairs": len(train_ds),
    "val_pairs": len(val_ds),
    "train_loss": train_result.training_loss,
    "metrics": train_result.metrics,
    "history": [
        {"step": x["step"], "loss": x.get("loss"), "eval_loss": x.get("eval_loss")}
        for x in trainer.state.log_history
    ],
}

log_path = Path(OUTPUT_DIR) / f"training_run_seed{SEED}.json"
log_path.parent.mkdir(parents=True, exist_ok=True)
with open(log_path, "w") as f:
    json.dump(log, f, indent=2)
print(f"Training log → {log_path}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved → {OUTPUT_DIR}")

if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"Adapter pushed → https://huggingface.co/{HF_REPO}")
else:
    print("HF_TOKEN not set — skipping push.")
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Sanity check: inference on 3 tasks
# ─────────────────────────────────────────────────────────────────────────────
CELL_6 = """
# Quick 3-task smoke test to verify the adapter generates coherent output.
# For full ablation scoring, run Cell 7 (all 41 held-out tasks).
HELD_OUT_PATH = "/content/held_out_tasks.jsonl"

FastLanguageModel.for_inference(model)

def run_inference(task: dict, max_new_tokens: int = 300) -> str:
    brief       = task["input"].get("hiring_signal_brief", {})
    instruction = task["input"].get("outreach_instruction", "")
    bench       = task["input"].get("bench_context")

    user_content = f"Hiring signal brief:\\n{json.dumps(brief, indent=2)}"
    if bench:
        user_content += f"\\n\\nBench context:\\n{json.dumps(bench, indent=2)}"
    user_content += f"\\n\\nInstruction: {instruction}"

    messages = [
        {"role": "system", "content": "You are a Tenacious Intelligence Corporation sales agent. Write policy-compliant outreach emails."},
        {"role": "user",   "content": user_content},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

if Path(HELD_OUT_PATH).exists():
    held_tasks = [json.loads(l) for l in open(HELD_OUT_PATH)][:3]
    for task in held_tasks:
        print(f"\\n{'='*60}")
        print(f"Task: {task['task_id']} | {task['failure_family']} | {task['difficulty']}")
        print(f"Instruction: {task['input']['outreach_instruction'][:100]}")
        print(f"{'─'*60}")
        print(run_inference(task))
else:
    print(f"Upload {HELD_OUT_PATH} to Colab to run sanity check")
    print("(Skip this cell if you just want to go straight to Cell 7 for full scoring)")
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Full held-out inference for ablation (saves trained_outputs.jsonl)
# ─────────────────────────────────────────────────────────────────────────────
CELL_7 = """
# Run all 41 held-out tasks and save outputs to trained_outputs.jsonl.
# This file is consumed by: ablations/run_ablations.py --condition trained
# Expected runtime: ~5-8 min on T4 (41 tasks x ~300 tokens)

HELD_OUT_PATH    = "/content/held_out_tasks.jsonl"
TRAINED_OUT_PATH = "/content/trained_outputs.jsonl"

if not Path(HELD_OUT_PATH).exists():
    print(f"ERROR: Upload {HELD_OUT_PATH} to Colab first")
else:
    held_tasks = [json.loads(l) for l in open(HELD_OUT_PATH) if l.strip()]
    print(f"Running inference on {len(held_tasks)} held-out tasks...")

    FastLanguageModel.for_inference(model)
    results = []

    for i, task in enumerate(held_tasks):
        output = run_inference(task)   # run_inference defined in Cell 6
        results.append({"task_id": task["task_id"], "output": output})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(held_tasks)} done")

    with open(TRAINED_OUT_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\\n")

    print(f"\\nSaved {len(results)} outputs → {TRAINED_OUT_PATH}")
    print("Download this file and place it at: ablations/trained_outputs.jsonl")
    print("Then run: python3 ablations/run_ablations.py --condition trained")
"""


if __name__ == "__main__":
    print("Tenacious-Bench SimPO Training Notebook")
    print("=" * 60)
    print("Copy each CELL_N string into a Colab cell and run in order.")
    print()
    print("Files to upload to Colab before running:")
    print("  - training_data/preference_pairs.jsonl  (required for Cell 3)")
    print("  - tenacious_bench_v0.1/held_out/tasks.jsonl  (required for Cells 6+7)")
    print()
    print("Environment variables to set in Colab (Secrets or os.environ):")
    print("  HF_TOKEN        your HuggingFace write token")
    print("  HF_MODEL_REPO   mistire37/tenacious-bench-lora-adapter")
    print()
    print("Run order: Cell 0 → restart runtime → Cell 1 → 2 → 3 → 4 → 5 → 6 → 7")
    print()
    for i in range(8):
        cell = globals().get(f"CELL_{i}", "")
        if cell:
            print(f"{'─'*60}")
            print(f"CELL {i}:")
            print(cell.strip())
            print()
