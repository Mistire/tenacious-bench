"""
Tenacious-Bench v0.1 — SimPO Training Notebook
================================================
Path B: Preference-tuned judge / critic on Qwen 3.5 0.8B via Unsloth

INSTRUCTIONS FOR COLAB:
1. Runtime → Change runtime type → T4 GPU
2. Run Cell 0 (install) — takes 6-10 min on first run (kernel compile)
3. Upload preference_pairs.jsonl to Colab or mount Drive
4. Run cells in order
5. Adapter is pushed to HuggingFace at the end

Cost: $0 on free Colab T4
Wall time: ~45-75 min for 3 epochs on 79 pairs
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 0 — Install (run once, ~6-10 min)
# ─────────────────────────────────────────────────────────────────────────────
CELL_0 = """
%%capture
!pip install unsloth
!pip install trl>=0.9.0 peft>=0.10.0 transformers>=4.40.0
!pip install datasets accelerate bitsandbytes
# Verify
import trl, peft, unsloth
print(f"trl={trl.__version__}  peft={peft.__version__}  unsloth={unsloth.__version__}")
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Imports and config
# ─────────────────────────────────────────────────────────────────────────────
CELL_1 = """
import json, os, random, torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SimPOConfig, SimPOTrainer
from transformers import TrainingArguments

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "unsloth/Qwen2.5-0.5B-Instruct"   # Qwen 3.5 0.8B via Unsloth
MAX_SEQ_LEN  = 1024
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
TARGET_MODS  = ["q_proj", "v_proj"]

EPOCHS       = 3
BATCH_SIZE   = 2          # T4 safe
GRAD_ACCUM   = 4          # effective batch = 8
LR           = 5e-5
BETA         = 2.0        # SimPO temperature
GAMMA        = 0.5        # SimPO margin

OUTPUT_DIR   = "/content/tenacious_simpo_adapter"
HF_REPO      = os.getenv("HF_MODEL_REPO", "your-username/tenacious-bench-lora-adapter")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

DATA_PATH    = "/content/preference_pairs.jsonl"  # upload this file to Colab

print("Config loaded. SEED =", SEED)
print("Model:", MODEL_NAME)
print("LoRA: r=%d α=%d dropout=%.2f" % (LORA_R, LORA_ALPHA, LORA_DROPOUT))
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Load model with Unsloth
# ─────────────────────────────────────────────────────────────────────────────
CELL_2 = """
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    dtype=None,          # auto: fp16 on T4, bf16 on A100/L4
    load_in_4bit=False,  # 16-bit LoRA per Unsloth Qwen 3.5 guide
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

print("Model loaded. Trainable params:")
model.print_trainable_parameters()
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Load and format dataset
# ─────────────────────────────────────────────────────────────────────────────
CELL_3 = """
def format_pair(row):
    \"\"\"
    SimPOTrainer expects: prompt, chosen, rejected
    Prompt is the system+user turn.
    Chosen/rejected are the assistant responses.
    \"\"\"
    # The prompt field already has <|system|>...<|user|>... format
    # We need to convert to the model's chat template
    prompt_text = row["prompt"]

    # Extract system and user parts
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
    messages_chosen = messages_prompt + [
        {"role": "assistant", "content": row["chosen"]}
    ]
    messages_rejected = messages_prompt + [
        {"role": "assistant", "content": row["rejected"]}
    ]

    prompt_str   = tokenizer.apply_chat_template(messages_prompt,   tokenize=False, add_generation_prompt=True)
    chosen_str   = tokenizer.apply_chat_template(messages_chosen,   tokenize=False)
    rejected_str = tokenizer.apply_chat_template(messages_rejected, tokenize=False)

    # SimPOTrainer wants just the assistant turn for chosen/rejected
    chosen_response   = chosen_str[len(prompt_str):]
    rejected_response = rejected_str[len(prompt_str):]

    return {
        "prompt":   prompt_str,
        "chosen":   chosen_response,
        "rejected": rejected_response,
        "task_id":  row["task_id"],
        "failure_family": row["failure_family"],
    }

# Load pairs (quality_score >= 3 only — already filtered in the file)
raw = [json.loads(l) for l in open(DATA_PATH) if l.strip()]
print(f"Loaded {len(raw)} preference pairs")

dataset = Dataset.from_list(raw)
dataset = dataset.map(format_pair, remove_columns=dataset.column_names)

# 90/10 train/val split
split = dataset.train_test_split(test_size=0.1, seed=SEED)
train_ds = split["train"]
val_ds   = split["test"]

print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
print("\\nSample prompt (first 200 chars):")
print(train_ds[0]["prompt"][:200])
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Training
# ─────────────────────────────────────────────────────────────────────────────
CELL_4 = """
training_args = SimPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    beta=BETA,
    gamma_beta_ratio=GAMMA / BETA,
    max_length=MAX_SEQ_LEN,
    max_prompt_length=768,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    seed=SEED,
    report_to="none",   # set to "wandb" if you want W&B logging
)

trainer = SimPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
)

print("Starting SimPO training...")
print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Grad accum: {GRAD_ACCUM}")
print(f"  Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
print(f"  LR: {LR} | Beta: {BETA} | Gamma: {GAMMA}")
print()

train_result = trainer.train()

print("\\nTraining complete!")
print(f"  Train loss: {train_result.training_loss:.4f}")
print(f"  Runtime: {train_result.metrics['train_runtime']:.0f}s")
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Save loss log and push adapter
# ─────────────────────────────────────────────────────────────────────────────
CELL_5 = """
import json
from pathlib import Path

# Save training log
log = {
    "seed": SEED,
    "model": MODEL_NAME,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "grad_accum": GRAD_ACCUM,
    "lr": LR,
    "beta": BETA,
    "gamma": GAMMA,
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
print(f"Training log saved → {log_path}")

# Save LoRA adapter locally
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved → {OUTPUT_DIR}")

# Push to HuggingFace (LoRA adapter only — NOT the merged model)
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)
    model.push_to_hub(HF_REPO, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO, token=HF_TOKEN)
    print(f"Adapter pushed → https://huggingface.co/{HF_REPO}")
else:
    print("HF_TOKEN not set — skipping push. Set os.environ['HF_TOKEN'] to push.")
"""

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Quick sanity check (inference on 3 held-out tasks)
# ─────────────────────────────────────────────────────────────────────────────
CELL_6 = """
# Quick inference test — run 3 held-out tasks through the trained model
# Upload held_out_tasks.jsonl to Colab alongside preference_pairs.jsonl

HELD_OUT_PATH = "/content/held_out_tasks.jsonl"

FastLanguageModel.for_inference(model)  # enable native 2x faster inference

def run_inference(task: dict, max_new_tokens: int = 300) -> str:
    brief = task["input"].get("hiring_signal_brief", {})
    instruction = task["input"].get("outreach_instruction", "")
    bench = task["input"].get("bench_context")

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
        output = run_inference(task)
        print(output)
else:
    print(f"Upload {HELD_OUT_PATH} to Colab to run inference test")
"""

if __name__ == "__main__":
    print("Tenacious-Bench SimPO Training Notebook")
    print("=" * 60)
    print("Copy each CELL_N string into a Colab cell and run in order.")
    print()
    print("Files to upload to Colab:")
    print("  - training_data/preference_pairs.jsonl")
    print("  - tenacious_bench_v0.1/held_out/tasks.jsonl  (for Cell 6)")
    print()
    print("Environment variables to set in Colab:")
    print("  - HF_TOKEN: your HuggingFace write token")
    print("  - HF_MODEL_REPO: your-username/tenacious-bench-lora-adapter")
    print()
    for i in range(7):
        cell = globals().get(f"CELL_{i}", "")
        if cell:
            print(f"{'─'*60}")
            print(f"CELL {i}:")
            print(cell.strip())
            print()
