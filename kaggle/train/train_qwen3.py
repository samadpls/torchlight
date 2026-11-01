"""Fine-tune Qwen3-8B on torchlight-threads on a Kaggle GPU and publish the results.

Runs as a Kaggle script kernel (see kaggle/README.md). Reads the Hugging Face token from the
Kaggle secret HF_API, trains a LoRA adapter on the 4-bit base with Unsloth, then pushes the
adapter, the merged 16-bit model and GGUF quants to the Hub.
"""

import json
import os
import subprocess
import sys

SMOKE = False
BASE_MODEL = "unsloth/Qwen3-8B"
DATASET = "samadpls/torchlight-threads"
HUB_REPO = "samadpls/torchlight-qwen3-8b"
MAX_LENGTH = 4096
OUT = "/kaggle/working"
SYSTEM_PROMPT = (
    "You are a careful PyTorch helper. Give specific, correct guidance with code where it helps. "
    "If a question lacks the details needed to answer it, ask for them, for example a minimal "
    "executable code snippet or the full error message, instead of guessing."
)

# Unsloth trains on one GPU; this must be set before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from kaggle_secrets import UserSecretsClient  # noqa: E402

# Fail now, not after hours of training, if the secret isn't available. Kaggle hands secrets
# only to runs started from the notebook editor, not to runs pushed through the API, so the
# smoke run (which publishes nothing) skips it.
HF_TOKEN = None if SMOKE else UserSecretsClient().get_secret("HF_API")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)

from unsloth import FastLanguageModel  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments  # noqa: E402

model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_MODEL, max_seq_length=MAX_LENGTH, load_in_4bit=True
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=0,
)


def tokenize(row):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + row["messages"]
    # With thinking off, Qwen3's prompt ends in an empty <think></think> block; the answer
    # follows it, and only the answer and its end token are trained.
    prompt = tokenizer.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    answer_ids = tokenizer(messages[-1]["content"] + "<|im_end|>\n", add_special_tokens=False).input_ids
    ids = prompt_ids + answer_ids
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": [-100] * len(prompt_ids) + answer_ids}


sft = load_dataset(DATASET, "sft")
columns = sft["train"].column_names
train = sft["train"].map(tokenize, remove_columns=columns).filter(lambda r: len(r["input_ids"]) <= MAX_LENGTH)
valid = sft["validation"].map(tokenize, remove_columns=columns).filter(lambda r: len(r["input_ids"]) <= MAX_LENGTH)
if SMOKE:
    train, valid = train.select(range(40)), valid.select(range(10))
print(f"train: {len(train)} chats, validation: {len(valid)} chats", flush=True)

trainer = Trainer(
    model=model,
    train_dataset=train,
    eval_dataset=valid,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    args=TrainingArguments(
        output_dir=f"{OUT}/checkpoints",
        num_train_epochs=1,
        max_steps=5 if SMOKE else -1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        # transformers 5 reads a float below 1 as a share of the steps.
        warmup_steps=0.05,
        weight_decay=0.01,
        optim="adamw_8bit",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        train_sampling_strategy="group_by_length",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=5 if SMOKE else 200,
        save_strategy="steps",
        save_steps=5 if SMOKE else 200,
        save_total_limit=1,
        seed=0,
        report_to="none",
    ),
)
base_eval = trainer.evaluate()
print("validation loss before training:", base_eval["eval_loss"], flush=True)
trainer.train()
log = [{"step": 0, "eval_loss": base_eval["eval_loss"]}] + trainer.state.log_history
with open(f"{OUT}/log_history.json", "w") as f:
    json.dump(log, f, indent=1)

FastLanguageModel.for_inference(model)
question = "How do I check which device a tensor is on?"
chat = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
text = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False, enable_thinking=False)
inputs = tokenizer(text, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print("sample answer:", tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True), flush=True)

model.save_pretrained(f"{OUT}/adapter")
tokenizer.save_pretrained(f"{OUT}/adapter")
if SMOKE:
    print("smoke run done", flush=True)
    sys.exit(0)

model.push_to_hub(f"{HUB_REPO}-lora", token=HF_TOKEN)
tokenizer.push_to_hub(f"{HUB_REPO}-lora", token=HF_TOKEN)
print("pushed the adapter", flush=True)
model.push_to_hub_merged(HUB_REPO, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
print("pushed the merged model", flush=True)
try:
    model.push_to_hub_gguf(f"{HUB_REPO}-GGUF", tokenizer, quantization_method=["q4_k_m", "q8_0"], token=HF_TOKEN)
    print("pushed the GGUF files", flush=True)
except Exception as error:  # GGUF is a bonus; the run has already published the model.
    print("GGUF export failed:", repr(error), flush=True)
print("done", flush=True)
