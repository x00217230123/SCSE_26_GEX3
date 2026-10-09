import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from trl import SFTTrainer, SFTConfig

MODEL_PATH = "models/domain_adapter"

model = AutoPeftModelForCausalLM.from_pretrained(
    MODEL_PATH,
    is_trainable=True,
    dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Ensure the tokenizer has a pad token assigned to avoid collation alignment issues
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

data = load_dataset(
    "json",
    data_files={
        "train": "data/sft_v3_train.jsonl",
        "validation": "data/sft_v3_validation.jsonl",
    }
)

# Keep prompt/completion separate so TRL constructs a real completion mask.
def format_example(example):
    return {
        "prompt": (
            "You are a university IT support assistant. "
            "Answer only the user's question using supported university IT information. "
            "If the requested fact is not available, say that it is not specified rather than inventing it. "
            "Keep the answer concise.\n\n"
            "User: " + example["prompt"] + "\n\nAssistant:"
        ),
        "completion": " " + example["completion"],
    }

data = data.map(format_example)

config = SFTConfig(
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    output_dir="models/specialized_adapter_v3",
    num_train_epochs=1, # 1 epoch is perfect for preventing memorization!
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=5e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    max_length=256,
    completion_only_loss=True, 
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    processing_class=tokenizer,
)

trainer.train()

model.save_pretrained("models/specialized_adapter_v3")
tokenizer.save_pretrained("models/specialized_adapter_v3")
