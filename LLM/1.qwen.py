import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType


MODEL_NAME = "Qwen/Qwen3-0.6B"   # 跑不动就改成 "Qwen/Qwen3-0.6B"
TRAIN_FILE = "../data/train.jsonl"
VALID_FILE = "../data/valid.jsonl"

OUTPUT_DIR_LORA = "./qwen3_lora_output"
OUTPUT_DIR_FULL = "./qwen3_full_output"

MAX_LENGTH = 2048

# 通用训练超参数
NUM_TRAIN_EPOCHS = 2
LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 100
SAVE_TOTAL_LIMIT = 2

# LoRA 超参数
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_LEARNING_RATE = 1e-4
LORA_BATCH_SIZE = 1
LORA_EVAL_BATCH_SIZE = 1
LORA_GRAD_ACC_STEPS = 8

# 全参微调超参数
FULL_LEARNING_RATE = 2e-5
FULL_BATCH_SIZE = 1
FULL_EVAL_BATCH_SIZE = 1
FULL_GRAD_ACC_STEPS = 16


def build_text(example, tokenizer):
    """
    把 messages 转成 Qwen chat template 文本。
    """
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def tokenize(example, tokenizer, max_length):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen LoRA/全参微调脚本")

    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--train_file", type=str, default=TRAIN_FILE)
    parser.add_argument("--valid_file", type=str, default=VALID_FILE)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)

    parser.add_argument(
        "--finetune_mode",
        "--finetuen_mode",
        dest="finetune_mode",
        choices=["lora", "full"],
        default='lora',
        help="微调模式: lora 或 full（兼容 --finetuen_mode 拼写）",
    )

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_train_epochs", type=float, default=NUM_TRAIN_EPOCHS)
    parser.add_argument("--logging_steps", type=int, default=LOGGING_STEPS)
    parser.add_argument("--eval_steps", type=int, default=EVAL_STEPS)
    parser.add_argument("--save_steps", type=int, default=SAVE_STEPS)
    parser.add_argument("--save_total_limit", type=int, default=SAVE_TOTAL_LIMIT)

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)

    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--lora_alpha", type=int, default=LORA_ALPHA)
    parser.add_argument("--lora_dropout", type=float, default=LORA_DROPOUT)

    return parser.parse_args()


def main():
    args = parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    print(f"Fine-tune mode: {args.finetune_mode}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "validation": args.valid_file,
        },
    )

    dataset = dataset.map(lambda x: build_text(x, tokenizer))
    dataset = dataset.map(
        lambda x: tokenize(x, tokenizer, args.max_length),
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = model.to(device)

    # 训练时建议关闭 cache
    model.config.use_cache = False

    if args.finetune_mode == "lora":
        # Qwen/LLaMA 类模型常见 LoRA target modules
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",  # y = W x + LoRA(x)
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)  # 把 lora 的权值加进去
        model.print_trainable_parameters()
        output_dir = args.output_dir or OUTPUT_DIR_LORA
        learning_rate = args.learning_rate or LORA_LEARNING_RATE
        train_batch_size = args.per_device_train_batch_size or LORA_BATCH_SIZE
        eval_batch_size = args.per_device_eval_batch_size or LORA_EVAL_BATCH_SIZE
        grad_acc_steps = args.gradient_accumulation_steps or LORA_GRAD_ACC_STEPS
    elif args.finetune_mode == "full":
        # 全参微调更吃显存，启用梯度检查点降低内存占用
        model.gradient_checkpointing_enable()
        output_dir = args.output_dir or OUTPUT_DIR_FULL
        learning_rate = args.learning_rate or FULL_LEARNING_RATE
        train_batch_size = args.per_device_train_batch_size or FULL_BATCH_SIZE
        eval_batch_size = args.per_device_eval_batch_size or FULL_EVAL_BATCH_SIZE
        grad_acc_steps = args.gradient_accumulation_steps or FULL_GRAD_ACC_STEPS
    else:
        raise ValueError("finetune_mode 只能是 'lora' 或 'full'")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=learning_rate,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        fp16=False,   # Mac MPS 上不要强行开 fp16 trainer
        bf16=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()