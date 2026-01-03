from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass
class TrainConfig:
    model_name_or_path: str
    dataset_path: str
    output_dir: str


def load_train_config(path: Path) -> TrainConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainConfig(
        model_name_or_path=data["model_name_or_path"],
        dataset_path=data["dataset_path"],
        output_dir=data["output_dir"],
    )


def train_dpo(config: TrainConfig) -> None:
    try:
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import DPOTrainer
    except ImportError as exc:
        raise RuntimeError("DPO training requires optional dependencies: trl, peft, transformers") from exc

    dataset = load_dataset("json", data_files=config.dataset_path)["train"]
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)

    peft_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=1,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=10,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
