from __future__ import annotations

from pathlib import Path

from trl import SFTConfig, SFTTrainer

from .config import ProjectConfig
from .data import build_training_dataset
from .modeling import ensure_bfloat16, load_base_model, prepare_lora_model


def run_training(config: ProjectConfig):
    training_cfg = config.training

    training_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_base_model(training_cfg)
    train_dataset, val_dataset = build_training_dataset(
        dataset_name=training_cfg.dataset_name,
        dataset_subset=training_cfg.dataset_subset,
        dataset_split=training_cfg.dataset_split,
        tokenizer=tokenizer,
        max_seq_length=training_cfg.max_seq_length,
        eval_ratio=training_cfg.eval_split_ratio,
        random_seed=training_cfg.random_seed,
        cache_dir=str(config.cache_dir) if config.cache_dir else None,
        max_samples=training_cfg.max_train_samples,
    )

    peft_model = prepare_lora_model(model, training_cfg)

    fp16, bf16 = ensure_bfloat16()

    training_kwargs = {
        "per_device_train_batch_size": training_cfg.batch_size,
        "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,
        "warmup_steps": training_cfg.warmup_steps,
        "max_steps": training_cfg.max_steps,
        "num_train_epochs": training_cfg.num_train_epochs,
        "learning_rate": training_cfg.learning_rate,
        "fp16": fp16,
        "bf16": bf16,
        "logging_steps": 10,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "linear",
        "seed": training_cfg.random_seed,
        "output_dir": str(training_cfg.output_dir),
        "eval_steps": 50,
        "save_steps": 200,
        "save_total_limit": 2,
        "dataset_num_proc": training_cfg.dataset_num_proc,
        "gradient_checkpointing": training_cfg.enable_gradient_checkpointing,
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }

    signature_params = SFTConfig.__init__.__code__.co_varnames  # type: ignore[attr-defined]
    if "evaluation_strategy" in signature_params:
        training_kwargs["evaluation_strategy"] = "steps"
    else:
        training_kwargs["eval_strategy"] = "steps"

    training_args = SFTConfig(**training_kwargs)

    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=training_cfg.max_seq_length,
        args=training_args,
    )

    train_result = trainer.train()

    peft_model.save_pretrained(training_cfg.finetuned_model_dir)
    tokenizer.save_pretrained(training_cfg.finetuned_model_dir)

    if training_cfg.save_merged_model:
        peft_model.save_pretrained_merged(
            training_cfg.merged_model_dir,
            tokenizer,
            save_method="merged_16bit",
        )

    return train_result
