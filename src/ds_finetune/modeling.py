from __future__ import annotations

from typing import Optional, Sequence

import torch
from unsloth import FastLanguageModel, is_bfloat16_supported

from .config import TrainingConfig


def load_base_model(config: TrainingConfig):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_path,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        full_finetuning=config.full_finetuning,
    )
    return model, tokenizer


def prepare_lora_model(model, config: TrainingConfig):
    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=config.lora_target_modules(),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth" if config.enable_gradient_checkpointing else False,
        random_state=config.random_seed,
        use_rslora=False,
        loftq_config=None,
    )
    return peft_model


def ensure_bfloat16() -> tuple[bool, bool]:
    bf16_support = is_bfloat16_supported()
    return not bf16_support, bf16_support


def generate_responses(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    device: Optional[str] = None,
) -> list[str]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        use_cache=True,
    )

    responses = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return responses
