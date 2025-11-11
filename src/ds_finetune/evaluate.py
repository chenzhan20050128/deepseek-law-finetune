from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel

from .config import ProjectConfig
from .data import build_evaluation_dataset
from .modeling import generate_responses


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator < 1e-6:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _extract_answer(text: str) -> str:
    marker = "### 回复:"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return text.strip()


def _load_text2vec(model_path: str) -> SentenceTransformer:
    return SentenceTransformer(model_path)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    config: ProjectConfig,
    *,
    model_path: Optional[str] = None,
    eval_sample_num: Optional[int] = None,
    eval_max_len: Optional[int] = None,
    result_suffix: str = "original",
) -> pd.DataFrame:
    training_cfg = config.training
    evaluation_cfg = config.evaluation

    eval_dataset = build_evaluation_dataset(
        dataset_name=training_cfg.dataset_name,
        dataset_subset=training_cfg.dataset_subset,
        dataset_split="train",  # dataset does not expose a separate eval split
        cache_dir=str(config.cache_dir) if config.cache_dir else None,
        batch_size=training_cfg.batch_size,
    )

    text2vec_model = _load_text2vec(evaluation_cfg.text2vec_model_path)

    model_to_use = model_path or training_cfg.base_model_path

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_to_use,
        max_seq_length=training_cfg.max_seq_length,
        dtype=training_cfg.dtype,
        load_in_4bit=training_cfg.load_in_4bit,
        load_in_8bit=training_cfg.load_in_8bit,
        full_finetuning=training_cfg.full_finetuning,
    )

    rows: list[dict[str, object]] = []
    sample_limit = eval_sample_num or evaluation_cfg.eval_sample_num
    max_length = eval_max_len or evaluation_cfg.eval_max_len

    for idx, example in enumerate(eval_dataset):
        if idx >= sample_limit:
            break
        prompts = example["text"]
        questions = example["question"]
        answers = example["answer"]

        responses = generate_responses(model, tokenizer, prompts, max_length)

        for question, answer, response in zip(questions, answers, responses):
            pred = _extract_answer(response)
            embeddings = text2vec_model.encode([answer, pred])
            cos_sim = _cos_sim(embeddings[0], embeddings[1])
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "prediction": pred,
                    "cos_sim": cos_sim,
                }
            )

    df = pd.DataFrame(rows)
    _ensure_dir(training_cfg.eval_result_dir)
    output_path = training_cfg.eval_result_dir / f"eval_result_{result_suffix}.parquet"
    df.to_parquet(output_path)
    return df
