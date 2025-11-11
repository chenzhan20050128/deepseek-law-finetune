from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from unsloth import FastLanguageModel

from .assets import (
    DEFAULT_BASE_MODEL_ID,
    DEFAULT_TEXT2VEC_MODEL_ID,
    ensure_base_model,
    ensure_text2vec_model,
)
from .config import EvaluationConfig, ProjectConfig, TrainingConfig
from .evaluate import evaluate_model
from .modeling import generate_responses
from .prompts import format_inference_prompt
from .train import run_training


def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-name", default="xxxcoder/law_finetune")
    parser.add_argument("--dataset-subset", default="default")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--cache-dir", type=Path)


def _add_model_loading_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--full-finetuning", action="store_true")


def _add_lora_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)


def _add_training_hyperparameters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dataset-num-proc", type=int, default=1)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=3000,
        help="Limit the number of training samples (set to a negative value to use the full dataset)",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--eval-split-ratio", type=float, default=0.01)
    parser.add_argument("--random-seed", type=int, default=3407)
    parser.add_argument("--experiment-name", default="ds_r1_law_1.5B_exp4")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--save-merged-model", action="store_true")


def _add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text2vec-model-id", default=DEFAULT_TEXT2VEC_MODEL_ID)
    parser.add_argument("--text2vec-model-path")
    parser.add_argument("--eval-sample-num", type=int, default=1000)
    parser.add_argument("--eval-max-len", type=int, default=512)


def _build_project_config(args: argparse.Namespace, base_model_path: str, text2vec_model_path: str) -> ProjectConfig:
    training_cfg = TrainingConfig(
        base_model_path=base_model_path,
        experiment_name=args.experiment_name,
        dataset_name=args.dataset_name,
        dataset_subset=args.dataset_subset,
        dataset_split=args.dataset_split,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        dataset_num_proc=args.dataset_num_proc,
        max_train_samples=None if args.max_train_samples < 0 else args.max_train_samples,
    save_merged_model=args.save_merged_model,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        eval_split_ratio=args.eval_split_ratio,
        random_seed=args.random_seed,
        output_dir=args.output_dir,
    )

    evaluation_cfg = EvaluationConfig(
        text2vec_model_path=text2vec_model_path,
        eval_sample_num=args.eval_sample_num,
        eval_max_len=args.eval_max_len,
    )

    project_config = ProjectConfig(
        training=training_cfg,
        evaluation=evaluation_cfg,
        cache_dir=args.cache_dir,
    )
    return project_config


def _train_command(args: argparse.Namespace) -> None:
    base_model_path = ensure_base_model(
        model_id=args.base_model_id,
        preferred_path=args.base_model_path,
    )

    text2vec_model_path = ensure_text2vec_model(
        model_id=args.text2vec_model_id,
        preferred_path=args.text2vec_model_path,
    )

    config = _build_project_config(args, str(base_model_path), str(text2vec_model_path))
    train_result = run_training(config)
    print("Training finished:")
    print(json.dumps(train_result.metrics, indent=2, default=str))

    if args.eval_original:
        original_df = evaluate_model(config, model_path=str(base_model_path), result_suffix="original")
        print(f"Original model average cos sim: {original_df['cos_sim'].mean():.4f}")

    if args.eval_trained:
        trained_model_path = str(config.training.merged_model_dir)
        trained_df = evaluate_model(config, model_path=trained_model_path, result_suffix="trained")
        print(f"Trained model average cos sim: {trained_df['cos_sim'].mean():.4f}")


def _evaluate_command(args: argparse.Namespace) -> None:
    base_model_path = ensure_base_model(
        model_id=args.base_model_id,
        preferred_path=args.model_path,
    )

    text2vec_model_path = ensure_text2vec_model(
        model_id=args.text2vec_model_id,
        preferred_path=args.text2vec_model_path,
    )

    config = _build_project_config(args, str(base_model_path), str(text2vec_model_path))
    df = evaluate_model(
        config,
        model_path=str(base_model_path),
        eval_sample_num=args.eval_sample_num,
        eval_max_len=args.eval_max_len,
        result_suffix=args.result_suffix,
    )
    print(df.describe())


def _predict_command(args: argparse.Namespace) -> None:
    prompts = [format_inference_prompt(question) for question in args.question]

    model_path = ensure_base_model(
        model_id=args.base_model_id,
        preferred_path=args.model_path,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
    )

    responses = generate_responses(
        model,
        tokenizer,
        prompts,
        max_length=args.eval_max_len,
    )

    for question, response in zip(args.question, responses):
        print("=== Question ===")
        print(question)
        print("=== Response ===")
        print(response)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek distilled model finetuning pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run LoRA finetuning")
    train_parser.add_argument("--base-model-path")
    _add_common_dataset_args(train_parser)
    _add_model_loading_args(train_parser)
    _add_lora_args(train_parser)
    _add_training_hyperparameters(train_parser)
    _add_evaluation_args(train_parser)
    train_parser.add_argument("--eval-original", action="store_true")
    train_parser.add_argument("--eval-trained", action="store_true")
    train_parser.set_defaults(func=_train_command)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model on the law QA dataset")
    eval_parser.add_argument("--model-path")
    _add_common_dataset_args(eval_parser)
    _add_model_loading_args(eval_parser)
    _add_evaluation_args(eval_parser)
    eval_parser.add_argument("--result-suffix", default="custom")
    eval_parser.set_defaults(func=_evaluate_command)

    predict_parser = subparsers.add_parser("predict", help="Generate answers for one or more questions")
    predict_parser.add_argument("--model-path")
    predict_parser.add_argument("--question", action="append", required=True,
                                help="Question prompt; repeat for multiple questions")
    _add_model_loading_args(predict_parser)
    predict_parser.add_argument("--eval-max-len", type=int, default=512)
    predict_parser.set_defaults(func=_predict_command)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
