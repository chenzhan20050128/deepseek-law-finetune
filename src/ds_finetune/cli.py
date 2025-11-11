from __future__ import annotations

import argparse  # 导入 argparse 模块，用于解析命令行参数
import json  # 导入 json 模块，用于格式化输出
from pathlib import Path  # 导入 Path 对象，用于处理文件路径
from typing import Iterable, Optional  # 导入类型提示工具

from unsloth import FastLanguageModel  # 从 unsloth 库导入核心模型类

# 从本地模块导入默认模型 ID、资源确保函数、配置类和核心功能函数
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


# --- 参数定义辅助函数 ---
# 这些函数将相关的命令行参数组织在一起，使得主解析器构建更清晰。

def _add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    """为解析器添加与数据集相关的通用参数."""
    parser.add_argument("--dataset-name", default="xxxcoder/law_finetune", help="Hugging Face Hub 上的数据集名称")
    parser.add_argument("--dataset-subset", default="default", help="数据集的子集")
    parser.add_argument("--dataset-split", default="train", help="使用的数据集切分")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face 的缓存目录")


def _add_model_loading_args(parser: argparse.ArgumentParser) -> None:
    """为解析器添加与模型加载相关的参数."""
    parser.add_argument("--base-model-id", default=DEFAULT_BASE_MODEL_ID, help="基础模型的 Hugging Face Hub ID")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="模型最大序列长度")
    parser.add_argument("--dtype", default=None, help="加载模型时的数据类型 (e.g., float16, bfloat16)")
    # 使用 action="store_true" 创建布尔标志，默认为 False
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="以 4-bit 精度加载模型")
    # dest="load_in_4bit" 和 action="store_false" 用于创建可以关闭默认行为的标志
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false", help="不以 4-bit 精度加载模型")
    parser.add_argument("--load-in-8bit", action="store_true", help="以 8-bit 精度加载模型")
    parser.add_argument("--full-finetuning", action="store_true", help="执行全参数微调而非 LoRA")


def _add_lora_args(parser: argparse.ArgumentParser) -> None:
    """为解析器添加与 LoRA 相关的参数."""
    parser.add_argument("--lora-rank", type=int, default=16, help="LoRA 的秩 (rank)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA 的 alpha 值")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA 层的 dropout 概率")


def _add_training_hyperparameters(parser: argparse.ArgumentParser) -> None:
    """为解析器添加训练超参数."""
    parser.add_argument("--batch-size", type=int, default=1, help="每个设备的训练批量大小")
    parser.add_argument("--dataset-num-proc", type=int, default=1, help="数据预处理的进程数")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=3000,
        help="最大训练样本数 (负数表示使用全部)",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="学习率")
    parser.add_argument("--num-train-epochs", type=int, default=1, help="训练轮次")
    parser.add_argument("--max-steps", type=int, default=-1, help="最大训练步数 (覆盖 epochs)")
    parser.add_argument("--warmup-steps", type=int, default=5, help="学习率预热步数")
    parser.add_argument("--eval-split-ratio", type=float, default=0.01, help="验证集划分比例")
    parser.add_argument("--random-seed", type=int, default=3407, help="随机种子")
    parser.add_argument("--experiment-name", default="ds_r1_law_1.5B_exp4", help="实验名称")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="输出目录")
    parser.add_argument("--save-merged-model", action="store_true", help="训练后保存合并后的模型")


def _add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    """为解析器添加与评估相关的参数."""
    parser.add_argument("--text2vec-model-id", default=DEFAULT_TEXT2VEC_MODEL_ID, help="SentenceTransformer 模型的 ID")
    parser.add_argument("--text2vec-model-path", help="SentenceTransformer 模型的本地路径 (优先)")
    parser.add_argument("--eval-sample-num", type=int, default=1000, help="评估样本数")
    parser.add_argument("--eval-max-len", type=int, default=512, help="评估时生成的最大长度")


def _build_project_config(args: argparse.Namespace, base_model_path: str, text2vec_model_path: str) -> ProjectConfig:
    """根据解析的命令行参数构建项目总配置对象."""
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


# --- 命令处理函数 ---
# 每个函数对应一个子命令 (train, evaluate, predict)。

def _train_command(args: argparse.Namespace) -> None:
    """处理 'train' 子命令."""
    # 确保基础模型和 text2vec 模型存在，如果不存在则下载
    base_model_path = ensure_base_model(
        model_id=args.base_model_id,
        preferred_path=args.base_model_path,
    )

    text2vec_model_path = ensure_text2vec_model(
        model_id=args.text2vec_model_id,
        preferred_path=args.text2vec_model_path,
    )

    # 构建配置并开始训练
    config = _build_project_config(args, str(base_model_path), str(text2vec_model_path))
    train_result = run_training(config)
    print("Training finished:")
    print(json.dumps(train_result.metrics, indent=2, default=str))

    # (可选) 评估原始基础模型
    if args.eval_original:
        print("Evaluating original model...")
        original_df = evaluate_model(config, model_path=str(base_model_path), result_suffix="original")
        print(f"Original model average cos sim: {original_df['cos_sim'].mean():.4f}")

    # (可选) 评估微调后的模型
    if args.eval_trained:
        print("Evaluating trained model...")
        # 必须先保存合并后的模型才能评估
        if not config.training.save_merged_model:
            print("Warning: --eval-trained requires --save-merged-model to be set. Skipping evaluation.")
        else:
            trained_model_path = str(config.training.merged_model_dir)
            trained_df = evaluate_model(config, model_path=trained_model_path, result_suffix="trained")
            print(f"Trained model average cos sim: {trained_df['cos_sim'].mean():.4f}")


def _evaluate_command(args: argparse.Namespace) -> None:
    """处理 'evaluate' 子命令."""
    base_model_path = ensure_base_model(
        model_id=args.base_model_id,
        preferred_path=args.model_path,  # 注意：这里用 --model-path 指定要评估的模型
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
    print("Evaluation results:")
    print(df.describe())


def _predict_command(args: argparse.Namespace) -> None:
    """处理 'predict' 子命令."""
    # 将所有问题格式化为推理提示
    prompts = [format_inference_prompt(question) for question in args.question]

    model_path = ensure_base_model(
        model_id=args.base_model_id,
        preferred_path=args.model_path,
    )

    # 加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path),
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        full_finetuning=args.full_finetuning,
    )

    # 生成回复
    responses = generate_responses(
        model,
        tokenizer,
        prompts,
        max_length=args.eval_max_len,
    )

    # 打印结果
    for question, response in zip(args.question, responses):
        print("\n" + "="*20)
        print("=== Question ===")
        print(question)
        print("\n=== Response ===")
        print(response)
        print("="*20)


def build_parser() -> argparse.ArgumentParser:
    """构建总的命令行参数解析器，包含所有子命令."""
    parser = argparse.ArgumentParser(description="DeepSeek-Law Finetuning Pipeline")
    # 创建子解析器，用于处理不同的命令 (train, evaluate, predict)
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- 'train' 命令的解析器 ---
    train_parser = subparsers.add_parser("train", help="Run LoRA finetuning")
    train_parser.add_argument("--base-model-path", help="本地基础模型路径 (优先于 --base-model-id)")
    _add_common_dataset_args(train_parser)
    _add_model_loading_args(train_parser)
    _add_lora_args(train_parser)
    _add_training_hyperparameters(train_parser)
    _add_evaluation_args(train_parser)
    train_parser.add_argument("--eval-original", action="store_true", help="训练后评估原始模型")
    train_parser.add_argument("--eval-trained", action="store_true", help="训练后评估微调过的模型 (需 --save-merged-model)")
    train_parser.set_defaults(func=_train_command)  # 将此子命令与处理函数关联

    # --- 'evaluate' 命令的解析器 ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model on the law QA dataset")
    eval_parser.add_argument("--model-path", help="要评估的模型的本地路径 (优先)")
    _add_common_dataset_args(eval_parser)
    _add_model_loading_args(eval_parser)
    _add_evaluation_args(eval_parser)
    eval_parser.add_argument("--result-suffix", default="custom", help="评估结果文件的后缀")
    eval_parser.set_defaults(func=_evaluate_command)

    # --- 'predict' 命令的解析器 ---
    predict_parser = subparsers.add_parser("predict", help="Generate answers for one or more questions")
    predict_parser.add_argument("--model-path", help="用于推理的模型的本地路径 (优先)")
    # action="append" 允许用户多次使用 --question 参数来输入多个问题
    predict_parser.add_argument("--question", action="append", required=True,
                                help="输入的问题 (可重复使用此参数)")
    _add_model_loading_args(predict_parser)
    predict_parser.add_argument("--eval-max-len", type=int, default=512, help="生成答案的最大长度")
    predict_parser.set_defaults(func=_predict_command)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    """程序主入口."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    # 调用与所选子命令关联的函数
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    # 当脚本作为主程序执行时，调用 main 函数
    main()
