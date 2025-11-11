from __future__ import annotations

from pathlib import Path  # 导入 Path 对象，用于处理文件路径
from typing import Optional  # 导入 Optional 类型提示

import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
from sentence_transformers import SentenceTransformer  # 导入 SentenceTransformer，用于计算句子嵌入
from unsloth import FastLanguageModel  # 从 unsloth 库导入核心模型类

from .config import ProjectConfig  # 从本地模块导入项目总配置
from .data import build_evaluation_dataset  # 从本地模块导入评估数据集构建函数
from .modeling import generate_responses  # 从本地模块导入文本生成函数


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个 NumPy 向量之间的余弦相似度。

    余弦相似度是衡量两个向量方向上差异的指标，值域为 [-1, 1]。
    值越接近 1，表示两个向量方向越相似。

    Args:
        a (np.ndarray): 第一个向量。
        b (np.ndarray): 第二个向量。

    Returns:
        float: 两个向量的余弦相似度。
    """
    # 计算分母：两个向量的 L2 范数（即长度）的乘积
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    # 为避免除以零的错误，如果分母接近于零，则直接返回 0.0
    if denominator < 1e-6:
        return 0.0
    # 计算分子：两个向量的点积，然后除以分母
    return float(np.dot(a, b) / denominator)


def _extract_answer(text: str) -> str:
    """
    从模型生成的完整文本中提取出 "### 回复:" 之后的内容。

    这是因为模型生成的文本包含了完整的提示模板，而我们只关心回复部分。

    Args:
        text (str): 模型生成的完整文本。

    Returns:
        str: 提取出的回复内容。如果找不到标记，则返回原始文本。
    """
    marker = "### 回复:"  # 定义回复内容的开始标记
    if marker in text:
        # 如果找到标记，则按标记分割字符串，取第二部分并去除首尾空白
        return text.split(marker, 1)[1].strip()
    # 如果未找到标记，则直接返回去除首尾空白的原始文本
    return text.strip()


def _load_text2vec(model_path: str) -> SentenceTransformer:
    """
    加载一个 SentenceTransformer 模型，用于将文本转换为向量嵌入。

    Args:
        model_path (str): 模型的路径或在 Hugging Face Hub 上的名称。

    Returns:
        SentenceTransformer: 加载好的模型对象。
    """
    return SentenceTransformer(model_path)


def _ensure_dir(path: Path) -> None:
    """
    确保指定的目录存在，如果不存在则创建它。

    Args:
        path (Path): 要检查或创建的目录路径。
    """
    path.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    config: ProjectConfig,
    *,
    model_path: Optional[str] = None,
    eval_sample_num: Optional[int] = None,
    eval_max_len: Optional[int] = None,
    result_suffix: str = "original",
) -> pd.DataFrame:
    """
    对指定模型进行评估，并计算生成回复与标准答案之间的余弦相似度。

    Args:
        config (ProjectConfig): 项目配置对象。
        model_path (Optional[str]): 要评估的模型的路径。如果为 None，则使用配置中的基础模型路径。
        eval_sample_num (Optional[int]): 用于评估的样本数量。如果为 None，则使用配置中的数量。
        eval_max_len (Optional[int]): 生成文本的最大长度。如果为 None，则使用配置中的长度。
        result_suffix (str): 评估结果文件名后缀，用于区分不同评估运行。

    Returns:
        pd.DataFrame: 包含评估结果的 DataFrame，包括问题、标准答案、模型预测和余弦相似度。
    """
    training_cfg = config.training
    evaluation_cfg = config.evaluation

    # 1. 构建评估数据集
    eval_dataset = build_evaluation_dataset(
        dataset_name=training_cfg.dataset_name,
        dataset_subset=training_cfg.dataset_subset,
        dataset_split="train",  # 注意：这里使用了 'train' 切分，因为原始数据集没有提供单独的评估集
        cache_dir=str(config.cache_dir) if config.cache_dir else None,
        batch_size=training_cfg.batch_size,
    )

    # 2. 加载用于计算文本嵌入的 SentenceTransformer 模型
    text2vec_model = _load_text2vec(evaluation_cfg.text2vec_model_path)

    # 3. 确定要评估的模型路径
    model_to_use = model_path or training_cfg.base_model_path

    # 4. 加载要评估的语言模型和分词器
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_to_use,
        max_seq_length=training_cfg.max_seq_length,
        dtype=training_cfg.dtype,
        load_in_4bit=training_cfg.load_in_4bit,
        load_in_8bit=training_cfg.load_in_8bit,
        full_finetuning=training_cfg.full_finetuning,
    )

    rows: list[dict[str, object]] = []  # 用于存储每条评估结果的列表
    # 确定评估样本数量和生成长度的上限
    sample_limit = eval_sample_num or evaluation_cfg.eval_sample_num
    max_length = eval_max_len or evaluation_cfg.eval_max_len

    # 5. 遍历评估数据集并进行推理和评估
    for idx, example in enumerate(eval_dataset):
        if idx >= sample_limit:
            break  # 达到样本数量上限后停止

        # 从数据集中获取一批提示、问题和答案
        prompts = example["text"]
        questions = example["question"]
        answers = example["answer"]

        # 批量生成回复
        responses = generate_responses(model, tokenizer, prompts, max_length)

        # 逐条处理批次内的结果
        for question, answer, response in zip(questions, answers, responses):
            # 从生成的完整文本中提取回复
            pred = _extract_answer(response)
            # 计算标准答案和模型预测的文本嵌入
            embeddings = text2vec_model.encode([answer, pred])
            # 计算余弦相似度
            cos_sim = _cos_sim(embeddings[0], embeddings[1])
            # 将结果存入列表
            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "prediction": pred,
                    "cos_sim": cos_sim,
                }
            )

    # 6. 将结果转换为 Pandas DataFrame 并保存
    df = pd.DataFrame(rows)
    _ensure_dir(training_cfg.eval_result_dir)  # 确保输出目录存在
    output_path = training_cfg.eval_result_dir / f"eval_result_{result_suffix}.parquet"
    df.to_parquet(output_path)  # 保存为 Parquet 格式，高效且节省空间
    return df
