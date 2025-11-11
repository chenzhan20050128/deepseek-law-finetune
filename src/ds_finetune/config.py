from __future__ import annotations

from dataclasses import dataclass, field  # 导入 dataclass，用于快速创建数据类
from pathlib import Path  # 导入 Path 对象，用于以面向对象的方式处理文件路径
from typing import Optional  # 导入 Optional 类型提示

# 使用 @dataclass 装饰器可以自动为类添加 __init__, __repr__, __eq__ 等方法。
# slots=True 是一个优化选项，它会使用 __slots__ 而不是 __dict__ 来存储实例属性，
# 这样可以减少内存占用并加快属性访问速度，适用于属性固定的数据类。
@dataclass(slots=True)
class TrainingConfig:
    """
    存储所有与模型训练相关的配置参数。
    """
    # --- 模型与数据路径 ---
    base_model_path: str  # 基础模型的路径，可以是本地路径或 Hugging Face Hub 上的模型名称
    experiment_name: str = "ds_r1_law_1.5B_exp4"  # 实验名称，用于命名输出目录
    dataset_name: str = "xxxcoder/law_finetune"  # 训练数据集在 Hugging Face Hub 上的名称
    dataset_subset: str = "default"  # 数据集的子集名称
    dataset_split: str = "train"  # 使用的数据集切分，例如 'train' 或 'validation'

    # --- 模型配置 ---
    max_seq_length: int = 1024  # 模型能处理的最大序列长度
    dtype: Optional[str] = None  # 模型加载时的数据类型，例如 "float16", "bfloat16"。None 表示自动选择
    load_in_4bit: bool = True  # 是否以 4-bit 精度加载模型（QLoRA），极大节省显存
    load_in_8bit: bool = False  # 是否以 8-bit 精度加载模型
    full_finetuning: bool = False  # 是否进行全参数微调。如果为 False，则使用 PEFT（如 LoRA）

    # --- LoRA (参数高效微调) 配置 ---
    lora_rank: int = 16  # LoRA 的秩 (r)，决定了可训练参数的数量，是关键超参数
    lora_alpha: int = 16  # LoRA 的缩放因子 (alpha)，通常设置为与 r 相等或两倍
    lora_dropout: float = 0.0  # 在 LoRA 层上应用的 dropout 概率，用于防止过拟合

    # --- 训练超参数 ---
    batch_size: int = 1  # 每个设备（GPU）的训练批量大小
    gradient_accumulation_steps: int = 16  # 梯度累积步数。有效批量大小 = batch_size * num_gpus * gradient_accumulation_steps
    learning_rate: float = 2e-5  # 学习率
    num_train_epochs: int = 1  # 训练的总轮次
    max_steps: int = -1  # 最大训练步数。如果大于 0，则会覆盖 num_train_epochs
    warmup_steps: int = 5  # 学习率预热的步数
    eval_split_ratio: float = 0.01  # 从训练数据中划分出用于验证的比例
    random_seed: int = 3407  # 随机种子，用于保证实验的可复现性

    # --- 目录与性能配置 ---
    output_dir: Path = Path("outputs")  # 保存检查点、日志等训练产物的根目录
    enable_gradient_checkpointing: bool = True  # 是否启用梯度检查点，以时间换空间，显著减少显存占用
    dataset_num_proc: int = 1  # 数据预处理时使用的进程数
    max_train_samples: Optional[int] = 3000  # 使用的最大训练样本数。如果为 None，则使用全部样本
    save_merged_model: bool = False  # 训练结束后是否将 LoRA 权重与基础模型合并并保存

    def lora_target_modules(self) -> list[str]:
        """
        返回要应用 LoRA 的模型模块列表。
        这通常是 Transformer 模型中的注意力层和前馈网络层。
        Unsloth 框架能够很好地自动识别这些模块，但这里显式指定可以提供更精细的控制。
        """
        return [
            "q_proj",  # 查询（Query）投影
            "k_proj",  # 键（Key）投影
            "v_proj",  # 值（Value）投影
            "o_proj",  # 输出（Output）投影
            "gate_proj",  # 门控投影（在前馈网络中）
            "up_proj",  # 上采样投影（在前馈网络中）
            "down_proj",  # 下采样投影（在前馈网络中）
        ]

    @property
    def finetuned_model_dir(self) -> Path:
        """动态生成用于保存微调后 LoRA 适配器（基础模型）的目录路径."""
        return Path(f"{self.experiment_name}_base")

    @property
    def merged_model_dir(self) -> Path:
        """动态生成用于保存合并后模型的目录路径."""
        return Path(f"{self.experiment_name}_merged")

    @property
    def eval_result_dir(self) -> Path:
        """动态生成用于保存评估结果的目录路径."""
        return Path(f"{self.experiment_name}_eval_result")


@dataclass(slots=True)
class EvaluationConfig:
    """
    存储所有与模型评估相关的配置参数。
    """
    text2vec_model_path: str  # 用于计算文本嵌入的 SentenceTransformer 模型路径
    eval_sample_num: int = 1000  # 评估时使用的样本数量
    eval_max_len: int = 512  # 评估时模型生成文本的最大长度


@dataclass(slots=True)
class ProjectConfig:
    """
    项目总配置，聚合了训练和评估的配置。
    """
    training: TrainingConfig  # 训练配置对象
    evaluation: EvaluationConfig  # 评估配置对象
    cache_dir: Optional[Path] = None  # Hugging Face datasets 和 models 的缓存目录

    def ensure_directories(self) -> None:
        """
        确保所有在配置中定义的输出目录都存在。
        """
        self.training.output_dir.mkdir(parents=True, exist_ok=True)
        self.training.finetuned_model_dir.mkdir(parents=True, exist_ok=True)
        if self.training.save_merged_model:
            self.training.merged_model_dir.mkdir(parents=True, exist_ok=True)
        self.training.eval_result_dir.mkdir(parents=True, exist_ok=True)
