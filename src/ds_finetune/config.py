from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class TrainingConfig:
    base_model_path: str
    experiment_name: str = "ds_r1_law_1.5B_exp4"
    dataset_name: str = "xxxcoder/law_finetune"
    dataset_subset: str = "default"
    dataset_split: str = "train"
    max_seq_length: int = 1024
    dtype: Optional[str] = None
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    max_steps: int = -1
    warmup_steps: int = 5
    eval_split_ratio: float = 0.01
    random_seed: int = 3407
    output_dir: Path = Path("outputs")
    enable_gradient_checkpointing: bool = True
    dataset_num_proc: int = 1
    max_train_samples: Optional[int] = 3000
    save_merged_model: bool = False

    def lora_target_modules(self) -> list[str]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    @property
    def finetuned_model_dir(self) -> Path:
        return Path(f"{self.experiment_name}_base")

    @property
    def merged_model_dir(self) -> Path:
        return Path(f"{self.experiment_name}_merged")

    @property
    def eval_result_dir(self) -> Path:
        return Path(f"{self.experiment_name}_eval_result")


@dataclass(slots=True)
class EvaluationConfig:
    text2vec_model_path: str
    eval_sample_num: int = 1000
    eval_max_len: int = 512


@dataclass(slots=True)
class ProjectConfig:
    training: TrainingConfig
    evaluation: EvaluationConfig
    cache_dir: Optional[Path] = None

    def ensure_directories(self) -> None:
        self.training.output_dir.mkdir(parents=True, exist_ok=True)
        self.training.finetuned_model_dir.mkdir(parents=True, exist_ok=True)
        self.training.merged_model_dir.mkdir(parents=True, exist_ok=True)
        self.training.eval_result_dir.mkdir(parents=True, exist_ok=True)
