"""High-level utilities to finetune DeepSeek distilled models on law QA data."""

from .config import ProjectConfig, TrainingConfig, EvaluationConfig
from .cli import main

__all__ = [
    "ProjectConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "main",
]
