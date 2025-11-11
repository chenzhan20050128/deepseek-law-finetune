from __future__ import annotations

from pathlib import Path  # 导入 Path 对象，用于处理文件路径

# 从 trl (Transformer Reinforcement Learning) 库导入 SFTConfig 和 SFTTrainer
# SFTTrainer 是一个专门用于监督式微调 (Supervised Fine-Tuning) 的高级训练器
# SFTConfig 用于配置 SFTTrainer 的所有训练参数
from trl import SFTConfig, SFTTrainer

from .config import ProjectConfig  # 从本地模块导入项目总配置
from .data import build_training_dataset  # 从本地模块导入数据构建函数
from .modeling import ensure_bfloat16, load_base_model, prepare_lora_model  # 从本地模块导入模型处理函数


def run_training(config: ProjectConfig):
    """
    执行完整的模型监督式微调流程。

    这个函数整合了数据加载、模型准备、训练器配置和执行训练的所有步骤。

    Args:
        config (ProjectConfig): 包含所有配置信息的项目配置对象。
    """
    # 提取训练相关的配置
    training_cfg = config.training

    # 创建输出目录，如果目录已存在则不报错
    training_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载基础模型和分词器
    # 使用 unsloth 优化过的方法加载，支持量化等功能
    model, tokenizer = load_base_model(training_cfg)
    
    # 2. 构建训练和验证数据集
    # 这个函数会处理数据下载、规范化、格式化、分词和划分
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

    # 3. 为模型准备 LoRA 适配器
    # 将 LoRA 层应用到基础模型上，使其变为一个 PeftModel
    peft_model = prepare_lora_model(model, training_cfg)

    # 4. 检查硬件对 bfloat16 的支持情况
    # fp16 和 bf16 是两种不同的半精度浮点数格式，用于加速训练和减少显存占用
    fp16, bf16 = ensure_bfloat16()

    # 5. 配置训练参数
    # 这里定义了所有传递给 SFTConfig 的参数
    training_kwargs = {
        "per_device_train_batch_size": training_cfg.batch_size,  # 每个 GPU 上的训练批量大小
        "gradient_accumulation_steps": training_cfg.gradient_accumulation_steps,  # 梯度累积步数，用于模拟更大的批量
        "warmup_steps": training_cfg.warmup_steps,  # 学习率预热步数
        "max_steps": training_cfg.max_steps,  # 训练总步数，如果设置了，会覆盖 num_train_epochs
        "num_train_epochs": training_cfg.num_train_epochs,  # 训练总轮次
        "learning_rate": training_cfg.learning_rate,  # 学习率
        "fp16": fp16,  # 是否启用 fp16 混合精度训练
        "bf16": bf16,  # 是否启用 bf16 混合精度训练
        "logging_steps": 10,  # 每隔多少步记录一次日志
        "optim": "adamw_8bit",  # 使用 8-bit 的 AdamW 优化器，可以节省显存
        "weight_decay": 0.01,  # 权重衰减
        "lr_scheduler_type": "linear",  # 学习率调度器类型
        "seed": training_cfg.random_seed,  # 随机种子
        "output_dir": str(training_cfg.output_dir),  # 模型检查点和日志的输出目录
        "eval_steps": 50,  # 每隔多少步进行一次评估
        "save_steps": 200,  # 每隔多少步保存一次模型检查点
        "save_total_limit": 2,  # 最多保存多少个检查点
        "dataset_num_proc": training_cfg.dataset_num_proc,  # 数据预处理时使用的进程数
        "gradient_checkpointing": training_cfg.enable_gradient_checkpointing,  # 是否启用梯度检查点以节省显存
        # Unsloth/TRL 特有参数，告知 SFTTrainer 数据集已经预处理和分词完毕，无需再次处理
        "dataset_kwargs": {"skip_prepare_dataset": True},
    }

    # 兼容不同版本的 TRL/Hugging Face Transformers
    # 新版本将 `eval_strategy` 重命名为 `evaluation_strategy`
    # 通过检查 SFTConfig 的构造函数签名来动态确定使用哪个参数名
    signature_params = SFTConfig.__init__.__code__.co_varnames  # type: ignore[attr-defined]
    if "evaluation_strategy" in signature_params:
        training_kwargs["evaluation_strategy"] = "steps"  # 新版参数名
    else:
        training_kwargs["eval_strategy"] = "steps"  # 旧版参数名

    # 使用上面定义的参数字典实例化 SFTConfig
    training_args = SFTConfig(**training_kwargs)

    # 6. 初始化 SFTTrainer
    trainer = SFTTrainer(
        model=peft_model,  # 传入准备好 LoRA 的模型
        tokenizer=tokenizer,  # 传入分词器
        train_dataset=train_dataset,  # 训练数据集
        eval_dataset=val_dataset,  # 验证数据集
        dataset_text_field="text",  # 指定数据集中包含完整格式化文本的字段名
        max_seq_length=training_cfg.max_seq_length,  # 再次确认最大序列长度
        args=training_args,  # 传入训练配置
    )

    # 7. 开始训练
    train_result = trainer.train()

    # 8. 保存微调后的 LoRA 适配器权重
    # 这只会保存 LoRA 层的权重，文件很小
    peft_model.save_pretrained(training_cfg.finetuned_model_dir)
    # 同时保存分词器配置，以便后续加载
    tokenizer.save_pretrained(training_cfg.finetuned_model_dir)

    # 9. (可选) 保存合并后的完整模型
    # 如果配置中启用了此选项，则将 LoRA 权重与基础模型权重合并，并保存为一个完整的模型
    if training_cfg.save_merged_model:
        peft_model.save_pretrained_merged(
            training_cfg.merged_model_dir,
            tokenizer,
            save_method="merged_16bit",  # 指定合并后的模型以 16-bit 精度保存
        )

    # 返回训练结果统计信息
    return train_result
