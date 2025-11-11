# 从 __future__ 导入 annotations，允许在类型提示中使用前向引用
from __future__ import annotations

from typing import Optional, Sequence  # 导入类型提示工具

import torch  # 导入 PyTorch 库
from unsloth import FastLanguageModel, is_bfloat16_supported  # 从 unsloth 库导入核心模型类和硬件支持检测函数

from .config import TrainingConfig  # 从本地 config 模块导入训练配置类


def load_base_model(config: TrainingConfig):
    """
    使用 Unsloth 的 FastLanguageModel 加载预训练的基础模型和分词器。

    Unsloth 对 Hugging Face 的 `from_pretrained` 方法进行了优化，可以显著加快模型加载速度，
    并集成了 4-bit/8-bit 量化、Flash Attention 等功能。

    Args:
        config (TrainingConfig): 包含模型加载所需参数的配置对象。

    Returns:
        tuple: 返回一个元组，包含加载的模型对象和分词器对象。
    """
    # 调用 FastLanguageModel.from_pretrained 方法加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_path,  # 指定要加载的模型的路径或 Hugging Face Hub 上的名称
        max_seq_length=config.max_seq_length,  # 设置模型能处理的最大序列长度
        dtype=config.dtype,  # 设置模型加载时的数据类型 (如 torch.float16, torch.bfloat16, or "auto")
        load_in_4bit=config.load_in_4bit,  # 是否以 4-bit 精度加载模型以节省显存
        load_in_8bit=config.load_in_8bit,  # 是否以 8-bit 精度加载模型
        # Unsloth 特有的参数，用于在 LoRA 微调时准备模型，这里根据配置决定是否启用
        # 注意：这里的 full_finetuning 参数在 Unsloth 的上下文中可能指的是准备模型以支持不同类型的微调
        # 而非传统意义上的全参数微调。在 LoRA 场景下，通常应为 False 或根据 Unsloth 文档调整。
        # 在此项目中，它可能用于控制模型内部的某些准备步骤。
        full_finetuning=config.full_finetuning,
    )
    return model, tokenizer


def prepare_lora_model(model, config: TrainingConfig):
    """
    为基础模型添加 LoRA (Low-Rank Adaptation) 适配器，使其准备好进行 PEFT (Parameter-Efficient Fine-Tuning)。

    Unsloth 的 `get_peft_model` 方法是 `peft` 库的封装和优化，可以更高效地将 LoRA 层应用到模型上。

    Args:
        model: 已经加载的基础语言模型。
        config (TrainingConfig): 包含 LoRA 配置参数的对象。

    Returns:
        PeftModel: 添加了 LoRA 适配器后的模型，可以进行高效微调。
    """
    # 使用 FastLanguageModel.get_peft_model 为模型添加 LoRA 层
    peft_model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,  # LoRA 的秩 (rank)，决定了低秩矩阵的大小，是关键超参数
        target_modules=config.lora_target_modules(),  # 指定要应用 LoRA 的模型模块，如 'q_proj', 'v_proj'
        lora_alpha=config.lora_alpha,  # LoRA 的缩放因子，alpha/r 是缩放权重，也是重要超参数
        lora_dropout=config.lora_dropout,  # 在 LoRA 层上应用的 dropout 概率，用于正则化
        bias="none",  # 设置 bias 的训练方式，"none" 表示不训练 bias，是 LoRA 的标准实践
        # 是否使用梯度检查点。Unsloth 推荐使用其自定义的 "unsloth" 实现，以节省显存
        use_gradient_checkpointing="unsloth" if config.enable_gradient_checkpointing else False,
        random_state=config.random_seed,  # 设置随机种子以保证 LoRA 权重初始化的可复现性
        use_rslora=False,  # 是否使用 Rank-Stabilized LoRA，一种 LoRA 的变体
        loftq_config=None,  # LoftQ 的配置，一种初始化 LoRA 权重的方法
    )
    return peft_model


def ensure_bfloat16() -> tuple[bool, bool]:
    """
    检查当前硬件和环境是否支持 bfloat16 数据类型。

    bfloat16 是一种数值格式，对于现代 GPU（如 Ampere 架构及以后）可以加速训练并保持较好的精度。
    Unsloth 强烈建议在支持的硬件上使用 bfloat16。

    Returns:
        tuple[bool, bool]: 返回一个元组 (needs_patch, supported)。
                           `needs_patch` 在此项目中似乎未使用，但 `supported` 表明了支持状态。
                           在原始 unsloth 实现中，这个函数可能还用于触发一些兼容性补丁。
    """
    bf16_support = is_bfloat16_supported()  # 调用 unsloth 的函数进行检测
    # 返回一个元组，第一个元素表示是否需要打补丁（这里简单地设为不支持），第二个元素是支持状态
    return not bf16_support, bf16_support


def generate_responses(
    model,
    tokenizer,
    prompts: Sequence[str],
    max_length: int,
    device: Optional[str] = None,
) -> list[str]:
    """
    使用微调后的模型为一批提示（prompts）生成回复。

    Args:
        model: 用于推理的模型 (通常是微调后合并了 LoRA 权重的模型)。
        tokenizer: 对应的分词器。
        prompts (Sequence[str]): 一个包含多个问题提示的字符串列表。
        max_length (int): 生成文本的最大长度。
        device (Optional[str]): 指定运行推理的设备 ('cuda' 或 'cpu')。如果为 None，则自动检测。

    Returns:
        list[str]: 包含每个提示对应生成回复的字符串列表。
    """
    # 自动选择设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Unsloth 提供的优化步骤，用于将模型切换到高效的推理模式。
    # 这可能会包括应用 Flash Attention、融合操作等。
    FastLanguageModel.for_inference(model)
    
    # 使用分词器对所有提示进行批量编码
    inputs = tokenizer(
        list(prompts),
        return_tensors="pt",  # 返回 PyTorch 张量
        padding=True,  # 对批次内的序列进行填充，使其长度一致
        truncation=True,  # 对超过最大长度的序列进行截断
    ).to(device)  # 将输入张量移动到指定的设备

    # 调用模型的 generate 方法进行批量生成
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,  # 控制生成内容的最大长度
        use_cache=True,  # 使用键/值缓存 (key/value cache) 来加速自回归生成过程
    )

    # 将生成的 token ID 批量解码回文本字符串
    responses = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,  # 跳过特殊 token (如 [PAD], [EOS])
        clean_up_tokenization_spaces=True,  # 清理分词过程中可能产生的多余空格
    )
    return responses
