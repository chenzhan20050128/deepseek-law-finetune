# DeepSeek 蒸馏模型法律微调流水线

本项目将原有的基于Jupyter Notebook的实验流程打包成可复用的Python模块和CLI工具，用于在https://www.modelscope.cn/datasets/xxxcoder/law_finetune数据集上对DeepSeek蒸馏模型进行微调。

## 项目结构

```
├── requirements.txt          # Python 3.12 + CUDA 12.8 的运行时依赖
├── run_ds_finetune_pipeline.sh
├── src/
│   └── ds_finetune/
│       ├── cli.py            # 命令行接口入口点
│       ├── config.py         # 描述训练/评估设置的数据类
│       ├── data.py           # 通过ModelScope加载数据集 + 提示词格式化
│       ├── evaluate.py       # 基于相似度的评估工具
│       ├── modeling.py       # 模型加载、LoRA准备、生成辅助函数
│       ├── prompts.py        # 跨任务共享的提示词模板
│       └── train.py          # 使用TRL的SFTTrainer的主要微调流程
└── tools/
		└── finetune_data_process.py # 旧版数据转换辅助工具（可选）
```

## 环境要求

目标平台：

- Linux（已在Ubuntu 22.04服务器上测试）
- CUDA 12.8驱动 + 工具链
- Python 3.12
- PyTorch 2.8.0（CUDA版本）

创建并激活虚拟环境，然后安装依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

下载所需的模型资源（路径见下文引用）：

- 基础DeepSeek蒸馏模型，例如 `DeepSeek-R1-Distill-Qwen-1.5B`
- 用于评估的Sentence Transformer模型，例如 `text2vec-base-chinese`

## 快速开始

设置指向已下载模型的环境变量，然后调用流水线脚本。该脚本会将任何额外参数转发给CLI，允许进一步定制（批大小、训练轮数等）。

```bash
./run_ds_finetune_pipeline.sh --num-train-epochs 1 --eval-trained
```

该命令执行以下步骤：

1.  使用 `prompts.py` 中的提示词模板加载并格式化ModelScope数据集。
2.  使用 `trl.SFTTrainer` 对基础模型应用LoRA微调。
3.  保存LoRA检查点（`<experiment>_base`）和合并后的16位模型（`<experiment>_merged`）。
4.  当提供 `--eval-original` 或 `--eval-trained` 参数时，可选择性地运行余弦相似度评估。

### 模型资源布局与自动下载

默认情况下，所有模型将下载到 `/root/autodl-tmp/models` 目录下：

```
/root/autodl-tmp/models
├── base/               # LoRA基础模型（例如 DeepSeek R1 Distill）
└── text2vec/           # 用于评估的Sentence-transformer模型
```

您可以通过 `DS_MODELS_ROOT` 环境变量自定义根目录。CLI接受显式的文件系统路径（`--base-model-path`, `--text2vec-model-path`），或者将自动下载由 `--base-model-id` 和 `--text2vec-model-id` 指定的模型仓库（默认分别为 `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` 和 `shibing624/text2vec-base-chinese`）。下载使用 `huggingface_hub.snapshot_download` 执行，如果提供了 `HF_TOKEN` 和/或 `HF_ENDPOINT` 则会使用它们。镜像配置示例：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN=hf_************************
```

## CLI 使用

您可以直接调用CLI以获得更精细的控制：

```bash
python -m ds_finetune.cli train \
	--experiment-name ds_r1_law_1.5B_exp4 \
	--batch-size 16 \
	--num-train-epochs 3 \
	--eval-trained
```

如果未提供本地路径，CLI将解析（并在需要时下载）通过 `--base-model-id` / `--text2vec-model-id` 声明的模型。

其他子命令：

*   `evaluate`: 对原始或微调后的模型运行评估。
*   `infer`: 使用微调后的模型进行推理。

运行 `python -m ds_finetune.cli --help` 获取完整的选项列表和默认值。

## 数据集访问

训练和评估数据集通过ModelScope按需获取：

```python
from modelscope.msdatasets import MsDataset

dataset = MsDataset.load('xxxcoder/law_finetune', subset_name='default', split='train')
```

如果您偏好离线使用，可以使用 `modelscope` CLI 下载数据集，并将 `--cache-dir` 指向本地缓存位置。

## 注意事项

- 原始的Jupyter Notebook仍供参考，但不再用于自动化流程。
- 使用 `load_in_4bit` 进行训练时，请确保GPU内存充足（建议24GB或以上）。使用 `--no-load-in-4bit` 可以切换到更高精度，但会消耗更多内存。
- 训练和评估的输出分别保存到 `outputs/`、`<experiment>_base/`、`<experiment>_merged/` 和 `<experiment>_eval_result/` 目录。