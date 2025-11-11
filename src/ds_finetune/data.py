from __future__ import annotations

import hashlib  # 导入哈希库，用于计算文件内容的哈希值，以验证缓存
import inspect  # 导入 inspect 模块，用于获取函数签名等信息，以实现动态补丁
import random  # 导入随机数模块，用于数据抽样和打乱顺序
from collections.abc import Sequence  # 从抽象基类中导入 Sequence，用于类型提示
from pathlib import Path  # 导入 Path 对象，用于以面向对象的方式操作文件路径
from typing import Iterable, Iterator, Mapping, MutableMapping, Optional  # 导入类型提示相关的工具

from datasets import Dataset, Features, Sequence as DsSequence, Value  # 从 datasets 库导入核心类

# 本模块封装了训练脚本所需的完整数据处理流水线，主要职责包括：
#   * ModelScope 兼容性修复：ModelScope 依赖旧版 datasets 模块的若干工厂类，
#     通过运行时补丁补齐缺失符号，避免因为签名不匹配导致的加载崩溃。
#   * 原始样本规范化：将原始法律问答数据中的多种字段命名方式整理为统一的
#     `{question, answer}` 结构，便于后续提示模板使用。
#   * 预分词与缓存：提前将文本映射为 token 序列，减少 Unsloth/TRL 在训练时
#     的额外进程与显存占用，使得在中小型 GPU 上也能稳定运行。

from .prompts import format_example, format_inference_prompt  # 从同级目录的 prompts 模块导入格式化函数

# 定义问题和答案的可能键名，以兼容不同来源的数据集
QUESTION_KEYS = ("question", "query", "input")
ANSWER_KEYS = ("answer", "response", "output", "reference")


def _extract_field(records: Mapping[str, object], keys: Iterable[str]) -> str:
    """遍历候选键列表，返回原始样本中第一个非空的字符串字段。"""

    # ModelScope 提供的数据集字段命名并不统一：同一个问题可能被写成
    # `question`、`query`，甚至是字符串列表。这里按照优先级依次检查，
    # 并将获取到的内容规范化为单一字符串。
    for key in keys:
        value = records.get(key)  # 尝试获取字段值
        if isinstance(value, str) and value.strip():
            return value.strip()  # 直接返回第一个非空字符串，并去除首尾空白。
        if isinstance(value, Sequence):
            # 如果字段值是一个序列（如列表），则将其中的非空字符串拼接成一个段落
            joined = "\n".join(part.strip() for part in value if isinstance(part, str) and part.strip())
            if joined:
                return joined  # 如果拼接后内容不为空，则返回。
    # 如果遍历完所有候选键都没有找到有效内容，则抛出异常
    raise KeyError(f"Could not extract field from keys {keys}")


def _normalize_record(record: Mapping[str, object]) -> dict[str, str]:
    """将原始 ModelScope 样本规整为 `{question, answer}` 结构。"""

    # 这个函数通过调用 _extract_field 来统一数据格式
    return {
        "question": _extract_field(record, QUESTION_KEYS),  # 提取并规范化 "question" 字段
        "answer": _extract_field(record, ANSWER_KEYS),  # 提取并规范化 "answer" 字段
    }


def _ensure_modelscope_compat() -> None:
    """为 ModelScope 打补丁，使其能在新版 datasets 模块下继续工作。

    ModelScope 的数据集封装编写于旧版 `datasets` API，随着官方升级，
    多个工厂类被重命名或移除。如果直接调用会抛出如
    `HubDatasetModuleFactoryWithParquetExport.__init__()` 接收到未知参数
    `data_dir` 的错误。为了避免降级依赖，我们在运行时动态创建别名，
    让 ModelScope 继续访问旧的入口点，同时保证补丁对其余训练代码透明。
    """

    try:
        import datasets.load as load_module  # type: ignore # 尝试导入 datasets 的加载模块
    except Exception:  # pragma: no cover - 防御性处理：环境缺少 datasets 模块时直接跳过
        return

    if getattr(load_module, "_modelscope_compat_patched", False):  # pragma: no cover - 若已打过补丁则快速返回
        return

    from collections.abc import Iterable as _Iterable

    def _install_factory_alias(missing_name: str, fallback_name: str, *, accept_data_dir: bool) -> None:
        """注册兼容别名，使 ModelScope 能按旧名称请求工厂对象。"""
        if hasattr(load_module, missing_name):
            return  # 如果目标别名已存在，则无需操作

        fallback = getattr(load_module, fallback_name, None)  # 获取新版 datasets 中对应的类
        if fallback is None:
            return  # 如果新版中也找不到，则无法创建别名

        if not accept_data_dir:
            setattr(load_module, missing_name, fallback)  # 如果不需要处理旧参数，直接创建别名
            return

        if inspect.isclass(fallback):
            # 如果 fallback 是一个类，我们需要创建一个包装类来处理不兼容的参数
            try:
                signature = inspect.signature(fallback.__init__)  # type: ignore[attr-defined] # 获取构造函数的签名
            except (TypeError, ValueError):
                signature = None

            accepts_var_kwargs = False  # 检查是否接受任意关键字参数
            accepted_keywords: set[str] = set()  # 记录已知的关键字参数
            if signature is not None:
                for parameter in signature.parameters.values():
                    if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                        accepts_var_kwargs = True
                    elif parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                        if parameter.name != "self":
                            accepted_keywords.add(parameter.name)

            # 定义一个兼容性包装类
            class _CompatFactory(fallback):  # type: ignore[misc,call-arg]
                def __init__(self, *args, data_dir=None, data_files=None, verification_mode=None, **kwargs):  # noqa: ANN001 - 兼容旧版签名
                    # 收集旧版 API 的参数
                    legacy_attrs: dict[str, object] = {
                        "data_dir": data_dir,
                        "data_files": data_files,
                        "verification_mode": verification_mode,
                    }

                    # 将 kwargs 中不被新版构造函数接受的参数也移入 legacy_attrs
                    for key, value in tuple(kwargs.items()):
                        if not accepts_var_kwargs and key not in accepted_keywords:
                            legacy_attrs[key] = kwargs.pop(key)

                    # 筛选出新版构造函数可以接受的参数
                    super_kwargs = kwargs if accepts_var_kwargs else {key: kwargs[key] for key in kwargs if key in accepted_keywords}
                    super().__init__(*args, **super_kwargs)  # 调用父类的构造函数

                    # 将旧版参数作为属性设置到实例上，以供 ModelScope 内部逻辑使用
                    for key, value in legacy_attrs.items():
                        try:
                            setattr(self, key, value)
                        except Exception:
                            pass

            _CompatFactory.__name__ = missing_name
            _CompatFactory.__qualname__ = missing_name
            setattr(load_module, missing_name, _CompatFactory)  # 将包装类注册为别名
        else:
            # 如果 fallback 是一个函数，定义一个兼容性包装函数
            def _compat_factory(*args, data_dir=None, data_files=None, verification_mode=None, **kwargs):  # noqa: ANN001 - 兼容旧版签名
                return fallback(*args, **kwargs)  # 原函数已接受 **kwargs，因此直接忽略多余参数

            _compat_factory.__name__ = missing_name
            setattr(load_module, missing_name, _compat_factory)  # 注册包装函数

    # 为一系列在 datasets 库新版本中被重命名或移除的工厂类创建别名
    _install_factory_alias("HubDatasetModuleFactoryWithoutScript", "HubDatasetModuleFactoryWithParquetExport", accept_data_dir=True)
    _install_factory_alias("HubDatasetModuleFactoryWithScript", "HubDatasetModuleFactory", accept_data_dir=True)
    _install_factory_alias("LocalDatasetModuleFactoryWithoutScript", "LocalDatasetModuleFactory", accept_data_dir=True)
    _install_factory_alias("LocalDatasetModuleFactoryWithScript", "LocalDatasetModuleFactory", accept_data_dir=True)

    # 如果 `files_to_hash` 函数不存在，则提供一个实现
    if not hasattr(load_module, "files_to_hash"):

        def _files_to_hash(paths: _Iterable[str]) -> str:
            """计算给定文件集合的稳定哈希，用于判断缓存是否需要刷新。"""
            hasher = hashlib.sha256()  # 使用 SHA256 算法
            for path in sorted(paths):  # 对路径排序以保证哈希的稳定性
                path_obj = Path(path)
                if path_obj.is_file():
                    # 如果是文件，则逐块读取并更新哈希值，避免一次性加载大文件
                    with path_obj.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            hasher.update(chunk)
                elif path_obj.exists():
                    # 如果是目录或其他存在的文件系统对象，则使用其解析后的绝对路径进行哈希
                    hasher.update(str(path_obj.resolve()).encode("utf-8"))
                else:
                    # 如果路径不存在，则直接对路径字符串进行哈希
                    hasher.update(str(path_obj).encode("utf-8"))
            return hasher.hexdigest()  # 返回十六进制表示的哈希值

        load_module.files_to_hash = _files_to_hash  # type: ignore[attr-defined] # 注册函数

    # 如果 `resolve_trust_remote_code` 函数不存在，则提供一个实现
    if not hasattr(load_module, "resolve_trust_remote_code"):

        def _resolve_trust_remote_code(trust_remote_code, repo_id):  # noqa: ANN001 - 兼容外部 API 签名
            """兼容旧版签名，根据参数或环境变量确定是否信任远程代码。"""
            if trust_remote_code is not None:
                return bool(trust_remote_code)
            # 检查环境变量 `HF_DATASETS_TRUST_REMOTE_CODE`
            config_value = getattr(load_module.config, "HF_DATASETS_TRUST_REMOTE_CODE", None)
            if config_value is None:
                return False
            return bool(config_value)

        load_module.resolve_trust_remote_code = _resolve_trust_remote_code  # type: ignore[attr-defined] # 注册函数

    # 如果 `init_dynamic_modules` 函数不存在，则提供一个实现
    if not hasattr(load_module, "init_dynamic_modules"):

        def _init_dynamic_modules(name="datasets_modules", hf_modules_cache=None):  # noqa: ANN001 - 兼容旧版签名
            """创建数据集动态模块缓存目录，模仿旧版 datasets 的行为。"""
            from pathlib import Path as _Path

            # 确定缓存基目录
            base_dir = _Path(hf_modules_cache) if hf_modules_cache is not None else _Path(load_module.config.HF_MODULES_CACHE)
            base_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
            module_dir = base_dir / name
            module_dir.mkdir(parents=True, exist_ok=True)
            (module_dir / "__init__.py").touch(exist_ok=True)  # 创建 __init__.py 使其成为一个包
            return str(module_dir)

        load_module.init_dynamic_modules = _init_dynamic_modules  # type: ignore[attr-defined] # 注册函数

    # 如果 `_get_importable_file_path` 函数不存在，则提供一个实现
    if not hasattr(load_module, "_get_importable_file_path"):

        def _get_importable_file_path(dynamic_modules_path, module_namespace, subdirectory_name, name):  # noqa: ANN001 - 兼容旧版签名
            """根据旧规则拼接模块 .py 文件路径，使 ModelScope 能够 import。"""
            import os

            # 构建符合旧版 datasets 结构的模块路径
            base_directory = os.path.join(dynamic_modules_path, module_namespace, name.replace("/", "--"))
            file_name = name.split("/")[-1] + ".py"
            return os.path.join(base_directory, subdirectory_name, file_name)

        load_module._get_importable_file_path = _get_importable_file_path  # type: ignore[attr-defined] # 注册函数

    # 如果 `_create_importable_file` 或 `_load_importable_file` 不存在，则提供实现
    if not hasattr(load_module, "_create_importable_file") or not hasattr(load_module, "_load_importable_file"):
        import contextlib
        import filecmp
        import json
        import os
        import shutil
        from pathlib import Path as _Path

        try:
            from filelock import FileLock  # 尝试导入文件锁，用于多进程同步
        except Exception:  # pragma: no cover - 当依赖缺失时降级为无锁行为
            FileLock = None  # type: ignore[assignment]

        from datasets.download.download_manager import DownloadMode

        def _copy_script_and_other_resources_in_importable_dir(
            *,
            name: str,
            importable_directory_path: str,
            subdirectory_name: str,
            original_local_path: str,
            local_imports: _Iterable[tuple[str, str]],
            additional_files: _Iterable[tuple[str, str]],
            download_mode: Optional[DownloadMode],
        ) -> str:
            """复制脚本及依赖文件到动态模块目录，复刻旧版缓存布局。"""
            importable_subdirectory = os.path.join(importable_directory_path, subdirectory_name)
            importable_file = os.path.join(importable_subdirectory, name + ".py")

            lock_path = importable_file + ".lock"
            # 使用文件锁确保多进程或多线程下文件操作的原子性
            lock_manager = FileLock(lock_path) if FileLock is not None else contextlib.nullcontext()

            with lock_manager:
                # 如果强制重新下载，则删除旧的缓存目录
                if download_mode == DownloadMode.FORCE_REDOWNLOAD and os.path.exists(importable_directory_path):
                    shutil.rmtree(importable_directory_path)

                os.makedirs(importable_subdirectory, exist_ok=True)
                _Path(importable_directory_path, "__init__.py").touch(exist_ok=True)
                _Path(importable_subdirectory, "__init__.py").touch(exist_ok=True)

                # 复制主脚本文件
                if not os.path.exists(importable_file):
                    shutil.copyfile(original_local_path, importable_file)

                # 创建元数据文件，记录原始路径
                meta_path = os.path.splitext(importable_file)[0] + ".json"
                if not os.path.exists(meta_path):
                    meta = {"original file path": original_local_path, "local file path": importable_file}
                    with open(meta_path, "w", encoding="utf-8") as meta_stream:
                        json.dump(meta, meta_stream)

                # 复制本地依赖的 Python 模块
                for import_name, import_path in local_imports:
                    if os.path.isfile(import_path):
                        destination = os.path.join(importable_subdirectory, import_name + ".py")
                        if not os.path.exists(destination):
                            shutil.copyfile(import_path, destination)
                    elif os.path.isdir(import_path):
                        destination_dir = os.path.join(importable_subdirectory, import_name)
                        if not os.path.exists(destination_dir):
                            shutil.copytree(import_path, destination_dir)
                    else:
                        raise ImportError(f"Unable to copy local import at {import_path}")

                # 复制其他附加文件
                for file_name, original_path in additional_files:
                    destination = os.path.join(importable_subdirectory, file_name)
                    if not os.path.exists(destination) or not filecmp.cmp(original_path, destination, shallow=False):
                        shutil.copyfile(original_path, destination)

            return importable_file

        def _create_importable_file(
            local_path,  # noqa: ANN001 - 兼容旧版签名
            local_imports,
            additional_files,
            dynamic_modules_path,
            module_namespace,
            subdirectory_name,
            name,
            download_mode,
        ) -> None:
            """在本地缓存目录下创建 importable 模块，供 ModelScope 导入。"""
            base_directory = os.path.join(dynamic_modules_path, module_namespace, name.replace("/", "--"))
            _Path(base_directory).mkdir(parents=True, exist_ok=True)
            _Path(base_directory).parent.joinpath("__init__.py").touch(exist_ok=True)

            importable_local_file = _copy_script_and_other_resources_in_importable_dir(
                name=name.split("/")[-1],
                importable_directory_path=base_directory,
                subdirectory_name=subdirectory_name,
                original_local_path=local_path,
                local_imports=local_imports,
                additional_files=additional_files,
                download_mode=download_mode,
            )

            logger = getattr(load_module, "logger", None)
            if logger is not None:
                try:
                    logger.debug("Created importable dataset file at %s", importable_local_file)
                except Exception:  # pragma: no cover - 日志失败不应影响主要流程
                    pass

        def _load_importable_file(
            dynamic_modules_path,  # noqa: ANN001 - 兼容旧版签名
            module_namespace,
            subdirectory_name,
            name,
        ):
            """返回符合旧版约定的模块路径，便于通过 importlib 加载。"""
            # 构造 Python 的点分模块路径，例如：datasets_modules.community.some_dataset.some_dataset
            module_path = ".".join(
                [
                    os.path.basename(dynamic_modules_path.rstrip(os.sep)),
                    module_namespace,
                    name.replace("/", "--"),
                    subdirectory_name,
                    name.split("/")[-1],
                ]
            )
            return module_path, subdirectory_name

        load_module._create_importable_file = _create_importable_file  # type: ignore[attr-defined]
        load_module._load_importable_file = _load_importable_file  # type: ignore[attr-defined]

    try:
        from datasets.utils import py_utils as _py_utils  # type: ignore
    except Exception:  # pragma: no cover - optional dependency might be missing
        _py_utils = None
    # 为 `get_imports` 提供一个空实现，因为在新版中它可能不存在
    if _py_utils is not None and not hasattr(_py_utils, "get_imports"):

        def _get_imports(_: str):  # noqa: ANN001 - 兼容旧版签名
            return ()

        _py_utils.get_imports = _get_imports  # type: ignore[attr-defined]

    # 标记补丁已完成，避免重复执行
    load_module._modelscope_compat_patched = True  # type: ignore[attr-defined]


def _load_raw_dataset(
    name: str,
    subset_name: str,
    split: str,
    cache_dir: Optional[str] = None,
) -> Iterator[MutableMapping[str, object]]:
    """以流式方式读取 ModelScope 数据集，返回原始记录迭代器。"""
    # 确保 ModelScope 兼容性补丁已应用
    _ensure_modelscope_compat()

    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:  # pragma: no cover - 运行环境缺少 ModelScope 时直接提示安装
        raise ImportError("ModelScope is required to load the specified dataset. Please install modelscope>=1.15.0.") from exc

    # 配置加载参数
    load_kwargs = {
        "subset_name": subset_name,  # 数据集子集
        "split": split,  # 数据切分，如 'train', 'validation'
        "cache_dir": str(cache_dir) if cache_dir is not None else None,  # 缓存目录
        "trust_remote_code": True,  # 信任远程代码，ModelScope 加载需要
        "use_streaming": True,  # 使用流式加载，避免一次性将整个数据集加载到内存
    }

    try:
        dataset = MsDataset.load(name, **load_kwargs)
    except Exception:
        # 部分公开数据集的列定义在不同文件间不一致，这里通过显式声明特征
        # 架构，帮助 loader 自动对齐可选字段，避免因 schema 不匹配报错。
        feature_schema = Features(
            {
                # 定义所有可能出现的字段及其类型，允许加载器处理不一致的列
                "question": Value("string"),
                "answer": Value("string"),
                "input": Value("string"),
                "output": Value("string"),
                "instruction": Value("string"),
                "response": Value("string"),
                "history": DsSequence(DsSequence(Value("string"))),
            }
        )
        load_kwargs["features"] = feature_schema
        dataset = MsDataset.load(name, **load_kwargs)
    for row in dataset:
        yield row  # 逐条返回数据记录


def build_training_dataset(
    *,
    dataset_name: str,
    dataset_subset: str,
    dataset_split: str,
    tokenizer,
    max_seq_length: int,
    eval_ratio: float,
    random_seed: int,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> tuple[Dataset, Dataset]:
    """构建用于训练和验证的 Dataset 对象。"""
    # 获取或设置 EOS (end-of-sentence) token
    eos_token = tokenizer.eos_token or tokenizer.pad_token or "</s>"

    # 从 ModelScope 加载原始数据，并立即进行规范化处理
    records = [_normalize_record(row) for row in _load_raw_dataset(dataset_name, dataset_subset, dataset_split, cache_dir)]

    # 如果设置了最大样本数，则进行随机抽样
    if max_samples is not None and max_samples > 0 and len(records) > max_samples:
        rng = random.Random(random_seed)
        # 使用不同的种子进行第二次随机化，以增加随机性
        rng.shuffle(records)
        records = records[:max_samples]

    # 将问答对格式化为模型训练所需的文本格式
    texts = [format_example(record["question"], record["answer"], eos_token) for record in records]
    # 使用 tokenizer 对文本进行编码
    encodings = tokenizer(
        texts,
        truncation=True,  # 开启截断，确保序列长度不超过 max_seq_length
        max_length=max_seq_length,
        return_attention_mask=True,  # 返回 attention mask
        padding=False,  # 不进行填充，后续由 DataCollator 处理
    )

    # 将所有处理好的数据整合到一个字典中
    dataset_dict = {
        "question": [record["question"] for record in records],
        "answer": [record["answer"] for record in records],
        "text": texts,
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }

    # 定义 Dataset 的特征 schema，这有助于提高效率和确保数据类型正确
    token_features = Features(
        {
            "question": Value("string"),
            "answer": Value("string"),
            "text": Value("string"),
            "input_ids": DsSequence(Value("int32")),
            "attention_mask": DsSequence(Value("int32")),
        }
    )

    # 创建索引并打乱，用于划分训练集和验证集
    indices = list(range(len(texts)))
    rng = random.Random(random_seed ^ 0xA5A5)  # 使用异或操作的种子，避免与之前的随机操作重复
    rng.shuffle(indices)

    # 计算验证集的大小
    eval_size = int(len(indices) * eval_ratio)
    # 确保即使在数据量很少的情况下，只要比例大于0，至少有一个验证样本
    if eval_ratio > 0 and eval_size == 0 and len(indices) > 1:
        eval_size = 1
    # 防止验证集过大，至少保留一个训练样本
    if eval_size >= len(indices):
        eval_size = max(0, len(indices) - 1)

    # 划分训练集和验证集的索引
    eval_indices = set(indices[:eval_size])
    train_indices = [idx for idx in indices if idx not in eval_indices]
    val_indices = [idx for idx in indices if idx in eval_indices]

    # 定义一个辅助函数，根据索引构建 Dataset 子集
    def _build_subset(selected: list[int]) -> Dataset:
        subset = {key: [column[i] for i in selected] for key, column in dataset_dict.items()}
        return Dataset.from_dict(subset, features=token_features)

    # 创建一个空的字典结构，用于在没有数据时创建空的数据集
    empty_subset = {key: [] for key in dataset_dict}

    # 构建训练和验证数据集
    train_dataset = _build_subset(train_indices) if train_indices else Dataset.from_dict(empty_subset, features=token_features)
    val_dataset = _build_subset(val_indices) if val_indices else Dataset.from_dict(empty_subset, features=token_features)

    return train_dataset, val_dataset


def build_evaluation_dataset(
    *,
    dataset_name: str,
    dataset_subset: str,
    dataset_split: str,
    cache_dir: Optional[str] = None,
    batch_size: int,
) -> Dataset:
    """构造评估阶段使用的 Dataset，保留原问题文本并批量包装提示。"""
    # 加载并规范化数据
    records = [_normalize_record(row) for row in _load_raw_dataset(dataset_name, dataset_subset, dataset_split, cache_dir)]
    # 定义基础特征
    base_features = Features({"question": Value("string"), "answer": Value("string")})
    # 从记录列表创建 Hugging Face Dataset
    hf_dataset = Dataset.from_list(records, features=base_features)

    # 使用 map 函数将每个问题格式化为推理时所需的提示格式
    formatted = hf_dataset.map(lambda ex: {"text": format_inference_prompt(ex["question"])}, batched=False)
    # 将单条样本包装成批次结构（list of values），这是为了适配某些评估流程的输入要求
    formatted = formatted.map(lambda ex: {k: [v] for k, v in ex.items()}, batched=True, batch_size=batch_size)
    return formatted
