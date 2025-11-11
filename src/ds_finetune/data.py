from __future__ import annotations

import hashlib
import inspect
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Optional

from datasets import Dataset, Features, Sequence as DsSequence, Value

# 本模块封装了训练脚本所需的完整数据处理流水线，主要职责包括：
#   * ModelScope 兼容性修复：ModelScope 依赖旧版 datasets 模块的若干工厂类，
#     通过运行时补丁补齐缺失符号，避免因为签名不匹配导致的加载崩溃。
#   * 原始样本规范化：将原始法律问答数据中的多种字段命名方式整理为统一的
#     `{question, answer}` 结构，便于后续提示模板使用。
#   * 预分词与缓存：提前将文本映射为 token 序列，减少 Unsloth/TRL 在训练时
#     的额外进程与显存占用，使得在中小型 GPU 上也能稳定运行。

from .prompts import format_example, format_inference_prompt

QUESTION_KEYS = ("question", "query", "input")
ANSWER_KEYS = ("answer", "response", "output", "reference")


def _extract_field(records: Mapping[str, object], keys: Iterable[str]) -> str:
    """遍历候选键列表，返回原始样本中第一个非空的字符串字段。"""

    # ModelScope 提供的数据集字段命名并不统一：同一个问题可能被写成
    # `question`、`query`，甚至是字符串列表。这里按照优先级依次检查，
    # 并将获取到的内容规范化为单一字符串。
    for key in keys:
        value = records.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()  # 直接返回第一个非空字符串。
        if isinstance(value, Sequence):
            joined = "\n".join(part.strip() for part in value if isinstance(part, str) and part.strip())
            if joined:
                return joined  # 如果是字符串序列，则拼接成段落形式返回。
    raise KeyError(f"Could not extract field from keys {keys}")


def _normalize_record(record: Mapping[str, object]) -> dict[str, str]:
    """将原始 ModelScope 样本规整为 `{question, answer}` 结构。"""

    return {
        "question": _extract_field(record, QUESTION_KEYS),
        "answer": _extract_field(record, ANSWER_KEYS),
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
        import datasets.load as load_module  # type: ignore
    except Exception:  # pragma: no cover - 防御性处理：环境缺少 datasets 模块时直接跳过
        return

    if getattr(load_module, "_modelscope_compat_patched", False):  # pragma: no cover - 若已打过补丁则快速返回
        return

    from collections.abc import Iterable as _Iterable

    def _install_factory_alias(missing_name: str, fallback_name: str, *, accept_data_dir: bool) -> None:
        """注册兼容别名，使 ModelScope 能按旧名称请求工厂对象。"""
        if hasattr(load_module, missing_name):
            return

        fallback = getattr(load_module, fallback_name, None)
        if fallback is None:
            return

        if not accept_data_dir:
            setattr(load_module, missing_name, fallback)  # 直接创建别名，签名保持一致即可。
            return

        if inspect.isclass(fallback):
            try:
                signature = inspect.signature(fallback.__init__)  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                signature = None

            accepts_var_kwargs = False
            accepted_keywords: set[str] = set()
            if signature is not None:
                for parameter in signature.parameters.values():
                    if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                        accepts_var_kwargs = True
                    elif parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                        if parameter.name != "self":
                            accepted_keywords.add(parameter.name)

            class _CompatFactory(fallback):  # type: ignore[misc,call-arg]
                def __init__(self, *args, data_dir=None, data_files=None, verification_mode=None, **kwargs):  # noqa: ANN001 - legacy signature compatibility
                    legacy_attrs: dict[str, object] = {
                        "data_dir": data_dir,
                        "data_files": data_files,
                        "verification_mode": verification_mode,
                    }

                    for key, value in tuple(kwargs.items()):
                        if not accepts_var_kwargs and key not in accepted_keywords:
                            legacy_attrs[key] = kwargs.pop(key)

                    super_kwargs = kwargs if accepts_var_kwargs else {key: kwargs[key] for key in kwargs if key in accepted_keywords}
                    super().__init__(*args, **super_kwargs)

                    for key, value in legacy_attrs.items():
                        try:
                            setattr(self, key, value)  # 为 ModelScope 内部逻辑保留旧字段。
                        except Exception:
                            pass

            _CompatFactory.__name__ = missing_name
            _CompatFactory.__qualname__ = missing_name
            setattr(load_module, missing_name, _CompatFactory)
        else:

            def _compat_factory(*args, data_dir=None, data_files=None, verification_mode=None, **kwargs):  # noqa: ANN001 - legacy signature compatibility
                return fallback(*args, **kwargs)  # 原函数已接受 **kwargs，因此忽略多余参数。

            _compat_factory.__name__ = missing_name
            setattr(load_module, missing_name, _compat_factory)

    _install_factory_alias("HubDatasetModuleFactoryWithoutScript", "HubDatasetModuleFactoryWithParquetExport", accept_data_dir=True)
    _install_factory_alias("HubDatasetModuleFactoryWithScript", "HubDatasetModuleFactory", accept_data_dir=True)
    _install_factory_alias("LocalDatasetModuleFactoryWithoutScript", "LocalDatasetModuleFactory", accept_data_dir=True)
    _install_factory_alias("LocalDatasetModuleFactoryWithScript", "LocalDatasetModuleFactory", accept_data_dir=True)

    if not hasattr(load_module, "files_to_hash"):

        def _files_to_hash(paths: _Iterable[str]) -> str:
            hasher = hashlib.sha256()
            for path in sorted(paths):
                path_obj = Path(path)
                if path_obj.is_file():
                    with path_obj.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            hasher.update(chunk)
                elif path_obj.exists():
                    hasher.update(str(path_obj.resolve()).encode("utf-8"))
                else:
                    hasher.update(str(path_obj).encode("utf-8"))
            return hasher.hexdigest()

        load_module.files_to_hash = _files_to_hash  # type: ignore[attr-defined]

    if not hasattr(load_module, "resolve_trust_remote_code"):

        def _resolve_trust_remote_code(trust_remote_code, repo_id):  # noqa: ANN001 - external API signature
            if trust_remote_code is not None:
                return bool(trust_remote_code)
            config_value = getattr(load_module.config, "HF_DATASETS_TRUST_REMOTE_CODE", None)
            if config_value is None:
                return False
            return bool(config_value)

        load_module.resolve_trust_remote_code = _resolve_trust_remote_code  # type: ignore[attr-defined]

    if not hasattr(load_module, "init_dynamic_modules"):

        def _init_dynamic_modules(name="datasets_modules", hf_modules_cache=None):  # noqa: ANN001 - legacy signature
            from pathlib import Path as _Path

            base_dir = _Path(hf_modules_cache) if hf_modules_cache is not None else _Path(load_module.config.HF_MODULES_CACHE)
            base_dir.mkdir(parents=True, exist_ok=True)
            module_dir = base_dir / name
            module_dir.mkdir(parents=True, exist_ok=True)
            (module_dir / "__init__.py").touch(exist_ok=True)
            return str(module_dir)

        load_module.init_dynamic_modules = _init_dynamic_modules  # type: ignore[attr-defined]

    if not hasattr(load_module, "_get_importable_file_path"):

        def _get_importable_file_path(dynamic_modules_path, module_namespace, subdirectory_name, name):  # noqa: ANN001 - legacy signature
            import os

            base_directory = os.path.join(dynamic_modules_path, module_namespace, name.replace("/", "--"))
            file_name = name.split("/")[-1] + ".py"
            return os.path.join(base_directory, subdirectory_name, file_name)

        load_module._get_importable_file_path = _get_importable_file_path  # type: ignore[attr-defined]

    if not hasattr(load_module, "_create_importable_file") or not hasattr(load_module, "_load_importable_file"):
        import contextlib
        import filecmp
        import json
        import os
        import shutil
        from pathlib import Path as _Path

        try:
            from filelock import FileLock
        except Exception:  # pragma: no cover - fallback when filelock is unavailable
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
            importable_subdirectory = os.path.join(importable_directory_path, subdirectory_name)
            importable_file = os.path.join(importable_subdirectory, name + ".py")

            lock_path = importable_file + ".lock"
            lock_manager = FileLock(lock_path) if FileLock is not None else contextlib.nullcontext()

            with lock_manager:
                if download_mode == DownloadMode.FORCE_REDOWNLOAD and os.path.exists(importable_directory_path):
                    shutil.rmtree(importable_directory_path)

                os.makedirs(importable_subdirectory, exist_ok=True)
                _Path(importable_directory_path, "__init__.py").touch(exist_ok=True)
                _Path(importable_subdirectory, "__init__.py").touch(exist_ok=True)

                if not os.path.exists(importable_file):
                    shutil.copyfile(original_local_path, importable_file)

                meta_path = os.path.splitext(importable_file)[0] + ".json"
                if not os.path.exists(meta_path):
                    meta = {"original file path": original_local_path, "local file path": importable_file}
                    with open(meta_path, "w", encoding="utf-8") as meta_stream:
                        json.dump(meta, meta_stream)

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

                for file_name, original_path in additional_files:
                    destination = os.path.join(importable_subdirectory, file_name)
                    if not os.path.exists(destination) or not filecmp.cmp(original_path, destination, shallow=False):
                        shutil.copyfile(original_path, destination)

            return importable_file

        def _create_importable_file(
            local_path,  # noqa: ANN001 - legacy signature
            local_imports,
            additional_files,
            dynamic_modules_path,
            module_namespace,
            subdirectory_name,
            name,
            download_mode,
        ) -> None:
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
                except Exception:  # pragma: no cover - logging shouldn't break the flow
                    pass

        def _load_importable_file(
            dynamic_modules_path,  # noqa: ANN001 - legacy signature
            module_namespace,
            subdirectory_name,
            name,
        ):
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
    if _py_utils is not None and not hasattr(_py_utils, "get_imports"):

        def _get_imports(_: str):  # noqa: ANN001 - legacy signature
            return ()

        _py_utils.get_imports = _get_imports  # type: ignore[attr-defined]

    load_module._modelscope_compat_patched = True  # type: ignore[attr-defined]


def _load_raw_dataset(
    name: str,
    subset_name: str,
    split: str,
    cache_dir: Optional[str] = None,
) -> Iterator[MutableMapping[str, object]]:
    _ensure_modelscope_compat()

    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:  # pragma: no cover - 运行环境缺少 ModelScope 时直接提示安装
        raise ImportError("ModelScope is required to load the specified dataset. Please install modelscope>=1.15.0.") from exc

    load_kwargs = {
        "subset_name": subset_name,
        "split": split,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "trust_remote_code": True,
        "use_streaming": True,
    }

    try:
        dataset = MsDataset.load(name, **load_kwargs)
    except Exception:
        # 部分公开数据集的列定义在不同文件间不一致，这里通过显式声明特征
        # 架构，帮助 loader 自动对齐可选字段，避免因 schema 不匹配报错。
        feature_schema = Features(
            {
                "question": Value("string"),
                "answer": Value("string"),
                "input": Value("string"),
                "output": Value("string"),
                "reference": Value("string"),
            }
        )
        load_kwargs["features"] = feature_schema
        dataset = MsDataset.load(name, **load_kwargs)
    for row in dataset:
        yield row


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
    eos_token = tokenizer.eos_token or tokenizer.pad_token or "</s>"

    records = [_normalize_record(row) for row in _load_raw_dataset(dataset_name, dataset_subset, dataset_split, cache_dir)]

    if max_samples is not None and max_samples > 0 and len(records) > max_samples:
        rng = random.Random(random_seed)
        rng.shuffle(records)
        records = records[:max_samples]

    texts = [format_example(record["question"], record["answer"], eos_token) for record in records]
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        return_attention_mask=True,
        padding=False,
    )

    dataset_dict = {
        "question": [record["question"] for record in records],
        "answer": [record["answer"] for record in records],
        "text": texts,
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
    }

    token_features = Features(
        {
            "question": Value("string"),
            "answer": Value("string"),
            "text": Value("string"),
            "input_ids": DsSequence(Value("int32")),
            "attention_mask": DsSequence(Value("int32")),
        }
    )

    indices = list(range(len(texts)))
    rng = random.Random(random_seed ^ 0xA5A5)
    rng.shuffle(indices)

    eval_size = int(len(indices) * eval_ratio)
    if eval_ratio > 0 and eval_size == 0 and len(indices) > 1:
        eval_size = 1
    if eval_size >= len(indices):
        eval_size = max(0, len(indices) - 1)

    eval_indices = set(indices[:eval_size])
    train_indices = [idx for idx in indices if idx not in eval_indices]
    val_indices = [idx for idx in indices if idx in eval_indices]

    def _build_subset(selected: list[int]) -> Dataset:
        subset = {key: [column[i] for i in selected] for key, column in dataset_dict.items()}
        return Dataset.from_dict(subset, features=token_features)

    empty_subset = {key: [] for key in dataset_dict}

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
    records = [_normalize_record(row) for row in _load_raw_dataset(dataset_name, dataset_subset, dataset_split, cache_dir)]
    base_features = Features({"question": Value("string"), "answer": Value("string")})
    hf_dataset = Dataset.from_list(records, features=base_features)

    formatted = hf_dataset.map(lambda ex: {"text": format_inference_prompt(ex["question"])}, batched=False)
    formatted = formatted.map(lambda ex: {k: [v] for k, v in ex.items()}, batched=True, batch_size=batch_size)
    return formatted
