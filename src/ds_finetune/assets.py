from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError

DEFAULT_BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_TEXT2VEC_MODEL_ID = "shibing624/text2vec-base-chinese"

_DEFAULT_ROOT = Path(os.environ.get("DS_MODELS_ROOT", "/root/autodl-tmp/models"))
BASE_MODELS_DIR = _DEFAULT_ROOT / "base"
TEXT2VEC_MODELS_DIR = _DEFAULT_ROOT / "text2vec"


def _sanitize_directory_name(model_id: str) -> str:
    return model_id.replace("/", "__")


def _has_model_weights(target_dir: Path) -> bool:
    weight_patterns = ("*.safetensors", "*.bin", "*.pt")
    for pattern in weight_patterns:
        if any(target_dir.rglob(pattern)):
            return True
    return False


def _resolve_target_dir(preferred_path: Optional[str], default_root: Path, model_id: str) -> Path:
    if preferred_path:
        return Path(preferred_path)
    return default_root / _sanitize_directory_name(model_id)


def _download_if_missing(model_id: str, *, target_dir: Path, force: bool = False) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)

    has_content = any(target_dir.iterdir())
    if has_content and not force and _has_model_weights(target_dir):
        return target_dir

    mirror_endpoint = os.environ.get("DS_HF_PRIMARY_ENDPOINT")
    fallback_endpoint = os.environ.get("DS_HF_FALLBACK_ENDPOINT", "https://huggingface.co")

    def _perform_download(endpoint: Optional[str], *, force_download: bool = False) -> None:
        download_kwargs = {
            "repo_id": model_id,
            "local_dir": str(target_dir),
            "local_dir_use_symlinks": False,
            "resume_download": True,
            "use_auth_token": os.environ.get("HF_TOKEN"),
            "max_workers": int(os.environ.get("HF_DOWNLOAD_THREADS", "8")),
        }
        if force_download:
            download_kwargs["force_download"] = True
        if endpoint:
            download_kwargs["endpoint"] = endpoint
        snapshot_download(**download_kwargs)

    fallback_attempted = False
    try:
        _perform_download(mirror_endpoint, force_download=force)
        if not _has_model_weights(target_dir) and mirror_endpoint and mirror_endpoint != fallback_endpoint:
            _perform_download(fallback_endpoint, force_download=True)
            fallback_attempted = True
    except (HfHubHTTPError, LocalEntryNotFoundError, RepositoryNotFoundError) as exc:  # pragma: no cover - network dependent
        if not fallback_attempted and mirror_endpoint and mirror_endpoint != fallback_endpoint:
            _perform_download(fallback_endpoint, force_download=True)
            fallback_attempted = True
        else:
            raise RuntimeError(
                f"无法从 Hugging Face 下载模型 `{model_id}`。请检查网络连接、镜像设置或访问令牌后重试。"
            ) from exc

    if not _has_model_weights(target_dir):
        raise RuntimeError(
            f"下载 `{model_id}` 时未找到权重文件，请确认模型仓库可访问并具有足够权限。"
        )
    return target_dir


def ensure_base_model(
    *,
    model_id: str = DEFAULT_BASE_MODEL_ID,
    preferred_path: Optional[str] = None,
    force: bool = False,
) -> Path:
    target_dir = _resolve_target_dir(preferred_path, BASE_MODELS_DIR, model_id)
    return _download_if_missing(model_id, target_dir=target_dir, force=force)


def ensure_text2vec_model(
    *,
    model_id: str = DEFAULT_TEXT2VEC_MODEL_ID,
    preferred_path: Optional[str] = None,
    force: bool = False,
) -> Path:
    target_dir = _resolve_target_dir(preferred_path, TEXT2VEC_MODELS_DIR, model_id)
    return _download_if_missing(model_id, target_dir=target_dir, force=force)


__all__ = [
    "DEFAULT_BASE_MODEL_ID",
    "DEFAULT_TEXT2VEC_MODEL_ID",
    "ensure_base_model",
    "ensure_text2vec_model",
]
