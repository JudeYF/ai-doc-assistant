"""嵌入模型工厂：本机 Ollama 或 OpenAI 兼容接口。

init_db 和运行时检索必须走同一个工厂，避免向量空间不一致。
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings

load_dotenv()

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "bge-m3:latest"
DEFAULT_API_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_API_MODEL = "BAAI/bge-m3"
DEFAULT_API_BATCH_SIZE = 32


def get_embedding_provider() -> str:
    """返回规范化后的提供方名称：ollama 或 api。未设置时默认为 ollama。"""
    raw = os.getenv("EMBEDDING_PROVIDER", "").strip().lower()
    if raw in ("", "ollama"):
        return "ollama"
    if raw == "api":
        return "api"
    raise RuntimeError(
        f"不支持的 EMBEDDING_PROVIDER={raw!r}，请使用 ollama 或 api。"
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    return int(raw)


def get_embedding() -> Embeddings:
    """按环境变量创建嵌入模型，供建库和检索共用。"""
    provider = get_embedding_provider()
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_MODEL),
            base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        )

    from langchain_openai import OpenAIEmbeddings

    api_key = os.getenv("EMBEDDING_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "EMBEDDING_PROVIDER=api 时需要设置 EMBEDDING_API_KEY。"
            "请复制 .env.example 为 .env 并填入密钥。"
        )

    # SiliconFlow 的 /embeddings 单次最多 32 条；默认按这个限制分包。
    try:
        batch_size = _env_int("EMBEDDING_BATCH_SIZE", DEFAULT_API_BATCH_SIZE)
    except ValueError as exc:
        raise RuntimeError("EMBEDDING_BATCH_SIZE 必须是正整数。") from exc
    if batch_size < 1:
        raise RuntimeError("EMBEDDING_BATCH_SIZE 必须是正整数。")

    base_url = os.getenv("EMBEDDING_BASE_URL", DEFAULT_API_BASE_URL).strip() or DEFAULT_API_BASE_URL
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_API_MODEL).strip() or DEFAULT_API_MODEL

    return OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
        chunk_size=batch_size,
        # 关闭按 OpenAI/tiktoken 规则截断，避免和非 OpenAI 接口不一致。
        check_embedding_ctx_length=False,
    )


def describe_embedding() -> str:
    """用于日志的简短描述，不包含密钥。"""
    provider = get_embedding_provider()
    if provider == "ollama":
        model = os.getenv("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_MODEL)
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
        return f"ollama {model} @ {base_url}"
    model = os.getenv("EMBEDDING_MODEL", DEFAULT_API_MODEL).strip() or DEFAULT_API_MODEL
    base_url = os.getenv("EMBEDDING_BASE_URL", DEFAULT_API_BASE_URL).strip() or DEFAULT_API_BASE_URL
    batch_size = os.getenv("EMBEDDING_BATCH_SIZE", str(DEFAULT_API_BATCH_SIZE)).strip() or str(DEFAULT_API_BATCH_SIZE)
    return f"api {model} @ {base_url} (batch_size={batch_size})"
