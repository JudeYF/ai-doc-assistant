"""把法律知识库检索暴露为本地 MCP 服务（stdio）。

stdout 只用于 JSON-RPC，日志和第三方库的输出一律转到 stderr，
避免污染 Cursor / Claude Desktop 的通信通道。
"""
from __future__ import annotations

import logging
import os
import re
import sys
import warnings
from contextlib import contextmanager, redirect_stdout
from typing import Annotated, Any

# 在导入 chromadb / langchain 之前关掉它们往 stdout 打的遥测和日志。
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.basicConfig(
    stream=sys.stderr,
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
    force=True,
)
warnings.simplefilter("default")
warnings.showwarning = (
    lambda message, category, filename, lineno, file=None, line=None: sys.stderr.write(
        warnings.formatwarning(message, category, filename, lineno, line)
    )
)

logger = logging.getLogger(__name__)


@contextmanager
def _stdout_to_stderr():
    """检索和建库路径里有 print()，调用期间把 stdout 重定向到 stderr。"""
    with redirect_stdout(sys.stderr):
        yield


with _stdout_to_stderr():
    from mcp.server.fastmcp import FastMCP
    from pydantic import Field

    from app.rag.retriever import (
        COLLECTION_NAME,
        DB_PATH,
        PDF_PATH,
        get_retrieval_result,
        init_database,
    )
    from app.rag.services import cn_to_int, int_to_cn, process_legal_document

_ARABIC_ARTICLE_RE = re.compile(r"^第?(\d+)条?$")
_CN_ARTICLE_RE = re.compile(r"^第?([一二三四五六七八九十百零]+)条?$")
_MAX_TOP_K = 20

_pdf_chunks_cache: list[Any] | None = None

mcp = FastMCP(
    "ai-doc-assistant",
    instructions=(
        "检索《中华人民共和国治安管理处罚法》条文。"
        "先用 search_law 按问题检索，需要核对原文时再用 get_article。"
        "语料是 2012 年修正版。回答不是法律意见。"
    ),
    log_level="WARNING",
)


def normalize_article_number(raw: str) -> str:
    """把「50」「第50条」「第五十条」统一成「第五十条」。"""
    if raw is None:
        raise ValueError("条文号不能为空。")
    text = re.sub(r"\s+", "", str(raw).strip())
    if not text:
        raise ValueError("条文号不能为空。")

    arabic = _ARABIC_ARTICLE_RE.fullmatch(text)
    if arabic:
        return f"第{int_to_cn(int(arabic.group(1)))}条"

    chinese = _CN_ARTICLE_RE.fullmatch(text)
    if chinese:
        return f"第{int_to_cn(cn_to_int(chinese.group(1)))}条"

    raise ValueError(f"无法识别的条文号: {raw!r}")


def _article_record(content: str, metadata: dict[str, Any], fallback: str) -> dict[str, Any]:
    return {
        "article": metadata.get("article") or fallback,
        "chapter": metadata.get("chapter") or "",
        "content": content or "",
        "chunk_id": metadata.get("chunk_id") or "",
        "source": metadata.get("source") or "",
    }


def _lookup_from_index(label: str) -> dict[str, Any] | None:
    """按 metadata 直接读 Chroma，不走嵌入，因此不依赖 Ollama。"""
    if not DB_PATH.exists():
        return None
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(DB_PATH))
        collection = client.get_collection(COLLECTION_NAME)
        result = collection.get(
            where={"article": label},
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            chunk_id = f"art-{cn_to_int(label):03d}"
            result = collection.get(
                where={"chunk_id": chunk_id},
                include=["documents", "metadatas"],
            )
        if not result["ids"]:
            return None
        return _article_record(
            result["documents"][0] or "",
            result["metadatas"][0] or {},
            label,
        )
    except Exception:
        logger.exception("从向量索引读取 %s 失败，将回退到 PDF。", label)
        return None


def _pdf_chunks() -> list[Any]:
    global _pdf_chunks_cache
    if _pdf_chunks_cache is None:
        if not PDF_PATH.exists():
            raise FileNotFoundError(f"未找到法律 PDF: {PDF_PATH}")
        with _stdout_to_stderr():
            _pdf_chunks_cache = process_legal_document(str(PDF_PATH))
    return _pdf_chunks_cache


def _lookup_from_pdf(label: str) -> dict[str, Any] | None:
    try:
        chunks = _pdf_chunks()
    except Exception:
        logger.exception("从 PDF 读取条文失败。")
        return None
    for chunk in chunks:
        metadata = chunk.metadata or {}
        if metadata.get("article") == label:
            return _article_record(chunk.page_content, metadata, label)
    return None


def lookup_article(article_number: str) -> dict[str, Any]:
    """按条号取原文，优先读索引，没有索引时再解析 PDF。"""
    try:
        label = normalize_article_number(article_number)
    except ValueError as exc:
        return {"found": False, "article": None, "error": str(exc)}

    if DB_PATH.exists():
        record = _lookup_from_index(label)
        if record is not None:
            return {"found": True, **record}
        ready, _message = _index_status()
        if ready:
            return {
                "found": False,
                "article": label,
                "error": f"知识库中没有找到{label}。",
            }

    record = _lookup_from_pdf(label)
    if record is None:
        return {
            "found": False,
            "article": label,
            "error": f"知识库中没有找到{label}。",
        }
    return {"found": True, **record}


def _index_status() -> tuple[bool, str]:
    if not DB_PATH.exists():
        return (
            False,
            "未找到向量索引 data/chroma_db_v2。请先在仓库根目录运行 python scripts/init_db.py。",
        )
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(DB_PATH))
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() <= 0:
            return False, "向量索引是空的。请先运行 python scripts/init_db.py。"
        return True, ""
    except Exception as exc:
        return False, f"无法打开向量索引: {exc}"


def _clamp_top_k(top_k: int) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        return 5
    return max(1, min(value, _MAX_TOP_K))


def _result_key(doc: Any) -> str:
    metadata = getattr(doc, "metadata", None) or {}
    return (
        str(metadata.get("chunk_id") or "")
        or str(metadata.get("article") or "")
        or (getattr(doc, "page_content", "") or "")[:80]
    )


def _score_map(query: str, k: int) -> dict[str, float]:
    """混合检索本身不带分数；若向量检索能给出相关度，再按 chunk 附上。"""
    try:
        with _stdout_to_stderr():
            database = init_database()
            scored = database.similarity_search_with_relevance_scores(
                query, k=max(k * 3, 12)
            )
        mapping: dict[str, float] = {}
        for doc, score in scored:
            if score is None:
                continue
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if value != value:  # NaN
                continue
            mapping[_result_key(doc)] = value
        return mapping
    except Exception:
        logger.exception("附加向量相关度分数失败，将只返回检索结果。")
        return {}


def search_law_results(query: str, top_k: int = 5) -> dict[str, Any]:
    """调用现有混合检索，并整理成结构化结果。"""
    cleaned = (query or "").strip()
    if not cleaned:
        return {"query": query, "results": [], "error": "查询不能为空。"}

    k = _clamp_top_k(top_k)
    ready, message = _index_status()
    if not ready:
        return {"query": cleaned, "results": [], "error": message}

    try:
        with _stdout_to_stderr():
            documents = get_retrieval_result(cleaned, k=k)
        scores = _score_map(cleaned, k)
    except Exception as exc:
        logger.exception("检索失败。")
        return {"query": cleaned, "results": [], "error": f"检索失败: {exc}"}

    results: list[dict[str, Any]] = []
    for index, doc in enumerate(documents[:k], start=1):
        metadata = getattr(doc, "metadata", None) or {}
        item: dict[str, Any] = {
            "rank": index,
            "article": metadata.get("article") or None,
            "chapter": metadata.get("chapter") or None,
            "content": getattr(doc, "page_content", "") or "",
            "chunk_id": metadata.get("chunk_id") or None,
            "source": metadata.get("source") or None,
        }
        score = scores.get(_result_key(doc))
        if score is not None:
            item["score"] = round(score, 4)
        results.append(item)
    return {"query": cleaned, "results": results}


@mcp.tool()
def search_law(
    query: Annotated[str, Field(description="自然语言问题或关键词，例如：阻碍消防车通行如何处罚")],
    top_k: Annotated[int, Field(description="返回条数，默认 5，最大 20", ge=1, le=20)] = 5,
) -> dict[str, Any]:
    """在《治安管理处罚法》知识库中检索相关条文。使用向量检索和 BM25 的混合结果。"""
    return search_law_results(query, top_k)


@mcp.tool()
def get_article(
    article_number: Annotated[
        str,
        Field(description="条文号，支持「第五十条」或「50」这类写法"),
    ],
) -> dict[str, Any]:
    """按条号精确取出一条原文，便于核对检索结果的完整表述。"""
    return lookup_article(article_number)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
