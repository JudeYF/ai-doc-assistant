"""统一提问入口：本机直接调用问答链，或把请求转到 FastAPI /chat。"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from app.rate_limit import rate_limit_message


class RateLimitExceeded(Exception):
    """上游 /chat 返回了 HTTP 429。"""


def _detail_from_http_body(body: str) -> str | None:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    detail = data.get("detail")
    if isinstance(detail, str) and detail.strip():
        return detail.strip()
    return None


def _ask_via_api(base_url: str, question: str, session_id: str) -> str:
    url = base_url.rstrip("/") + "/chat"
    payload = json.dumps(
        {"question": question, "session_id": session_id},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = _detail_from_http_body(body)
        if exc.code == 429:
            raise RateLimitExceeded(detail or rate_limit_message()) from exc
        raise RuntimeError(detail or f"接口返回 HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接问答接口 {url}: {exc.reason}") from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError("问答接口返回了无法解析的内容") from exc
    answer = data.get("answer")
    if not isinstance(answer, str):
        raise RuntimeError("问答接口没有返回 answer 字段")
    return answer


def ask_question(question: str, session_id: str = "default") -> str:
    """若设置了 API_BASE_URL，则走 HTTP /chat（可触发 429）；否则直接调用本地问答链。"""
    base_url = os.getenv("API_BASE_URL", "").strip()
    if base_url:
        return _ask_via_api(base_url, question, session_id)
    from app.rag.chains import ask

    return ask(question, session_id)
