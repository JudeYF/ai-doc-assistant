"""按客户端 IP 的滑动窗口限流，不依赖第三方库。"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from threading import Lock

DEFAULT_RATE_LIMIT_PER_MINUTE = 10
WINDOW_SECONDS = 60.0


def get_rate_limit_per_minute() -> int:
    raw = os.getenv("RATE_LIMIT_PER_MINUTE", "").strip()
    if not raw:
        return DEFAULT_RATE_LIMIT_PER_MINUTE
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("RATE_LIMIT_PER_MINUTE 必须是整数，0 表示关闭限流。") from exc
    if value < 0:
        raise RuntimeError("RATE_LIMIT_PER_MINUTE 不能为负数，0 表示关闭限流。")
    return value


def rate_limit_message(limit: int | None = None) -> str:
    if limit is None:
        limit = get_rate_limit_per_minute()
    return f"请求过于频繁，请稍后再试。当前限制为每分钟 {limit} 次。"


def trust_forwarded_for() -> bool:
    return os.getenv("TRUST_FORWARDED_FOR", "").strip().lower() in {"1", "true", "yes", "on"}


def get_client_ip(request) -> str:
    """取客户端 IP。默认用 TCP 对端地址，避免直接暴露时伪造 X-Forwarded-For 绕过限流。

    若前面有反向代理，把 TRUST_FORWARDED_FOR 设为 true，再读取 X-Forwarded-For 的第一个地址。
    """
    if trust_forwarded_for():
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            first = forwarded.split(",")[0].strip()
            if first:
                return first
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client is not None else None
    return host or "unknown"


class SlidingWindowRateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def allow(self, key: str, limit: int) -> tuple[bool, int]:
        """返回 (是否允许, 建议的 Retry-After 秒数)。limit<=0 时不限流。"""
        if limit <= 0:
            return True, 0

        now = time.monotonic()
        with self._lock:
            queue = self._hits[key]
            while queue and now - queue[0] >= WINDOW_SECONDS:
                queue.popleft()
            if not queue:
                self._hits.pop(key, None)
                queue = self._hits[key]
            if len(queue) >= limit:
                retry_after = max(1, int(WINDOW_SECONDS - (now - queue[0])) + 1)
                return False, retry_after
            queue.append(now)
            return True, 0


limiter = SlidingWindowRateLimiter()
