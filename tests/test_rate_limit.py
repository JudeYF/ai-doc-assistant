"""滑动窗口限流与 /chat 的 429 行为。"""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.chat_client import RateLimitExceeded, _ask_via_api, _detail_from_http_body
from app.main import app
from app.rate_limit import (
    SlidingWindowRateLimiter,
    get_client_ip,
    get_rate_limit_per_minute,
    limiter,
    rate_limit_message,
)


class SlidingWindowTests(unittest.TestCase):
    def test_allows_up_to_limit_then_blocks(self) -> None:
        window = SlidingWindowRateLimiter()
        for _ in range(3):
            allowed, _retry = window.allow("1.2.3.4", 3)
            self.assertTrue(allowed)
        allowed, retry_after = window.allow("1.2.3.4", 3)
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry_after, 1)

    def test_zero_limit_disables(self) -> None:
        window = SlidingWindowRateLimiter()
        for _ in range(20):
            allowed, retry_after = window.allow("1.2.3.4", 0)
            self.assertTrue(allowed)
            self.assertEqual(retry_after, 0)

    def test_separate_ips_are_independent(self) -> None:
        window = SlidingWindowRateLimiter()
        self.assertTrue(window.allow("10.0.0.1", 1)[0])
        self.assertFalse(window.allow("10.0.0.1", 1)[0])
        self.assertTrue(window.allow("10.0.0.2", 1)[0])


class ClientIpTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.get("TRUST_FORWARDED_FOR")
        os.environ.pop("TRUST_FORWARDED_FOR", None)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop("TRUST_FORWARDED_FOR", None)
        else:
            os.environ["TRUST_FORWARDED_FOR"] = self._saved

    def test_defaults_to_peer_host(self) -> None:
        request = SimpleNamespace(
            headers={"x-forwarded-for": "9.9.9.9, 8.8.8.8"},
            client=SimpleNamespace(host="10.0.0.5"),
        )
        self.assertEqual(get_client_ip(request), "10.0.0.5")

    def test_uses_x_forwarded_for_when_trusted(self) -> None:
        os.environ["TRUST_FORWARDED_FOR"] = "true"
        request = SimpleNamespace(
            headers={"x-forwarded-for": "9.9.9.9, 8.8.8.8"},
            client=SimpleNamespace(host="10.0.0.5"),
        )
        self.assertEqual(get_client_ip(request), "9.9.9.9")


class ChatRateLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            key: os.environ.get(key)
            for key in ("RATE_LIMIT_PER_MINUTE", "TRUST_FORWARDED_FOR", "LLM_API_KEY")
        }
        os.environ["RATE_LIMIT_PER_MINUTE"] = "2"
        os.environ.pop("TRUST_FORWARDED_FOR", None)
        limiter.reset()
        self.ask_patch = patch("app.main.ask", return_value="ok")
        self.ask_patch.start()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.ask_patch.stop()
        limiter.reset()
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_health_is_not_rate_limited(self) -> None:
        for _ in range(5):
            response = self.client.get("/health")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertIn(body["status"], {"healthy", "warning"})
            self.assertIn("embedding_provider", body)
            self.assertIn("llm_configured", body)

    def test_chat_returns_429_after_limit(self) -> None:
        payload = {"question": "消防车怎么处罚？", "session_id": "test"}
        for _ in range(2):
            response = self.client.post("/chat", json=payload)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json(), {"answer": "ok"})

        response = self.client.post("/chat", json=payload)
        self.assertEqual(response.status_code, 429)
        body = response.json()
        self.assertEqual(body["detail"], rate_limit_message(2))
        self.assertIn("请求过于频繁", body["detail"])
        self.assertIn("Retry-After", response.headers)
        self.assertEqual(get_rate_limit_per_minute(), 2)

    def test_disabled_limit_allows_many_requests(self) -> None:
        os.environ["RATE_LIMIT_PER_MINUTE"] = "0"
        limiter.reset()
        payload = {"question": "消防车怎么处罚？", "session_id": "test"}
        for _ in range(8):
            response = self.client.post("/chat", json=payload)
            self.assertEqual(response.status_code, 200)

    def test_trusted_forwarded_for_shares_a_bucket(self) -> None:
        os.environ["TRUST_FORWARDED_FOR"] = "true"
        os.environ["RATE_LIMIT_PER_MINUTE"] = "1"
        limiter.reset()
        payload = {"question": "消防车怎么处罚？", "session_id": "test"}
        first = self.client.post(
            "/chat", json=payload, headers={"X-Forwarded-For": "203.0.113.10"}
        )
        self.assertEqual(first.status_code, 200)
        second = self.client.post(
            "/chat", json=payload, headers={"X-Forwarded-For": "203.0.113.10, 10.0.0.1"}
        )
        self.assertEqual(second.status_code, 429)


class ChatClientTests(unittest.TestCase):
    def test_parse_detail(self) -> None:
        self.assertEqual(
            _detail_from_http_body(json.dumps({"detail": "请求过于频繁，请稍后再试。当前限制为每分钟 10 次。"})),
            "请求过于频繁，请稍后再试。当前限制为每分钟 10 次。",
        )
        self.assertIsNone(_detail_from_http_body("not-json"))

    def test_ask_via_api_raises_rate_limit_exceeded(self) -> None:
        import io
        import urllib.error

        error = urllib.error.HTTPError(
            url="http://example/chat",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=io.BytesIO(json.dumps({"detail": "请求过于频繁，请稍后再试。当前限制为每分钟 10 次。"}).encode()),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(RateLimitExceeded) as ctx:
                _ask_via_api("http://example", "hello", "s1")
        self.assertEqual(str(ctx.exception), "请求过于频繁，请稍后再试。当前限制为每分钟 10 次。")


if __name__ == "__main__":
    unittest.main()
