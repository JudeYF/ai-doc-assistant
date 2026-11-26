"""MCP 服务：工具注册、条文号解析，以及 stdio 端到端调用。"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document

from app.mcp_server import (
    get_article,
    lookup_article,
    mcp,
    normalize_article_number,
    search_law_results,
)
from app.rag.retriever import DB_PATH, PDF_PATH
from app.rag.services import cn_to_int, int_to_cn

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class ArticleNumberTests(unittest.TestCase):
    def test_normalize_arabic_and_chinese(self) -> None:
        cases = {
            "50": "第五十条",
            "第50条": "第五十条",
            "第 50 条": "第五十条",
            "050": "第五十条",
            "第五十条": "第五十条",
            " 第五十条 ": "第五十条",
            "第五十": "第五十条",
            "1": "第一条",
            "10": "第十条",
            "11": "第十一条",
            "101": "第一百零一条",
            "110": "第一百一十条",
            "119": "第一百一十九条",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(normalize_article_number(raw), expected)

    def test_int_to_cn_roundtrip_1_to_119(self) -> None:
        for number in range(1, 120):
            label = f"第{int_to_cn(number)}条"
            self.assertEqual(cn_to_int(label), number, label)
            self.assertEqual(normalize_article_number(str(number)), label)

    def test_invalid_article_number(self) -> None:
        with self.assertRaisesRegex(ValueError, "条文号不能为空"):
            normalize_article_number("  ")
        with self.assertRaisesRegex(ValueError, "无法识别"):
            normalize_article_number("第五十条之二")


class ToolRegistrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_lists_expected_tools_and_schemas(self) -> None:
        tools = {tool.name: tool for tool in await mcp.list_tools()}
        self.assertEqual(set(tools), {"search_law", "get_article"})

        search = tools["search_law"]
        self.assertIn("检索", search.description or "")
        search_props = search.inputSchema["properties"]
        self.assertEqual(search.inputSchema["required"], ["query"])
        self.assertEqual(search_props["query"]["type"], "string")
        self.assertEqual(search_props["top_k"]["type"], "integer")
        self.assertEqual(search_props["top_k"]["default"], 5)
        self.assertEqual(search_props["top_k"]["minimum"], 1)
        self.assertEqual(search_props["top_k"]["maximum"], 20)

        article = tools["get_article"]
        self.assertIn("条号", article.description or "")
        article_props = article.inputSchema["properties"]
        self.assertEqual(article.inputSchema["required"], ["article_number"])
        self.assertEqual(article_props["article_number"]["type"], "string")


class ToolLogicTests(unittest.TestCase):
    def test_search_law_formats_mocked_documents(self) -> None:
        docs = [
            Document(
                page_content="第五十条 阻碍执行紧急任务的消防车……",
                metadata={
                    "article": "第五十条",
                    "chapter": "第三章 违反治安管理的行为和处罚",
                    "chunk_id": "art-050",
                    "source": "中华人民共和国治安管理处罚法.pdf",
                },
            ),
            Document(
                page_content="第五十二条 ……",
                metadata={
                    "article": "第五十二条",
                    "chapter": "第三章 违反治安管理的行为和处罚",
                    "chunk_id": "art-052",
                    "source": "中华人民共和国治安管理处罚法.pdf",
                },
            ),
        ]
        with (
            patch("app.mcp_server._index_status", return_value=(True, "")),
            patch("app.mcp_server.get_retrieval_result", return_value=docs) as retriever,
            patch("app.mcp_server._score_map", return_value={"art-050": 0.91234}),
        ):
            payload = search_law_results("阻碍执行紧急任务的消防车", top_k=2)

        retriever.assert_called_once_with("阻碍执行紧急任务的消防车", k=2)
        self.assertEqual(payload["query"], "阻碍执行紧急任务的消防车")
        self.assertEqual(len(payload["results"]), 2)
        first = payload["results"][0]
        self.assertEqual(first["rank"], 1)
        self.assertEqual(first["article"], "第五十条")
        self.assertEqual(first["chapter"], "第三章 违反治安管理的行为和处罚")
        self.assertIn("消防车", first["content"])
        self.assertEqual(first["score"], 0.9123)
        self.assertNotIn("score", payload["results"][1])

    def test_search_law_rejects_empty_query(self) -> None:
        payload = search_law_results("   ", top_k=5)
        self.assertEqual(payload["results"], [])
        self.assertIn("查询不能为空", payload["error"])

    def test_get_article_from_corpus(self) -> None:
        self.assertTrue(PDF_PATH.exists() or DB_PATH.exists())
        payload = get_article("50")
        self.assertTrue(payload["found"], payload)
        self.assertEqual(payload["article"], "第五十条")
        self.assertIn("消防车", payload["content"])
        self.assertIn("第三章", payload["chapter"])
        self.assertEqual(payload["chunk_id"], "art-050")

        same = lookup_article("第五十条")
        self.assertEqual(same["content"], payload["content"])

    def test_get_article_not_found(self) -> None:
        payload = get_article("999")
        self.assertFalse(payload["found"])
        self.assertEqual(payload["article"], "第九百九十九条")
        self.assertIn("没有找到", payload["error"])

    def test_get_article_invalid(self) -> None:
        payload = get_article("不是条文")
        self.assertFalse(payload["found"])
        self.assertIsNone(payload["article"])


def _stdio_env() -> dict[str, str]:
    keys = (
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "LANG",
        "LC_ALL",
        "EMBEDDING_PROVIDER",
        "OLLAMA_BASE_URL",
        "OLLAMA_EMBEDDING_MODEL",
        "EMBEDDING_API_KEY",
        "EMBEDDING_BASE_URL",
        "EMBEDDING_MODEL",
        "EMBEDDING_BATCH_SIZE",
        "VIRTUAL_ENV",
    )
    env = {key: os.environ[key] for key in keys if os.environ.get(key)}
    env["PYTHONUNBUFFERED"] = "1"
    env["ANONYMIZED_TELEMETRY"] = "False"
    return env


def _parse_tool_result(result) -> dict:
    if result.isError:
        raise AssertionError(f"tools/call returned an error: {result}")
    if result.structuredContent:
        return result.structuredContent
    texts = []
    for item in result.content:
        text = getattr(item, "text", None)
        if text:
            texts.append(text)
    if not texts:
        raise AssertionError(f"empty tool result: {result}")
    return json.loads(texts[0])


def _ollama_available() -> bool:
    try:
        import urllib.request

        base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        with urllib.request.urlopen(base + "/api/tags", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


class StdioIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_initialize_list_tools_and_get_article(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=PYTHON,
            args=["-m", "app.mcp_server"],
            cwd=str(REPO_ROOT),
            env=_stdio_env(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                self.assertEqual(init.serverInfo.name, "ai-doc-assistant")
                tools = await session.list_tools()
                names = {tool.name for tool in tools.tools}
                self.assertEqual(names, {"search_law", "get_article"})
                result = await session.call_tool(
                    "get_article", {"article_number": "第五十条"}
                )
                payload = _parse_tool_result(result)

        self.assertTrue(payload["found"], payload)
        self.assertEqual(payload["article"], "第五十条")
        self.assertIn("消防车", payload["content"])

    async def test_stdio_search_law_with_local_index(self) -> None:
        if not DB_PATH.exists():
            self.skipTest(f"本地没有向量索引: {DB_PATH}")
        if not _ollama_available():
            self.skipTest("Ollama 不可用，跳过需要嵌入的检索测试")

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=PYTHON,
            args=["-m", "app.mcp_server"],
            cwd=str(REPO_ROOT),
            env=_stdio_env(),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await asyncio.wait_for(
                    session.call_tool(
                        "search_law",
                        {
                            "query": "阻碍执行紧急任务的消防车",
                            "top_k": 5,
                        },
                    ),
                    timeout=120,
                )
                payload = _parse_tool_result(result)

        self.assertNotIn("error", payload, payload)
        self.assertGreaterEqual(len(payload["results"]), 1, payload)
        articles = [item.get("article") for item in payload["results"]]
        self.assertIn("第五十条", articles, payload)
        top = payload["results"][0]
        self.assertIn("content", top)
        # 消防车这条应当排在很前面；不强制必须是第一，但内容里应能对上。
        self.assertTrue(
            any("消防车" in (item.get("content") or "") for item in payload["results"]),
            payload,
        )


if __name__ == "__main__":
    unittest.main()
