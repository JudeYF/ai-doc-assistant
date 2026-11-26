"""嵌入工厂的选择与默认行为。"""

from __future__ import annotations

import os
import unittest

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.rag.embeddings import describe_embedding, get_embedding, get_embedding_provider


class EmbeddingFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {
            key: os.environ.get(key)
            for key in (
                "EMBEDDING_PROVIDER",
                "EMBEDDING_API_KEY",
                "EMBEDDING_BASE_URL",
                "EMBEDDING_MODEL",
                "EMBEDDING_BATCH_SIZE",
                "OLLAMA_BASE_URL",
                "OLLAMA_EMBEDDING_MODEL",
            )
        }
        for key in self._saved:
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_default_provider_is_ollama(self) -> None:
        self.assertEqual(get_embedding_provider(), "ollama")
        embedding = get_embedding()
        self.assertIsInstance(embedding, OllamaEmbeddings)
        self.assertEqual(embedding.model, "bge-m3:latest")
        self.assertEqual(embedding.base_url, "http://127.0.0.1:11434")

    def test_ollama_reads_env(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "OLLAMA"
        os.environ["OLLAMA_EMBEDDING_MODEL"] = "bge-m3:latest"
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
        embedding = get_embedding()
        self.assertIsInstance(embedding, OllamaEmbeddings)
        self.assertIn("ollama bge-m3:latest", describe_embedding())

    def test_api_provider_uses_openai_compatible_client(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "api"
        os.environ["EMBEDDING_API_KEY"] = "sk-test-not-real"
        os.environ["EMBEDDING_BASE_URL"] = "https://api.siliconflow.cn/v1"
        os.environ["EMBEDDING_MODEL"] = "BAAI/bge-m3"
        os.environ["EMBEDDING_BATCH_SIZE"] = "16"
        embedding = get_embedding()
        self.assertIsInstance(embedding, OpenAIEmbeddings)
        self.assertEqual(embedding.model, "BAAI/bge-m3")
        self.assertEqual(embedding.openai_api_base, "https://api.siliconflow.cn/v1")
        self.assertEqual(embedding.chunk_size, 16)
        self.assertFalse(embedding.check_embedding_ctx_length)
        self.assertIn("api BAAI/bge-m3", describe_embedding())

    def test_api_defaults_target_siliconflow_bge_m3_batch_32(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "api"
        os.environ["EMBEDDING_API_KEY"] = "sk-test-not-real"
        embedding = get_embedding()
        self.assertEqual(embedding.model, "BAAI/bge-m3")
        self.assertEqual(embedding.openai_api_base, "https://api.siliconflow.cn/v1")
        self.assertEqual(embedding.chunk_size, 32)

    def test_api_mode_requires_key(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "api"
        with self.assertRaisesRegex(RuntimeError, "EMBEDDING_API_KEY"):
            get_embedding()

    def test_invalid_provider(self) -> None:
        os.environ["EMBEDDING_PROVIDER"] = "foo"
        with self.assertRaisesRegex(RuntimeError, "不支持的 EMBEDDING_PROVIDER"):
            get_embedding()

    def test_retriever_and_init_db_share_factory(self) -> None:
        import app.rag.retriever as retriever
        import scripts.init_db as init_db  # noqa: F401 — 确认脚本可导入

        self.assertIs(retriever.get_embedding, get_embedding)

    def test_ollama_embed_query_smoke(self) -> None:
        embedding = get_embedding()
        try:
            vector = embedding.embed_query("阻碍执行紧急任务的消防车")
        except Exception as exc:
            raise unittest.SkipTest(f"Ollama 不可用: {exc}") from exc
        self.assertEqual(len(vector), 1024)


if __name__ == "__main__":
    unittest.main()
