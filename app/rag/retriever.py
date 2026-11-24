"""向量数据库管理模块"""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# 路径配置
BASE_DIR = Path(__file__).parent.parent.parent
PDF_PATH = BASE_DIR / "data" / "中华人民共和国治安管理处罚法.pdf"
# 新索引单独放目录，旧的 data/chroma_db 原样保留，便于回退
DB_PATH = BASE_DIR / "data" / "chroma_db_v2"
LEGACY_DB_PATH = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "china_public_security_law_v2"

# 全局变量
_db = None
_vector_retriever = None
_bm25_retriever = None
_ensemble_retriever = None


def get_embedding():
    """获取嵌入模型（Ollama，模型名和地址来自环境变量）"""
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )


def init_database():
    """加载按条切分后的向量库；目录不存在或 collection 为空时才新建。"""
    global _db

    if _db is not None:
        return _db

    embedding = get_embedding()

    if DB_PATH.exists():
        print("📂 加载已有数据库...")
        _db = Chroma(
            persist_directory=str(DB_PATH),
            embedding_function=embedding,
            collection_name=COLLECTION_NAME,
        )
        if _db._collection.count() > 0:
            return _db
        print("⚠️ 新 collection 为空，将重新写入索引...")

    print("📝 初始化新数据库...")
    from app.rag.services import process_legal_document

    chunks = process_legal_document(str(PDF_PATH))
    DB_PATH.mkdir(parents=True, exist_ok=True)

    if _db is not None and _db._collection.count() == 0:
        _db.add_documents(chunks)
    else:
        _db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name=COLLECTION_NAME,
            persist_directory=str(DB_PATH),
        )
    print("✅ 数据库初始化完成")
    return _db


def _apply_k(k: int) -> None:
    """把 k 写回已经缓存的子检索器。"""
    if _bm25_retriever is not None:
        _bm25_retriever.k = k
    if _vector_retriever is not None:
        _vector_retriever.search_kwargs["k"] = k


def init_retrievers(k: int = 6):
    """
    初始化所有检索器（只执行一次，使用缓存）

    Returns:
        混合检索器
    """
    global _db, _vector_retriever, _bm25_retriever, _ensemble_retriever

    if _ensemble_retriever is not None:
        _apply_k(k)
        return _ensemble_retriever

    _db = init_database()

    _vector_retriever = _db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    from app.rag.services import process_legal_document

    chunks = process_legal_document(str(PDF_PATH))
    _bm25_retriever = BM25Retriever.from_documents(chunks, k=k)

    _ensemble_retriever = EnsembleRetriever(
        retrievers=[_vector_retriever, _bm25_retriever],
        weights=[0.6, 0.4],
    )

    print("✅ 检索器初始化完成（已缓存）混合检索：向量检索 + BM25")
    return _ensemble_retriever


def get_retrieval_result(query: str, k: int = 6):
    """
    混合检索：向量检索 + BM25（使用缓存的检索器）

    Args:
        query: 查询文本
        k: 每个子检索器返回的条数

    Returns:
        检索到的文档列表
    """
    ensemble_retriever = init_retrievers(k=k)
    _apply_k(k)
    return ensemble_retriever.invoke(query)


if __name__ == "__main__":
    print("测试数据库初始化...")
    db = init_database()
    print(f"数据库大小: {db._collection.count()} 条记录")

    print("\n测试检索...")
    results = get_retrieval_result("阻碍执行紧急任务的消防车", k=6)
    for i, doc in enumerate(results, 1):
        print(f"\n--- 结果 {i} metadata={doc.metadata} ---")
        text = doc.page_content
        print(text[:200] + "..." if len(text) > 200 else text)
