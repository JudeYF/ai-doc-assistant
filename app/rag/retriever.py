"""向量数据库管理模块"""
import os
import shutil
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

# 路径配置
BASE_DIR = Path(__file__).parent.parent.parent
PDF_PATH = BASE_DIR / "data" / "中华人民共和国治安管理处罚法.pdf"
DB_PATH = BASE_DIR / "data" / "chroma_db"

# 全局变量
_db = None
_bm25_retriever = None
_ensemble_retriever = None

def get_embedding():
    """获取嵌入模型（Ollama，模型名和地址来自环境变量）"""
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3:latest"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )

def init_database():
    """初始化或加载向量数据库"""
    global _db
    
    if _db is not None:
        return _db
    
    embedding = get_embedding()
    
    if DB_PATH.exists():
        print("📂 加载已有数据库...")
        _db = Chroma(
            persist_directory=str(DB_PATH),
            embedding_function=embedding,
            collection_name="china_public_security_law"
        )
    else:
        print("📝 初始化新数据库...")
        from app.rag.services import process_legal_document
        
        chunks = process_legal_document(str(PDF_PATH))
        DB_PATH.mkdir(parents=True, exist_ok=True)
        
        _db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            collection_name="china_public_security_law",
            persist_directory=str(DB_PATH)
        )
        print("✅ 数据库初始化完成")
    
    return _db

def init_retrievers():
    """
    初始化所有检索器（只执行一次，使用缓存）

    Returns:
        混合检索器
    """
    global _db, _bm25_retriever, _ensemble_retriever

    # 如果已经初始化，直接返回
    if _ensemble_retriever is not None:
        return _ensemble_retriever

    # 1. 初始化向量数据库
    _db = init_database()

    # 2. 创建向量检索器
    vector_retriever = _db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    # 3. 创建 BM25 检索器
    from app.rag.services import process_legal_document
    chunks = process_legal_document(str(PDF_PATH))
    _bm25_retriever = BM25Retriever.from_documents(chunks, k=6)

    # 4. 创建混合检索器
    _ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, _bm25_retriever],
        weights=[0.6, 0.4]
    )

    print("✅ 检索器初始化完成（已缓存）混合检索：向量检索 + BM25")
    return _ensemble_retriever

def get_retrieval_result(query: str, k: int = 6):
    """
    混合检索：向量检索 + BM25（使用缓存的检索器）

    Args:
        query: 查询文本
        k: 返回结果数量

    Returns:
        检索到的文档列表
    """
    ensemble_retriever = init_retrievers()
    return ensemble_retriever.invoke(query)

if __name__ == "__main__":
    # 测试
    print("测试数据库初始化...")
    db = init_database()
    print(f"数据库大小: {db._collection.count()} 条记录")
    
    print("\n测试检索...")
    results = get_retrieval_result("阻碍执行紧急任务的消防车")
    for i, doc in enumerate(results, 1):
        print(f"\n--- 结果 {i} ---")
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
