"""文档处理模块"""
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.rag.utils import clean_legal_text

def process_legal_document(file_path: str):
    """
    加载并切分法律文档
    
    Args:
        file_path: PDF 文件路径
    
    Returns:
        切分后的文档列表
    """
    # ==========================
    # 1. 加载文档
    # ==========================
    print("📄 开始加载文档...")
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    
    # ==========================
    # 2. 数据清洗
    # ==========================
    print("🧹 正在清洗文档数据...")
    for doc in docs:
        doc.page_content = clean_legal_text(doc.page_content)
    
    # 将清洗后的多页内容合并为一个完整长文本
    #（有助于跨页条文的完整切分）
    full_text = "\n".join([doc.page_content for doc in docs])
    
    # ==========================
    # 3. 配置针对法律条文的切分器
    # ==========================
    # 核心是使用前瞻断言 (?=...)，这样匹配到的"第X条"不会被当作分隔符删掉，
    # 而是保留在下一个 Chunk 的开头
    legal_separators = [
        r"\n(?=\s*第[一二三四五六七八九十百零]+章\s+)",
        r"\n(?=\s*第[一二三四五六七八九十百零]+节\s+)",
        r"\n(?=\s*第[一二三四五六七八九十百零]+条\s*)",
        "。\n",
        "；\n",
        "。",
        "\n",
        " "
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=legal_separators,
        is_separator_regex=True,  # 必须开启此项以支持正则分隔符
        chunk_size=800,           # 《治安管理处罚法》单条长度极少超过 800 字符
        chunk_overlap=50,         # 少量重叠保证极端情况下的上下文连贯
        length_function=len,
    )

    # ==========================
    # 4. 执行切分
    # ==========================
    print("✂️ 开始执行结构化切分...")
    # 由于之前合并了全文本，这里直接 create_documents
    split_docs = text_splitter.create_documents([full_text])
    
    # 手动将原始文件的 metadata 重新赋给切分后的 docs
    for chunk in split_docs:
        chunk.metadata = {"source": file_path}
    
    print(f"✅ 切分完成，共 {len(split_docs)} 个片段")
    return split_docs

if __name__ == "__main__":
    # 测试
    import sys
    from pathlib import Path
    
    default_pdf = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "中华人民共和国治安管理处罚法.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else str(default_pdf)
    
    if not Path(pdf_path).exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)
    
    chunks = process_legal_document(pdf_path)
    
    print(f"\n前 3 个片段预览：")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- 片段 {i} ---")
        print(chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content)
