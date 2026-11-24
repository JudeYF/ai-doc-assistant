"""数据库初始化脚本

默认写入 data/chroma_db_v2（collection: china_public_security_law_v2）。
旧索引 data/chroma_db 不会被覆盖。

使用方法:
    python scripts/init_db.py
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from app.rag.retriever import COLLECTION_NAME, DB_PATH, LEGACY_DB_PATH, init_database

    print("=" * 50)
    print("🔧 初始化向量数据库")
    print(f"新索引目录: {DB_PATH}")
    print(f"collection: {COLLECTION_NAME}")
    if LEGACY_DB_PATH.exists():
        print(f"旧索引保留在: {LEGACY_DB_PATH}")
    print("=" * 50)

    try:
        db = init_database()
        count = db._collection.count()

        print("\n" + "=" * 50)
        print("✅ 数据库初始化成功")
        print(f"📊 记录数: {count} 条")
        print("=" * 50)

    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
