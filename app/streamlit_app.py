"""Streamlit 应用 - 法律问答系统"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from app.rag.chains import ask
from app.rag.retriever import get_retrieval_result

# ==========================
# 页面配置
# ==========================
st.set_page_config(
    page_title="法律问答系统",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# 自定义 CSS 样式
# ==========================
st.markdown("""
<style>
    h1, h2, h3 {
        color: #2563eb;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================
# 初始化会话状态
# ==========================
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'session_id' not in st.session_state:
    st.session_state.session_id = "default"
if 'pending_user_input' not in st.session_state:
    st.session_state.pending_user_input = None

# ==========================
# 侧边栏
# ==========================
with st.sidebar:
    st.title("⚖️ 法律问答助手")
    st.markdown("---")

    st.markdown("### 📚 关于本系统")
    st.info("""
    基于《中华人民共和国治安管理处罚法》的智能问答系统

    **技术特点：**
    - RAG 检索增强生成
    - 向量检索 + BM25 混合检索
    - 支持多轮对话
    """)

    st.markdown("---")

    # 会话管理
    st.markdown("### 💬 会话控制")

    # 新建会话
    if st.button("🆕 新建会话"):
        st.session_state.messages = []
        st.session_state.session_id = f"session_{len(st.session_state.messages)}"
        st.session_state.pending_user_input = None
        st.rerun()

    # 清空当前会话
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.session_state.pending_user_input = None
        st.rerun()

    st.markdown("---")

    # 显示示例问题
    st.markdown("### 💡 示例问题")
    example_questions = [
        "治安管理处罚的具体种类有哪些？",
        "阻碍执行紧急任务的消防车会如何处罚？",
        "对于未成年人违反治安管理，法律在处罚上有什么特殊规定？",
        "殴打他人会受到什么处罚？",
        "盗窃公私财物如何处罚？",
        "在公共场所故意裸露身体如何处罚？"
    ]

    for q in example_questions:
        if st.button(q, key=f"example_{q}", use_container_width=True):
            st.session_state.pending_user_input = q
            st.rerun()

# ==========================
# 主界面
# ==========================
st.title("⚖️ 法律问答系统")
st.markdown("基于《中华人民共和国治安管理处罚法》的智能问答助手")

# ==========================
# 显示对话历史
# ==========================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            # 显示检索来源
            if "sources" in msg and msg["sources"]:
                with st.expander("📄 查看相关条文"):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**来源 {i}:**")
                        st.text(source[:300] + "..." if len(source) > 300 else source)

# ==========================
# 处理待处理的输入（示例问题）- 在历史显示之后处理
# ==========================
if st.session_state.pending_user_input:
    prompt = st.session_state.pending_user_input

    # 立即显示用户消息
    with st.chat_message("user"):
        st.write(prompt)

    # 获取检索结果
    with st.spinner("正在检索相关法律条文..."):
        retrieval_results = get_retrieval_result(prompt)
        sources = [doc.page_content for doc in retrieval_results]

    # 生成 AI 回答
    with st.spinner("正在生成回答..."):
        try:
            response = ask(prompt, st.session_state.session_id)
            with st.chat_message("assistant"):
                st.write(response)
                # 显示检索来源
                with st.expander("📄 查看相关条文"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**来源 {i}:**")
                        st.text(source[:300] + "..." if len(source) > 300 else source)

            # 保存到历史
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
        except Exception as e:
            error_msg = f"处理出错: {str(e)}"
            with st.chat_message("assistant"):
                st.error(error_msg)
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # 清除待处理输入并刷新
    st.session_state.pending_user_input = None
    st.rerun()

# ==========================
# 用户输入
# ==========================
if prompt := st.chat_input("请输入您的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 获取检索结果（用于显示来源）
    with st.spinner("正在检索相关法律条文..."):
        retrieval_results = get_retrieval_result(prompt)
        sources = [doc.page_content for doc in retrieval_results]

    # 生成 AI 回答
    with st.spinner("正在生成回答..."):
        try:
            response = ask(prompt, st.session_state.session_id)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
            with st.chat_message("assistant"):
                st.write(response)
                # 显示检索来源
                with st.expander("📄 查看相关条文"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**来源 {i}:**")
                        st.text(source[:300] + "..." if len(source) > 300 else source)
        except Exception as e:
            error_msg = f"处理出错: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)

# ==========================
# 页脚
# ==========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    ⚖️ 法律问答系统 | 基于 LangChain & RAG | 仅供参考，不作为法律依据
</div>
""", unsafe_allow_html=True)
