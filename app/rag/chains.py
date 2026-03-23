"""LangChain 链式调用模块"""
from langchain_openai import ChatOpenAI
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_classic.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from app.rag.retriever import get_retrieval_result
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 对话历史数据库路径
HISTORY_DB_PATH = Path(__file__).parent.parent.parent / "data" / "chat_history.db"

# ==========================
# 提示词模板
# ==========================
prompt = ChatPromptTemplate.from_template(
    """
你是一个专业的法律助手，专门回答关于中国治安管理处罚法的问题。
请基于以下提供的法律条文内容回答用户的问题。

回答要求：
1. 严格严格基于提供的法律条文内容回答，不要编造信息
2. 如果提供的材料中没有相关内容，请明确说明
3. 回答时要准确引用相关条文
4. 使用清晰、专业的法律语言
5. 保持客观中立的立场

以下是相关的法律条文内容：
{context}

以下是你们聊过的历史对话记录：
{history}

用户的问题：
{question}
"""
)

# ==========================
# LLM 配置
# ==========================
llm = ChatNVIDIA(
    model="openai/gpt-oss-120b",
    # model="z-ai/glm5",
    api_key=os.getenv('NVIDIA_API_KEY'),
    temperature=0.6,
)
llm = ChatOpenAI(
    temperature=0.6,
    model="glm-5",
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# ==========================
# 对话历史管理（SQLite 持久化）
# ==========================
def get_chat_history(session_id: str):
    """
    获取会话历史（持久化到 SQLite）

    Args:
        session_id: 会话ID

    Returns:
        SQLChatMessageHistory 实例
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{HISTORY_DB_PATH}"
    )

# ==========================
# RAG 链构建
# ==========================
chain = (
    {
        'context': RunnableLambda(lambda x: get_retrieval_result(x['question'])),
        'question': RunnablePassthrough(),
        'history': RunnableLambda(lambda x: get_chat_history(x.get('session_id', 'default')))
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ==========================
# 带历史记录的链
# ==========================
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key='question',
    history_messages_key='history',
)

# ==========================
# 主函数
# ==========================
def ask(question: str, session_id: str = "default") -> str:
    """
    提问并获取答案
    
    Args:
        question: 用户问题
        session_id: 会话ID，用于多轮对话
    
    Returns:
        AI 回答
    """
    result = chain_with_history.invoke(
        {'question': question, 'session_id': session_id},
        config={'configurable': {'session_id': session_id}}
    )
    return result

if __name__ == "__main__":
    # 测试
    print("测试 RAG 链...")
    result = ask(
        question='阻碍执行紧急任务的消防车、救护车。会如何处罚。',
        session_id='test_session'
    )
    print(f"\n问题：阻碍执行紧急任务的消防车、救护车。会如何处罚。")
    print(f"\n回答：\n{result}")
