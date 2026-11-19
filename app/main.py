"""FastAPI 主应用入口"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径（支持直接运行）
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

from app.rag.chains import ask

load_dotenv()

# ==========================
# FastAPI 应用
# ==========================
app = FastAPI(
    title="法律问答机器人",
    description="基于 LangChain 和 RAG 的《治安管理处罚法》智能问答系统",
    version="1.0.0"
)

# ==========================
# 数据模型
# ==========================
class ChatRequest(BaseModel):
    """聊天请求"""
    question: str = Field(..., description="用户问题", min_length=1)
    session_id: str = Field(default="default", description="会话ID，用于多轮对话")

class ChatResponse(BaseModel):
    """聊天响应"""
    answer: str = Field(..., description="AI 回答")

# ==========================
# 路由
# ==========================
@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "法律问答机器人",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    llm_api_key = os.getenv("LLM_API_KEY", "").strip()

    if not llm_api_key:
        return {
            "status": "warning",
            "service": "legal-qa-bot",
            "llm_configured": False,
            "message": "LLM_API_KEY 未配置，问答接口无法调用模型",
        }

    return {
        "status": "healthy",
        "service": "legal-qa-bot",
        "llm_configured": True,
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    法律问答接口
    
    - **question**: 用户的问题
    - **session_id**: 会话ID，支持多轮对话（可选）
    """
    try:
        answer = ask(
            question=request.question,
            session_id=request.session_id
        )
        return ChatResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

# ==========================
# 启动脚本
# ==========================
if __name__ == "__main__":
    import uvicorn
    
    print("启动 FastAPI 服务...")
    # reload 需要以导入字符串启动（uvicorn app.main:app --reload）
    uvicorn.run(app, host="0.0.0.0", port=8000)
