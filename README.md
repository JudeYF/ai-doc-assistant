# ⚖️ 智能法律问答系统

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)](https://langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 **LangChain + RAG** 的智能法律问答系统，支持多轮对话、检索来源追溯，并提供友好的 Web 交互界面。

## 📋 演示视频

<iframe src="//player.bilibili.com/player.html?bvid=BV1mfXcBjExh&page=1&high_quality=1"
        scrolling="no"
        border="0"
        frameborder="no"
        framespacing="0"
        allowfullscreen="true"
        width="720"
        height="480">
</iframe>

[![视频演示](https://player.bilibili.com/player.html?bvid=BV1mfXcBjExh&as_wide=1&high_quality=1&danmaku=0)](https://player.bilibili.com/player.html?bvid=BV1mfXcBjExh&as_wide=1&high_quality=1&danmaku=0)
   

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🤖 **智能问答** | 基于《中华人民共和国治安管理处罚法》的专业法律问答 |
| 🔍 **混合检索** | 向量检索 + BM25 关键词检索，提升召回准确度 |
| 💬 **多轮对话** | 支持上下文记忆的连续对话，追问自然流畅 |
| 📄 **来源追溯** | 每个回答都标注相关法律条文，可点击查看原文 |
| 🌐 **双接口支持** | FastAPI RESTful API + Streamlit 可视化界面 |
| 🚀 **本地部署**` | 支持 Ollama 本地嵌入模型 + 自选 LLM 服务 |

---

## 🛠️ 技术栈

```
┌─────────────────────────────────────────────────────────────┐
│                        用户交互层                             │
│  ┌──────────────┐              ┌──────────────────────────┐  │
│  │  Streamlit   │              │  FastAPI (REST API)     │  │
│  │  Web 界面     │              │                          │  │
│  └──────────────┘              └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        应用逻辑层                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  LangChain Chains - RAG 问答链、提示词管理             │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  会话管理 (SQLite) - 多轮对话历史存储                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        检索存储层                             │
│  ┌──────────────────┐      ┌────────────────────────────┐  │
│  │  Chroma 向量库    │  +   │  BM25 关键词检索            │  │
│  │  (bge-m3 嵌入)   │      │                            │  │
│  └──────────────────┘      └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        模型服务层                             │
│  ┌──────────────────┐      ┌────────────────────────────┐  │
│  │  Ollama 本地     │      │  NVIDIA AI (或自定义 LLM)   │  │
│  │  bge-m3 嵌入     │      │                            │  │
│  └──────────────────┘      └────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

- **FastAPI** - 现代高性能 Web 框架
- **LangChain** - LLM 应用开发框架
- **Chroma** - 本地向量数据库
- **Ollama** - 本地嵌入模型 (bge-m3)
- **NVIDIA AI** - 大语言模型服务
- **Streamlit** - 交互式 Web 界面
- **SQLite** - 会话历史存储

---

## 🚀 快速开始开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 NVIDIA API Key：

```env
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 3. 启动 Ollama 服务

```bash
# 如果 Ollama 还未安装，访问 https://ollama.ai 下载安装

# 启动 Ollama 服务
ollama serve

# 拉取嵌入模型（新终端）
ollama pull bge-m3:latest
```

### 4. 初始化向量数据库

```bash
python scripts/init_db.py
```

首次运行会自动从 PDF 切分文档并创建向量数据库，之后会直接加载已有数据库。

### 5. 启动服务

#### 方式一：启动 FastAPI 后端

```bash
# 直接运行
python app/main.py

# 或使用 uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后访问：
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

#### 方式二：启动 Streamlit 前端（推荐）

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

访问 Web 界面：http://localhost:8501

---

## 📖 使用指南

### Streamlit Web 界面

推荐使用 Streamlit 前端进行交互，功能包括：

- **聊天对话**：自然语言提问，获得专业法律解答
- **来源追溯**：展开查看回答相关的法律条文原文
- **示例问题**：点击预设问题快速体验
- **会话管理**：新建会话、清空对话
- **多轮对话**：支持追问，保持上下文连贯

### FastAPI 接口

#### 健康检查

```bash
curl http://localhost:8000/health
```

响应：
```json
{
  "status": "healthy",
  "service": "legal-qa-bot",
  "nvidia_configured": true
}
```

#### 发起问答

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "阻碍执行紧急任务的消防车会如何处罚？",
    "session_id": "user123"
  }'
```

响应：
```json
{
  "answer": "根据《治安管理处罚法》第五十条规定，阻碍执行紧急任务的消防车、救护车、工程抢险车、警车等车辆通行的，处警告或者二百元以下罚款..."
}
```

#### 多轮对话

```bash
# 第一轮
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "什么是扰乱公共秩序的行为？",
    "session_id": "user123"
  }'

# 第二轮（保持上下文）
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "这种行为一般会怎么处罚？",
    "session_id": "user123"
  }'
```

---

## 📂 项目结构

```
fastapi-langchain-portfolio/
├── app/
│   ├── main.py              # FastAPI 入口和路由
│   ├── streamlit_app.py     # Streamlit Web 界面
│   └── rag/
│       ├── chains.py        # LangChain 链、提示词、问答逻辑
│       ├── services.py      # 文档加载、切分、清洗
│       ├── retriever.py     # 向量数据库管理、混合检索
│       └── utils.py         # 文本清洗工具函数
├── data/
│   ├── 中华人民共和国治安管理处罚法.pdf
│   └── chroma_db/           # 向量数据库（自动生成）
├── scripts/
│   └── init_db.py           # 数据库初始化脚本
├── test/                    # 测试文件
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ 开发说明

### 运行单个模块测试

```bash
# 测试文档切分
python app/rag/services.py

# 测试检索
python app/rag/retriever.py

# 测试 RAG 链
python app/rag/chains.py
```

### 重建向量数据库

如果 PDF 内容有更新，需要重建数据库：

```bash
# 删除旧数据库
rm -rf data/chroma_db

# 重新初始化
python scripts/init_db.py
```

---

## ❓ 常见问题

**Q: 首次请求响应很慢，是正常的吗？**
A: 是的。首次请求时会初始化向量数据库，需要 10-30 秒。后续请求会很快。

**Q: 提示 "Ollama 服务未运行" 怎么办？**
A: 确保已启动 Ollama 服务：`ollama serve`

**Q: 对话历史会丢失吗？**
A: 对话历史保存在 SQLite 数据库中，服务重启不会丢失。

**Q: 可以换用其他 LLM 吗？**
A: 可以。修改 `.env` 文件中的 LLM 配置，支持 OpenAI、Azure OpenAI、本地 LLaMA 等。

---

## 🗺️ 项目路线图

- [x] **阶段 0**：法律问答系统（已完成）
  - RAG 问答
  - 混合检索（向量 + BM25）
  - 多轮对话
  - FastAPI 接口
  - Streamlit Web 界面
  - 检索来源追溯

- [ ] **阶段 1**：通用文档支持
  - 支持多种文档格式（PDF、Word、TXT）
  - 文档上传和管理
  - 用户系统

- [ ] **阶段 2**：智能文档分析
  - 文档自动总结
  - 关键信息提取
  - 文档对比

- [ ] **阶段 3**：Agent 能力
  - 智能任务拆解
  - 多工具协作
  - 复杂业务场景

---

## 📝 注意事项

1. **首次启动慢**：首次请求时会初始化向量数据库，需要 10-30 秒
2. **Ollama 服务**：确保 Ollama 服务正在运行
3. **NVIDIA API Key**：需要在 `.env` 文件中配置
4. **仅供参考**：本系统回答仅供参考，不构成法律建议

---

## 🤝 贡献和反馈

这是一个持续迭代的个人项目，欢迎提出建议和反馈。

---

## 📄 许可证

MIT License
