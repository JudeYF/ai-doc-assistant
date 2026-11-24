# 治安管理处罚法问答示例

A small LangChain RAG demo: question answering over the PRC Public Security Administration Punishments Law（《中华人民共和国治安管理处罚法》）.

语料只有这一份公开 PDF。提问后，系统用 Chroma 向量检索和 BM25 各取一批片段，交给一个 OpenAI 兼容的对话模型生成回答。可以通过 FastAPI 或 Streamlit 使用。

这是一个能跑起来的学习项目。`eval/` 里有一份基于当前语料的评测集。回答不构成法律意见。

## 运行前需要

- Python 3.12
- 本机 [Ollama](https://ollama.com) 已启动，并已拉取嵌入模型 `bge-m3:latest`
- 一个 OpenAI 兼容的对话模型接口和密钥。`.env.example` 里的示例是智谱 GLM，换成别的兼容地址也可以

## 安装并启动

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

编辑 `.env`，至少填入 `LLM_API_KEY`。模型名和接口地址也在这个文件里。

另开一个终端，准备嵌入模型：

```bash
ollama serve
ollama pull bge-m3:latest
```

构建向量索引（只需第一次，或 PDF 变更后）：

```bash
python scripts/init_db.py
```

脚本读取 `data/中华人民共和国治安管理处罚法.pdf`，按条切分后写入 `data/chroma_db_v2/`（collection 名 `china_public_security_law_v2`）。旧索引若仍在 `data/chroma_db/`，不会被覆盖。这两个目录和会话文件 `data/chat_history.db` 是运行产物，不进 git。要重建新索引，删掉 `data/chroma_db_v2` 再执行一次上面的命令。

### API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- 健康检查：<http://127.0.0.1:8000/health>
- 接口文档：<http://127.0.0.1:8000/docs>

```bash
curl -s http://127.0.0.1:8000/health

curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"阻碍执行紧急任务的消防车会如何处罚？","session_id":"demo"}'
```

`/health` 在密钥已填写时返回 `llm_configured: true`。未填写时服务仍能启动，但 `/chat` 会返回错误。

### 页面

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

打开 <http://127.0.0.1:8501>。侧栏有几条示例问题。回答下面可以展开本次检索到的片段。

## 配置

| 变量 | 含义 | 默认 |
| --- | --- | --- |
| `LLM_API_KEY` | 对话模型密钥 | 空 |
| `LLM_MODEL` | 模型名 | `glm-5` |
| `LLM_BASE_URL` | OpenAI 兼容接口 | `https://open.bigmodel.cn/api/paas/v4/` |
| `LLM_TEMPERATURE` | 采样温度 | `0.6` |
| `OLLAMA_BASE_URL` | Ollama 地址 | `http://127.0.0.1:11434` |
| `OLLAMA_EMBEDDING_MODEL` | 嵌入模型 | `bge-m3:latest` |

密钥、模型名和地址都从环境变量读取，代码里没有写死的密钥或本机绝对路径。

## 项目结构

```
app/main.py            FastAPI：/health、/chat
app/streamlit_app.py   聊天页
app/rag/services.py    加载 PDF、清洗、按条切分（写入条号等 metadata）
app/rag/retriever.py   Chroma + BM25 混合检索
app/rag/chains.py      提示词和问答链，会话写入 SQLite
app/rag/utils.py       去掉页码等噪音
scripts/init_db.py     构建或加载向量库
eval/                  评测集和跑分脚本
data/                  法律 PDF（索引和会话库运行时生成）
```

`.streamlit/config.toml` 只关闭了 Streamlit 的使用统计。

## 已知限制

- 评测集在 `eval/`，用法和基线对比见 `eval/README.md`。检索仍未系统调参：BM25 用的是按空格切分的默认分词，中文效果有限，向量和关键词的权重仍是写死的 0.6 / 0.4。
- Agent 层还没做。之前的搜索 / SQL Agent 跑不起来，已经从仓库里去掉。如果以后要加，会单独做一层，而不是塞进现在的问答链。
- 多轮历史会写入 SQLite，但提示词里的历史接法还不对，追问不一定用到上一轮内容。问答链的这部分留着以后改。
- 嵌入模型必须由本机 Ollama 提供。Ollama 没开时，建索引和提问都会失败。
- 页面上展开的「相关条文」是另一次检索的结果，和生成回答时用的那一次不是同一次调用。
- 模型可能说出条文里没有的话。提示词要求它依据材料作答，代码没有再做校验。

回答只供演示，不要当成法律依据。
