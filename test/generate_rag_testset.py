"""
RAG 冷启动测试集生成脚本
使用 LangSmith 创建测试集，基于《治安管理处罚法》文档生成问答对
使用 ChatOpenAI 调用智谱 GLM-5 模型
"""
import os
import time
from pathlib import Path
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ==========================
# 配置
# ==========================
# 智谱 GLM-5 模型（通过 OpenAI 兼容接口调用智谱）
ZHIPU_MODEL = "glm-5"

# 智谱 API 配置（与 chains.py 保持一致）
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# 测试集名称
DATASET_NAME = "security-law-rag-cold-start"

# 数据源路径
PDF_PATH = Path(__file__).parent.parent / "data" / "中华人民共和国治安管理处罚法.pdf"

# 生成配置
TARGET_QA_COUNT = 40  # 目标生成问答对数量
QUESTIONS_PER_CHUNK = 4  # 每个片段生成的问答对数量（增加到 4 个）
CHUNK_SIZE = 2500  # 每个片段的字符数

# API 调用延迟
API_CALL_DELAY = 1.0  # 每次调用间隔 1 秒


# ==========================
# 初始化
# ==========================
print("🔧 初始化 LangSmith 客户端...")
client = Client()

# 检查/创建数据集
if client.has_dataset(dataset_name=DATASET_NAME):
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    print(f"✅ 找到已存在的数据集: {DATASET_NAME}")
    existing_count = len(list(client.list_examples(dataset_id=dataset.id)))
    print(f"   当前包含 {existing_count} 个示例")

    # 询问是否重新创建
    print(f"\n⚠️  你计划生成 {TARGET_QA_COUNT} 个新问答对")
    response = input("\n是否删除旧数据集并重新创建？(y/n): ")
    if response.lower() == 'y':
        client.delete_dataset(dataset.id)
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="治安管理处罚法 RAG 冷启动测试集 - 使用智谱 GLM-5 生成，60 个问答对"
        )
        print("🗑️  旧数据集已删除，创建新数据集")
else:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="治安管理处罚法 RAG 冷启动测试集 - 使用智谱 GLM-5 生成，60 个问答对"
    )
    print(f"✅ 创建新数据集: {DATASET_NAME}")


# ==========================
# 初始化 LLM（通过 ChatOpenAI 调用智谱）
# ==========================
print(f"🤖 使用智谱模型: {ZHIPU_MODEL}")
print(f"   Base URL: {ZHIPU_BASE_URL}")

generator_llm = ChatOpenAI(
    model=ZHIPU_MODEL,
    base_url=ZHIPU_BASE_URL,
    api_key=ZHIPU_API_KEY,
    temperature=0.3,
    max_tokens=2048,
)


# ==========================
# 加载文档
# ==========================
print(f"\n📄 加载法律文档: {PDF_PATH}")
loader = PyMuPDFLoader(str(PDF_PATH))
docs = loader.load()

# 合并所有页面内容
full_text = "\n".join([doc.page_content for doc in docs])
print(f"📖 文档总长度: {len(full_text)} 字符")

# 计算需要的片段数
num_chunks = (TARGET_QA_COUNT + QUESTIONS_PER_CHUNK - 1) // QUESTIONS_PER_CHUNK
text_chunks = [full_text[i:i+CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]

# 限制处理片段数
text_chunks = text_chunks[:num_chunks]

print(f"📝 将处理 {len(text_chunks)} 个文本片段")
print(f"   预计生成 {len(text_chunks) * QUESTIONS_PER_CHUNK} 个问答对")
print(f"⏱️  预计耗时约 {len(text_chunks) * API_CALL_DELAY} 秒\n")


# ==========================
# 生成测试数据的 Prompt
# ==========================
prompt = ChatPromptTemplate.from_template("""
你是一个专业的法律知识出题专家。请阅读以下《中华人民共和国治安管理处罚法》的文本片段，
从中生成 {num_questions} 个不同角度、贴近实际生活的用户提问，以及对应的绝对准确的标准答案。

**问题类型要求**（务必覆盖不同类型）：
1. **行为认定类**：询问某种行为是否违法、属于什么性质
   例如："有人在电影院抽烟，算不算违反治安管理？"
2. **处罚标准类**：询问违法行为的处罚力度、罚款金额、拘留天数
   例如："打架斗殴如果造成轻微伤，会被拘留多久？"
3. **情节轻重类**：询问从重、从轻处罚的情形
   例如："教唆他人违法，处罚会更重吗？"
4. **程序问题类**：询问执法程序、复议、申诉等
   例如："对处罚不服，可以向哪个部门申诉？"
5. **特殊情形类**：涉及年龄、身份等特殊情况
   例如："不满14周岁的未成年人违法会怎么处理？"

**生成要求**：
- 问题要符合真实用户的口吻，具有实际应用场景
- 答案必须100%严格基于提供的文本内容，不能有任何外部知识或推测
- 每个问题的类型要尽量不同
- 请严格输出为 JSON 格式，包含一个 'examples' 数组，每个元素包含 'question' 和 'ground_truth' 两个字段

**输出格式示例**：
{{
  "examples": [
    {{
      "question": "问题1",
      "ground_truth": "答案1"
    }},
    {{
      "question": "问题2",
      "ground_truth": "答案2"
    }},
    {{
      "question": "问题3",
      "ground_truth": "答案3"
    }}
  ]
}}

**法律文本片段**:
{text}
""")


# ==========================
# 生成并上传数据
# ==========================
chain = prompt | generator_llm | JsonOutputParser()

print(f"🚀 开始生成测试数据...\n")

total_generated = 0
failed = 0
question_types = set()

for i, chunk in enumerate(text_chunks, 1):
    print(f"📦 处理片段 {i}/{len(text_chunks)}...")

    # 添加延迟
    if i > 1:
        time.sleep(API_CALL_DELAY)

    try:
        synthetic_data = chain.invoke({
            "text": chunk,
            "num_questions": QUESTIONS_PER_CHUNK
        })
        examples = synthetic_data.get("examples", [])

        if not examples:
            print(f"    ⚠️  未生成问答对，跳过")
            failed += 1
            continue

        for example in examples:
            question = example.get("question", "")
            ground_truth = example.get("ground_truth", "")

            if not question or not ground_truth:
                continue

            client.create_example(
                inputs={"question": question},
                outputs={"ground_truth": ground_truth},
                dataset_id=dataset.id,
            )
            total_generated += 1

            # 识别问题类型
            q_type = "行为认定"
            if any(word in question for word in ["处罚", "拘留", "罚款"]):
                q_type = "处罚标准"
            elif any(word in question for word in ["从重", "从轻", "减轻", "情节"]):
                q_type = "情节轻重"
            elif any(word in question for word in ["复议", "申诉", "执法", "程序"]):
                q_type = "程序问题"
            elif any(word in question for word in ["年龄", "未成年", "老人", "孕妇"]):
                q_type = "特殊情形"

            question_types.add(q_type)
            print(f"    ✅ [{q_type}] {total_generated}: {question[:45]}...")

    except Exception as e:
        print(f"    ❌ 片段 {i} 生成失败: {e}")
        failed += 1
        continue


# ==========================
# 完成
# ==========================
print(f"\n{'='*70}")
print(f"🎉 测试集生成完成！")
print(f"{'='*70}")
print(f"📊 统计信息:")
print(f"   - 成功生成: {total_generated} 个问答对")
print(f"   - 失败片段: {failed} 个")
print(f"   - 成功率: {total_generated / (total_generated + failed * QUESTIONS_PER_CHUNK) * 100:.1f}%")
print(f"\n📋 问题类型分布:")
for q_type in sorted(question_types):
    print(f"   - {q_type}")
print(f"\n📦 数据集信息:")
print(f"   - 名称: {DATASET_NAME}")
print(f"   - 生成模型: {ZHIPU_MODEL}")
print(f"{'='*70}")
print(f"\n🔗 请在 LangSmith 控制台查看测试集详情")
print(f"   URL: https://smith.langchain.com")
