"""
RAG 冷启动评估脚本
使用 LangSmith 评估 RAG 系统的性能
使用 ChatOpenAI 调用智谱 GLM-5 模型
"""
import os
import time
from langsmith import evaluate
from langsmith import Client
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# 导入项目的 RAG 模模块
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from app.rag.chains import ask
from app.rag.retriever import get_retrieval_result

load_dotenv()

# ==========================
# 配置
# ==========================
# 智谱 GLM-5 模型（通过 OpenAI 兼容接口调用智谱）
ZHIPU_MODEL = "glm-5"

# 智谱 API 配置（与 chains.py 保持一致）
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")

# API 调用延迟
API_CALL_DELAY = 1.0  # 每次调用间隔 1 秒

# 测试集名称
DATASET_NAME = "security-law-rag-cold-start"
# 实验名称前缀
EXPERIMENT_PREFIX = "RAG-Cold-Start-GLM5"

# 并发度
MAX_CONCURRENCY = 2

# LangSmith 评估器
from langsmith.evaluation import (
    LangChainStringEvaluator,
    TrajectoryEvaluator,
    TrajectoryEvalLoss,
    CotQAEvalChain,
)
from langchain.evaluation import EvaluatorType


# ==========================
# 1. 包装 RAG 系统
# ==========================
def predict_rag_answer(inputs: dict) -> dict:
    """
    LangSmith 评估器调用的函数

    Args:
        inputs: 包含 'question' 的字典

    Returns:
        包含 'answer' 和 'contexts' 的字典
    """
    question = inputs["question"]

    # 获取检索结果（用于上下文评估）
    contexts = get_retrieval_result(question, k=6)
    context_texts = [doc.page_content for doc in contexts]

    # 调用 RAG 系统获取答案
    # 使用新的 session_id 避免历史干扰
    answer = ask(question, session_id=f"eval_{hash(question)}")

    return {
        "answer": answer,
        "contexts": context_texts,
    }


# ==========================
# 2. 配置评估裁判（通过 ChatOpenAI 调用智谱）
# ==========================
print(f"🤖 使用智谱评估模型: {ZHIPU_MODEL}")
print(f"   Base URL: {ZHIPU_BASE_URL}")

eval_llm = ChatOpenAI(
    model=ZHIPU_MODEL,
    base_url=ZHIPU_BASE_URL,
    api_key=ZHIPU_API_KEY,
    temperature=0,
    max_tokens=2048,
)


# ==========================
# 3. 定义评估指标
# ==========================
# QA 评估器：对比生成的答案和 ground_truth
qa_evaluator = LangChainStringEvaluator(
    input_key="question",
    prediction_key="answer",
    reference_key="ground_truth",
    evaluation_type=EvaluatorType.QA,
    llm=eval_llm,
    string_mapping={"contexts": "contexts"},
)

# Context QA 评估器：检查答案是否超出检索到的上下文
context_qa_evaluator = LangChainStringEvaluator(
    input_key="question",
    prediction_key="answer",
    reference_key="contexts",
    evaluation_type=EvaluatorType.CONTEXT_QA,
    llm=eval_llm,
)

# 汇总评估器列表
evaluators = [qa_evaluator, context_qa_evaluator]


# ==========================
# 4. 运行评估
# ==========================

# 获取测试集大小
client = Client()
dataset = client.read_dataset(dataset_name=DATASET_NAME)
total_examples = len(list(client.list_examples(dataset_id=dataset.id)))

print(f"\n🚀 开始评估 RAG 系统...")
print(f"📊 测试集: {DATASET_NAME}")
print(f"🔬 实验前缀: {EXPERIMENT_PREFIX}")
print(f"📏 测试集大小: {total_examples} 个问答对")

# 询问评估数量
print(f"\n每个问答对触发 2 次评估器调用（qa, context_qa）")
print(f"完整评估需要 {total_examples * 2} 次 API 调用")

default_eval_count = min(20, total_examples)  # 默认评估 20 个
eval_count_input = input(f"\n请输入要评估的问答对数量（默认 {default_eval_count}）：")
eval_count = int(eval_count_input) if eval_count_input.strip() else default_eval_count
eval_count = min(eval_count, total_examples)

print(f"\n📋 将评估 {eval_count} 个问答对")
print(f"⏱️  预计耗时约 {eval_count * 2 * API_CALL_DELAY / MAX_CONCURRENCY:.0f} 秒\n")

try:
    experiment_results = evaluate(
        predict_rag_answer,  # 待评估的函数
        data=DATASET_NAME,  # 测试集名称
        evaluators=evaluators,  # 评估器列表
        experiment_prefix=EXPERIMENT_PREFIX,  # 实验名称
        limit=eval_count,  # 限制评估数量
        num_repetitions=1,  # 每个问题重复评估次数
        max_concurrency=MAX_CONCURRENCY,  # 并发度
    )

    print("\n" + "="*70)
    print("✅ 评估完成！")
    print("="*70)
    print(f"📊 评估结果:")
    print(f"   - 实验名称: {EXPERIMENT_PREFIX}")
    print(f"   - 评估数量: {eval_count}")
    print(f"   - 剩余未评估: {total_examples - eval_count}")

except Exception as e:
    print(f"\n❌ 评估失败: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "="*70)
print("🔗 请前往 LangSmith 控制台查看详细评估报告")
print("   URL: https://smith.langchain.com")
print("="*70)
