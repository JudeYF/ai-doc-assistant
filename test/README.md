# RAG 冷启动测试

本项目提供了使用 LangSmith 对 RAG 系统进行冷启动评估的测试脚本。

## 模型配置

### 智谱 GLM-5 (推荐)
- **模型**: `glm-4-plus` (GLM-5 旗舰版)
- **通过**: OpenAI 兼容接口
- **优势**: 无严格调用限制，适合生成大量测试数据

### 备选模型
- `glm-4` - 通用模型
- `glm-4-flash` - 快速响应
- `glm-4-air` - 经济实惠

## 文件说明

- `generate_rag_testset.py` - 生成测试集脚本（使用智谱 GLM-5）
- `run_rag_eval.py` - 运行评估脚本（使用智谱 GLM-5）

## 使用步骤

### 1. 生成测试集

使用智谱 GLM-5 模型基于《治安管理处罚法》文档自动生成问答对测试集：

```bash
python test/generate_rag_testset.py
```

**配置说明：**

- `ZHIPU_MODEL = "glm-4-plus"` - 生成测试集的模型
- `DATASET_NAME = "security-law-rag-cold-start"` - LangSmith 测试集名称
- `TARGET_QA_COUNT = 60` - 目标生成问答对数量
- `QUESTIONS_PER_CHUNK = 3` - 每个片段生成的问答对数量

**生成的问答对类型：**

1. **行为认定类** - 询问某种行为是否违法、属于什么性质
2. **处罚标准类** - 询问违法行为的处罚力度、罚款金额、拘留天数
3. **情节轻重类** - 询问从重、从轻处罚的情形
4. **程序问题类** - 询问执法程序、复议、申诉等
5. **特殊情形类** - 涉及年龄、身份等特殊情况

**示例输出：**
```
🚀 开始生成测试数据...
📦 处理片段 1/20...
    ✅ [处罚标准] 1: 打架斗殴如果造成轻微伤，会被拘留多久？
    ✅ [行为认定] 2: 有人在电影院抽烟，算不算违反治安管理？
    ✅ [程序问题] 3: 对处罚不服，可以向哪个部门申诉？
...
```

### 2. 运行评估

生成测试集后，运行评估：

```bash
python test/run_rag_eval.py
```

脚本会提示输入要评估的问答对数量：

```
请输入要评估的问答对数量（默认 20）：[输入数量或按回车使用默认]
```

**评估指标：**（4 个核心指标）

| 指标 | 说明 |
|------|------|
| **qa** | 答案正确性：对比生成的答案和 ground_truth |
| **context_qa** | 忠实度/无幻觉：检查答案是否超出检索到的上下文 |
| **relevance** | 答案相关性：答案是否与问题相关 |
| **context_relevance** | 上下文相关性：检索到的上下文是否与问题相关 |

**调用计算：**
- 每个问答对触发 4 次评估器调用
- 评估 20 个问答对 = 80 次 API 调用
- 预计耗时约 40 秒（并发度 2）

### 3. 查看结果

评估完成后，访问 [LangSmith 控制台](https://smith.langchain.com) 查看详细的评估报告。

在报告中你可以查看：
- 每个问答对的详细评分
- 指标汇总统计（平均分、中位数等）
- 问题示例和答案对比
- 检索到的上下文内容

## 测试集管理

### 查看现有测试集

```python
from langsmith import Client
client = Client()
datasets = client.list_datasets()
for ds in datasets:
    examples = list(client.list_examples(dataset_id=ds.id))
    print(f"{ds.name}: {len(examples)} examples")
```

### 查看测试集内容

```python
from langsmith import Client
client = Client()
dataset = client.read_dataset(dataset_name="security-law-rag-cold-start")
examples = list(client.list_examples(dataset_id=dataset.id))
for i, ex in enumerate(examples, 1):
    print(f"\n--- 示例 {i} ---")
    print(f"问题: {ex.inputs['question']}")
    print(f"答案: {ex.outputs['ground_truth']}")
```

### 删除测试集

```python
from langsmith import Client
client = Client()
dataset = client.read_dataset(dataset_name="security-law-rag-cold-start")
client.delete_dataset(dataset.id)
```

## 成本与性能

### 智谱 API 定价（参考）

| 模型 | 输入价格 | 输出价格 | 适用场景 |
|------|---------|---------|---------|
| glm-4-plus | ¥12/M tokens | ¥12/M tokens | 生成和评估 |
| glm-4 | ¥5/M tokens | ¥5/M tokens | 成本敏感场景 |
| glm-4-flash | ¥1/M tokens | ¥2/M tokens | 快速测试 |
| glm-4-air | ¥1/M tokens | ¥1/M tokens | 经济评估 |

### 成本估算（60 个问答对）

| 操作 | Token 估算 | 成本估算 |
|------|-----------|---------|
| 生成 60 个问答对 | ~50K tokens | ~¥0.6 |
| 评估 20 个问答对 | ~100K tokens | ~¥1.2 |
| 评估全部 60 个问答对 | ~300K tokens | ~¥3.6 |

### 建议评估策略

1. **首次测试**：评估 10-20 个问答对，快速验证
2. **中期验证**：评估 30-40 个问答对，检查指标稳定性
3. **完整评估**：评估全部 60 个问答对，获得最终基准

## 常见问题

### Q: 生成的问答对质量不好怎么办？
A:
1. 调整生成 Prompt 中的要求描述
2. 使用更强的模型（`glm-4-plus`）
3. 手动补充高质量问答对

### Q: 评估分数偏低如何分析？
A:
1. 在 LangSmith 控制台查看具体失败案例
2. 检查 `context_qa` 分数：低分说明答案有幻觉
3. 检查 `context_relevance` 分数：低分说明检索质量差

### Q: 如何提高 `qa` 分数？
A:
1. 检查检索器配置（k 值、权重）
2. 优化 Prompt 模板
3. 检查文档切分质量

### Q: 如何提高 `context_qa` 分数？
A:
1. 在 Prompt 中强调"严格基于提供的法律条文"
2. 降低 LLM 的 temperature 参数
3. 添加"如果材料中没有相关内容，请明确说明"的指令

### Q: 如何提高 `context_relevance` 分数？
A:
1. 调整检索器的 k 值（增加检索结果数量）
2. 优化文档切分策略
3. 尝试不同的嵌入模型

## 进阶用法

### 对比实验

对比不同 RAG 配置的性能：

```python
# 实验前缀用于区分不同配置
EXPERIMENT_PREFIX = "RAG-Cold-Start-Config-v1"  # 修改这里

# 运行第一次评估
python test/run_rag_eval.py

# 修改 RAG 配置后，再运行
EXPERIMENT_PREFIX = "RAG-Cold-Start-Config-v2"
python test/run_rag_eval.py

# 在 LangSmith 中对比两个实验
```

### 自定义评估指标

添加自定义评估函数：

```python
def legal_citation_accuracy(run, example):
    """检查答案是否包含正确的法律条文引用"""
    answer = run.outputs["answer"]
    ground_truth = example.outputs["ground_truth"]

    # 提取条文号并对比
    # ... 自定义逻辑

    return {
        "key": "legal_citation",
        "score": 0.8,
        "comment": "答案包含了正确的条文号"
    }

# 在 eval_config.evaluators 中添加
eval_config = RunEvalConfig(
    evaluators=["qa", "context_qa", "relevance", "context_relevance"],
    custom_evaluators=[legal_citation_accuracy],
    ...
)
```

## 项目结构

```
test/
├── generate_rag_testset.py  # 生成测试集（智谱 GLM-5）
├── run_rag_eval.py          # 运行评估（智谱 GLM-5）
└── README.md                # 本文档
```
