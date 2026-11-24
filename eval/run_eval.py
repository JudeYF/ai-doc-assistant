#!/usr/bin/env python3
"""评测 ai-doc-assistant 的检索和（可选）回答质量。

默认只跑检索指标，不启动 FastAPI / Streamlit，也不会重建向量库。
检索会调用项目里的 get_retrieval_result，因此本机需要已有
data/chroma_db_v2（若没有，请先运行 python scripts/init_db.py），并且 Ollama 已加载嵌入模型（查询时要算 query 向量）。

用法（在仓库根目录）:

    source .venv/bin/activate
    python eval/run_eval.py                 # 仅检索
    python eval/run_eval.py --with-answers  # 检索 + 生成回答
    python eval/run_eval.py --with-answers --with-llm-judge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
DEFAULT_SET = EVAL_DIR / "eval_set.jsonl"
DEFAULT_OUT = EVAL_DIR / "results"

# 只匹配作为条文标题出现的“第X条”，避免把正文里的“依照本法第四十三条”
# 这种交叉引用算成该条被检索到。
ARTICLE_HEADER_RE = re.compile(r"(?:^|\n)\s*(第[一二三四五六七八九十百零]+条)")
# 回答里的引用，允许“第十条”或“第10条”。
CITATION_RE = re.compile(r"第\s*([一二三四五六七八九十百零]+|\d+)\s*条")

CN_DIGIT = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
# 拒答启发式：用于不可回答题目。宁可漏判、再交给可选的 LLM judge。
REFUSAL_RE = re.compile(
    r"("
    r"材料中没有|提供的材料中没有|所提供的.{0,20}没有|"
    r"没有相关(内容|规定|条文)|未[见有]相关|"
    r"未(规定|提及|载明|说明)|"
    r"本法(条文本身)?并?未规定|条文本身并未规定|"
    r"无法根据.{0,20}(回答|判断|得出)|不能(编造|从材料|根据材料)|"
    r"不属于本(法|材料)|超出.{0,10}(范围|材料)|"
    r"未找到|找不到|检索.{0,8}没有|"
    r"没有关于.{0,20}的规定"
    r")"
)


def die(msg: str, code: int = 1) -> None:
    print(f"错误: {msg}", file=sys.stderr)
    sys.exit(code)


def cn_to_int(s: str) -> int:
    s = s.strip()
    if s.isdigit():
        return int(s)
    s = s.replace("第", "").replace("条", "").replace("章", "").replace("节", "")

    def below_100(part: str) -> int:
        if not part:
            return 0
        if part.startswith("零"):
            return below_100(part[1:])
        if part == "十":
            return 10
        if part.startswith("十"):
            return 10 + CN_DIGIT.get(part[1], 0)
        if "十" in part:
            a, b = part.split("十", 1)
            return CN_DIGIT[a] * 10 + (CN_DIGIT.get(b, 0) if b else 0)
        if len(part) == 1 and part in CN_DIGIT:
            return CN_DIGIT[part]
        raise ValueError(f"无法解析中文数字: {part!r}")

    if "百" in s:
        left, right = s.split("百", 1)
        hundreds = CN_DIGIT[left] if left else 1
        return hundreds * 100 + below_100(right)
    return below_100(s)


def int_to_cn_article(n: int) -> str:
    digits = "零一二三四五六七八九"
    if n <= 0 or n > 119:
        raise ValueError(n)
    if n < 10:
        body = digits[n]
    elif n == 10:
        body = "十"
    elif n < 20:
        body = "十" + digits[n - 10]
    elif n < 100:
        tens, ones = divmod(n, 10)
        body = digits[tens] + "十" + (digits[ones] if ones else "")
    elif n == 100:
        body = "一百"
    elif n < 110:
        body = "一百零" + digits[n - 100]
    elif n == 110:
        body = "一百一十"
    else:
        body = "一百一十" + digits[n - 110]
    return f"第{body}条"


def normalize_article(token: str) -> str | None:
    try:
        n = cn_to_int(token)
    except ValueError:
        return None
    if n < 1 or n > 119:
        return None
    return int_to_cn_article(n)


def articles_from_headers(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for raw in ARTICLE_HEADER_RE.findall(text or ""):
        name = normalize_article(raw)
        if name and name not in seen:
            seen.add(name)
            found.append(name)
    return found


def articles_from_metadata(metadata: dict[str, Any] | None) -> list[str]:
    """优先用 chunk.metadata['article']；支持逗号分隔的多条号。"""
    if not metadata:
        return []
    raw = metadata.get("article")
    if raw is None or str(raw).strip() == "":
        return []
    found: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[,，、;；|]+", str(raw)):
        name = normalize_article(part)
        if name and name not in seen:
            seen.add(name)
            found.append(name)
    return found


def articles_from_doc(doc: Any) -> list[str]:
    """命中判断：有 metadata['article'] 时用它，否则回退到正文标题匹配。"""
    metadata = dict(getattr(doc, "metadata", {}) or {})
    from_meta = articles_from_metadata(metadata)
    if from_meta:
        return from_meta
    return articles_from_headers(getattr(doc, "page_content", "") or "")


def articles_from_citations(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for raw in CITATION_RE.findall(text or ""):
        name = normalize_article(raw)
        if name and name not in seen:
            seen.add(name)
            found.append(name)
    return found


def flatten(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def evidence_in_text(evidence: str, text: str) -> bool:
    ev = flatten(evidence)
    body = flatten(text)
    return bool(ev) and ev in body


def looks_like_refusal(answer: str) -> bool:
    return bool(REFUSAL_RE.search(answer or ""))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                die(f"{path} 第 {line_no} 行不是合法 JSON: {exc}")
    return items


def fmt_pct(num: float, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{num / den:.1%} ({num:.0f}/{den})" if float(num).is_integer() else f"{num / den:.1%}"


def average(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def setup_imports() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)


def ensure_index_exists() -> Path:
    v2 = PROJECT_ROOT / "data" / "chroma_db_v2"
    if v2.exists():
        return v2
    die(
        f"未找到向量库 {v2}。请先在仓库根目录运行 "
        f"`python scripts/init_db.py`。本脚本不会自动重建索引。"
    )


def retrieve(question: str, k: int = 6) -> list[Any]:
    from app.rag.retriever import get_retrieval_result

    docs = get_retrieval_result(question, k=k)
    return list(docs or [])


def generate_answer(question: str, context_docs: list[Any]) -> str:
    """用项目同一套提示词和 LLM 配置生成回答，但不写入 chat_history.db。"""
    from langchain_core.output_parsers import StrOutputParser

    from app.rag.chains import _llm_api_key, llm, prompt

    if not _llm_api_key():
        die("LLM_API_KEY 未设置，无法生成回答。请检查 .env，或去掉 --with-answers。")

    context = "\n\n".join(getattr(doc, "page_content", str(doc)) for doc in context_docs)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke(
        {
            "context": context,
            "history": "",
            "question": question,
        }
    )


JUDGE_PROMPT = """你是法律问答评测员。只根据给定的参考答案和原文证据，判断模型回答是否正确。
不要引入材料之外的法律知识。

问题：{question}

参考答案：{reference_answer}

原文证据：
{evidence}

模型回答：
{model_answer}

请只输出一段 JSON，不要附加其他文字，字段如下：
- "score": 1 表示实质正确（关键结论和主要数字/条文与参考答案一致，允许措辞不同）；0.5 表示部分正确；0 表示错误、编造或答非所问。
- "reason": 一句话理由。
对于不可回答的问题：若模型明确说明材料中没有、并拒绝编造，即使提到了相关但不含答案的条文，也判 1；若编造了材料中没有的具体规则或数字，判 0。
"""


def llm_judge(item: dict[str, Any], model_answer: str) -> dict[str, Any]:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from app.rag.chains import _llm_api_key, llm

    if not _llm_api_key():
        die("LLM_API_KEY 未设置，无法做 LLM judge。")

    evidence = "\n".join(item.get("gold_evidence") or []) or "（无，本题在语料中不可回答）"
    chain = ChatPromptTemplate.from_template(JUDGE_PROMPT) | llm | StrOutputParser()
    raw = chain.invoke(
        {
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "evidence": evidence,
            "model_answer": model_answer,
        }
    )
    parsed = {"score": None, "reason": raw.strip(), "raw": raw}
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            score = data.get("score")
            if score in (0, 0.5, 1, "0", "0.5", "1"):
                parsed["score"] = float(score)
            parsed["reason"] = str(data.get("reason") or parsed["reason"])
        except json.JSONDecodeError:
            pass
    return parsed


def eval_retrieval(item: dict[str, Any], docs: list[Any], ks: list[int]) -> dict[str, Any]:
    gold = list(item.get("source", {}).get("articles") or [])
    rows = []
    first_hit_rank = None
    found_by_rank: dict[str, int] = {}
    for rank, doc in enumerate(docs, 1):
        text = getattr(doc, "page_content", "") or ""
        metadata = dict(getattr(doc, "metadata", {}) or {})
        headers = articles_from_doc(doc)
        ev_hits = [evidence_in_text(ev, text) for ev in (item.get("gold_evidence") or [])]
        rows.append(
            {
                "rank": rank,
                "articles": headers,
                "source_metadata": metadata.get("source"),
                "article_metadata": metadata.get("article"),
                "chunk_id": metadata.get("chunk_id"),
                "evidence_hit": any(ev_hits),
                "preview": text[:180].replace("\n", " "),
            }
        )
        for name in headers:
            if name in gold and name not in found_by_rank:
                found_by_rank[name] = rank
        if first_hit_rank is None and any(name in gold for name in headers):
            first_hit_rank = rank

    result: dict[str, Any] = {
        "gold_articles": gold,
        "retrieved_count": len(docs),
        "first_hit_rank": first_hit_rank,
        "mrr": (1.0 / first_hit_rank) if first_hit_rank else 0.0,
        "articles_found": sorted(found_by_rank, key=lambda n: found_by_rank[n]),
        "article_ranks": found_by_rank,
        "retrieved": rows,
        "hits": {},
        "article_recall": {},
    }
    for k in ks:
        top = rows[:k]
        hit = any(any(a in gold for a in r["articles"]) for r in top)
        found = {a for r in top for a in r["articles"] if a in gold}
        result["hits"][f"hit@{k}"] = bool(hit) if gold else None
        result["article_recall"][f"recall@{k}"] = (
            (len(found) / len(gold)) if gold else None
        )
    return result


def eval_answer(item: dict[str, Any], answer: str) -> dict[str, Any]:
    gold = list(item.get("source", {}).get("articles") or [])
    cited = articles_from_citations(answer)
    cited_set, gold_set = set(cited), set(gold)
    inter = cited_set & gold_set
    answerable = bool(item.get("answerable", item.get("type") != "unanswerable"))
    refusal = looks_like_refusal(answer)
    out: dict[str, Any] = {
        "cited_articles": cited,
        "citation_hit": bool(inter) if answerable else None,
        "citation_precision": (len(inter) / len(cited_set)) if cited_set and answerable else None,
        "citation_recall": (len(inter) / len(gold_set)) if gold_set and answerable else None,
        "citation_exact": (cited_set == gold_set) if answerable else None,
        "spurious_citation": bool(cited_set) if not answerable else None,
        "refusal_detected": refusal,
        "refusal_correct": (refusal if not answerable else None),
    }
    return out


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    widths = [max(len(c) for c in col) for col in zip(*rows)]
    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
    header, *body = rows
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(header), sep, *[fmt(r) for r in body]])


def main() -> None:
    parser = argparse.ArgumentParser(description="评测 ai-doc-assistant 检索与回答")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_SET, help="评测集 jsonl 路径")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="结果输出目录")
    parser.add_argument("--with-answers", action="store_true", help="调用项目 LLM 生成回答并计算引用/拒答指标")
    parser.add_argument("--with-llm-judge", action="store_true", help="再用项目 LLM 给回答正确性打分（隐含 --with-answers）")
    parser.add_argument("--k", type=int, default=6, help="Hit@k 的最大 k；默认 6，与检索器配置一致")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，便于试跑")
    parser.add_argument("--ids", type=str, default="", help="只跑指定 id，逗号分隔，如 e01,e21,e35")
    args = parser.parse_args()
    if args.with_llm_judge:
        args.with_answers = True

    setup_imports()
    ensure_index_exists()

    eval_set = args.eval_set if args.eval_set.is_absolute() else (Path.cwd() / args.eval_set)
    if not eval_set.exists():
        die(f"找不到评测集: {eval_set}")
    items = load_jsonl(eval_set)
    if args.ids:
        keep = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [it for it in items if it.get("id") in keep]
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    if not items:
        die("没有要评测的题目。")

    ks = sorted({1, 3, 5, args.k})
    ks = [k for k in ks if k > 0]

    out_dir = args.out_dir if args.out_dir.is_absolute() else (PROJECT_ROOT / args.out_dir if not str(args.out_dir).startswith("eval/") else args.out_dir)
    # Keep results under eval/results by default even if cwd changed.
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"评测集: {eval_set}  ({len(items)} 条)")
    print(f"输出目录: {out_dir}")
    print("模式: " + ("检索 + 回答" + (" + LLM judge" if args.with_llm_judge else "") if args.with_answers else "仅检索"))
    print("正在初始化检索器（首次会加载 Chroma 和 BM25）...")

    per_item: list[dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        qid = item.get("id", f"item{i}")
        print(f"[{i}/{len(items)}] {qid} ({item.get('type')}) {item['question'][:40]}...")
        docs = retrieve(item["question"], k=args.k)
        retrieval = eval_retrieval(item, docs, ks)
        record: dict[str, Any] = {
            "id": qid,
            "type": item.get("type"),
            "answerable": bool(item.get("answerable", item.get("type") != "unanswerable")),
            "question": item["question"],
            "source": item.get("source"),
            "retrieval": retrieval,
        }
        if args.with_answers:
            try:
                answer = generate_answer(item["question"], docs)
            except Exception as exc:  # noqa: BLE001 — 单条失败不应中断整次评测
                answer = ""
                record["answer_error"] = str(exc)
                print(f"    生成回答失败: {exc}")
            record["model_answer"] = answer
            record["answer"] = eval_answer(item, answer)
            if args.with_llm_judge and answer:
                try:
                    record["llm_judge"] = llm_judge(item, answer)
                except Exception as exc:  # noqa: BLE001
                    record["llm_judge"] = {"score": None, "reason": f"judge 失败: {exc}"}
                    print(f"    LLM judge 失败: {exc}")
        per_item.append(record)

    # -------- aggregate --------
    answerable = [r for r in per_item if r["answerable"]]
    unanswerable = [r for r in per_item if not r["answerable"]]
    by_type: dict[str, list] = defaultdict(list)
    for r in per_item:
        by_type[r.get("type") or "unknown"].append(r)

    def mean_or_none(vals: list[float | None]) -> float | None:
        xs = [v for v in vals if v is not None]
        return average(xs)

    summary: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "eval_set": str(eval_set),
        "n_items": len(per_item),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "counts_by_type": dict(Counter(r.get("type") for r in per_item)),
        "k_values": ks,
        "with_answers": args.with_answers,
        "with_llm_judge": args.with_llm_judge,
        "notes": [
            "检索命中优先使用 chunk.metadata['article']；没有该字段时，回退到从正文识别条文标题（第X条）。",
            "get_retrieval_result 的 k 会传给向量检索和 BM25；本脚本按返回列表的名次计算 Hit@k。",
            "生成回答时使用与项目相同的提示词和 LLM 环境变量，但 history 为空，且不写入 data/chat_history.db。",
        ],
    }

    retrieval_metrics: dict[str, Any] = {}
    for k in ks:
        hits = [r["retrieval"]["hits"][f"hit@{k}"] for r in answerable]
        hits_f = [1.0 if h else 0.0 for h in hits if h is not None]
        recalls = [r["retrieval"]["article_recall"][f"recall@{k}"] for r in answerable]
        retrieval_metrics[f"hit@{k}"] = mean_or_none(hits_f)
        retrieval_metrics[f"article_recall@{k}"] = mean_or_none(recalls)
    retrieval_metrics["mrr"] = mean_or_none([r["retrieval"]["mrr"] for r in answerable])
    summary["retrieval"] = retrieval_metrics

    if args.with_answers:
        ans_metrics: dict[str, Any] = {
            "citation_hit": mean_or_none(
                [1.0 if r.get("answer", {}).get("citation_hit") else 0.0 for r in answerable if r.get("model_answer")]
            ),
            "citation_precision": mean_or_none(
                [r.get("answer", {}).get("citation_precision") for r in answerable]
            ),
            "citation_recall": mean_or_none(
                [r.get("answer", {}).get("citation_recall") for r in answerable]
            ),
            "citation_exact": mean_or_none(
                [1.0 if r.get("answer", {}).get("citation_exact") else 0.0 for r in answerable if r.get("model_answer")]
            ),
            "refusal_accuracy": mean_or_none(
                [1.0 if r.get("answer", {}).get("refusal_correct") else 0.0 for r in unanswerable if r.get("model_answer")]
            ),
        }
        if args.with_llm_judge:
            judge_all = [
                r["llm_judge"]["score"]
                for r in per_item
                if r.get("llm_judge") and r["llm_judge"].get("score") is not None
            ]
            judge_ans = [
                r["llm_judge"]["score"]
                for r in answerable
                if r.get("llm_judge") and r["llm_judge"].get("score") is not None
            ]
            ans_metrics["llm_judge_mean_all"] = average(judge_all)
            ans_metrics["llm_judge_mean_answerable"] = average(judge_ans)
        summary["answers"] = ans_metrics

    type_rows = {}
    for typ, group in by_type.items():
        ans_g = [r for r in group if r["answerable"]]
        type_rows[typ] = {
            "n": len(group),
            "hit@1": mean_or_none([1.0 if r["retrieval"]["hits"].get("hit@1") else 0.0 for r in ans_g]),
            "hit@" + str(args.k): mean_or_none(
                [1.0 if r["retrieval"]["hits"].get(f"hit@{args.k}") else 0.0 for r in ans_g]
            ),
            "mrr": mean_or_none([r["retrieval"]["mrr"] for r in ans_g]),
        }
    summary["by_type"] = type_rows

    per_path = out_dir / "per_item.jsonl"
    with per_path.open("w", encoding="utf-8") as f:
        for r in per_item:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sum_path = out_dir / "summary.json"
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def pct(v: float | None) -> str:
        return "n/a" if v is None else f"{v:.1%}"

    table = [
        ["指标", "数值", "说明"],
        ["题目数", str(len(per_item)), f"可回答 {len(answerable)} / 不可回答 {len(unanswerable)}"],
    ]
    for k in ks:
        table.append(
            [
                f"Hit@{k}",
                pct(retrieval_metrics.get(f"hit@{k}")),
                "可回答题中，前 k 个片段至少命中一条金标条文",
            ]
        )
        table.append(
            [
                f"Article-Recall@{k}",
                pct(retrieval_metrics.get(f"article_recall@{k}")),
                "可回答题中，金标条文在前 k 个片段里被找回的比例（再对题目取平均）",
            ]
        )
    table.append(["MRR", f"{retrieval_metrics['mrr']:.3f}" if retrieval_metrics["mrr"] is not None else "n/a", "第一条金标条文所在片段名次的倒数，再平均"])
    if args.with_answers:
        table.append(["引用命中", pct(summary["answers"]["citation_hit"]), "可回答题的回答里至少引用了一条金标条文"])
        table.append(["引用精确率", pct(summary["answers"]["citation_precision"]), "回答里引用的条文中，属于金标的比例"])
        table.append(["引用召回率", pct(summary["answers"]["citation_recall"]), "金标条文中被回答引用到的比例"])
        table.append(["拒答准确率", pct(summary["answers"]["refusal_accuracy"]), "不可回答题中，回答被判定为拒绝/说明未找到的比例"])
        if args.with_llm_judge:
            v = summary["answers"].get("llm_judge_mean_all")
            table.append(["LLM judge 均分", "n/a" if v is None else f"{v:.3f}", "0 / 0.5 / 1，全部题目"])

    md_lines = [
        "# 评测结果摘要",
        "",
        f"- 时间: {summary['created_at']}",
        f"- 评测集: `{eval_set}`",
        f"- 题目数: {len(per_item)}（可回答 {len(answerable)}，不可回答 {len(unanswerable)}）",
        "",
        markdown_table(table),
        "",
        "## 按题型",
        "",
    ]
    type_table = [["题型", "题数", "Hit@1", f"Hit@{args.k}", "MRR"]]
    for typ in ("single_article", "multi_article", "paraphrase", "unanswerable"):
        if typ not in type_rows:
            continue
        row = type_rows[typ]
        type_table.append(
            [
                typ,
                str(row["n"]),
                pct(row["hit@1"]),
                pct(row.get(f"hit@{args.k}")),
                "n/a" if row["mrr"] is None else f"{row['mrr']:.3f}",
            ]
        )
    md_lines.append(markdown_table(type_table))
    md_lines.append("")
    md_lines.append("不可回答题不计入 Hit@k / MRR。逐条结果见 `per_item.jsonl`。")
    md_lines.append("")
    md_path = out_dir / "summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print()
    print(markdown_table(table))
    print()
    print(f"已写入:\n  {per_path}\n  {sum_path}\n  {md_path}")


if __name__ == "__main__":
    main()
