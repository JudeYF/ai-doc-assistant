"""文档处理模块：按条切分法律文本，并写入条文级 metadata。"""
from __future__ import annotations

import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from app.rag.utils import clean_legal_text

# 条文标题出现在行首或换行后，如「第一条」「第一百零一条」
ARTICLE_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(第[一二三四五六七八九十百零]+条)(?=\s|$)"
)

CHAPTER_HEADER_RE = re.compile(r"第[一二三四五六七八九十百零]+章[ \t]*[^\n]+")
STRUCTURAL_HEADER_RE = re.compile(
    r"第[一二三四五六七八九十百零]+[章节][ \t]*[^\n]+"
)
TRAILING_STRUCTURE_RE = re.compile(
    r"(?P<body>[\s\S]*?)"
    r"(?P<tail>(?:\s*(?:第[一二三四五六七八九十百零]+[章节][ \t]+[^\n]+))+)\s*$"
)

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


def cn_to_int(label: str) -> int:
    """把「第一百零一条」这类条文数字转成 int。"""
    s = label.strip().replace("第", "").replace("条", "").replace("章", "").replace("节", "")

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


def int_to_cn(n: int) -> str:
    """把正整数转成条文里用的中文数字，例如 50 → 五十、101 → 一百零一。"""
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0 or n >= 1000:
        raise ValueError(f"无法转换为中文数字: {n!r}")

    digits = "零一二三四五六七八九"

    def below_100(value: int, *, leading_ten: bool = False) -> str:
        if value < 10:
            return digits[value]
        if value < 20:
            prefix = "一十" if leading_ten else "十"
            return prefix if value == 10 else prefix + digits[value - 10]
        tens, ones = divmod(value, 10)
        return digits[tens] + "十" + (digits[ones] if ones else "")

    if n < 100:
        return below_100(n)
    hundreds, rest = divmod(n, 100)
    head = digits[hundreds] + "百"
    if rest == 0:
        return head
    if rest < 10:
        return head + "零" + digits[rest]
    return head + below_100(rest, leading_ten=True)


def _normalize_header(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def split_trailing_structure(text: str) -> tuple[str, str | None]:
    """剥离末尾的章/节标题，供下一条使用。"""
    match = TRAILING_STRUCTURE_RE.match(text)
    if not match:
        return text.strip(), None
    body = match.group("body").strip()
    tail = match.group("tail")
    chapters = CHAPTER_HEADER_RE.findall(tail)
    chapter = _normalize_header(chapters[-1]) if chapters else None
    return body, chapter


def is_title_only(text: str) -> bool:
    """是否只剩章/节标题，没有条文正文。"""
    stripped = text.strip()
    if not stripped:
        return True
    remainder = STRUCTURAL_HEADER_RE.sub("", stripped)
    return not re.sub(r"\s+", "", remainder)


def process_legal_document(file_path: str):
    """
    加载并按条切分法律文档。

    每条保持完整；只丢掉纯章/节标题片段。前言（法名、通过日期、目录）单独成一块。
    过短的条文仍单独保留，不与邻条合并，避免 metadata 里的条号含混。

    Returns:
        切分后的 Document 列表。metadata 含 source（文件名）、article、chapter、chunk_id。
    """
    print("📄 开始加载文档...")
    loader = PyMuPDFLoader(file_path)
    pages = loader.load()

    print("🧹 正在清洗文档数据...")
    for page in pages:
        page.page_content = clean_legal_text(page.page_content)
    full_text = clean_legal_text("\n".join(page.page_content for page in pages))

    source_name = Path(file_path).name
    matches = list(ARTICLE_HEADER_RE.finditer(full_text))
    print(f"✂️ 按条切分（共识别 {len(matches)} 条）...")

    chunks: list[Document] = []
    current_chapter = ""

    preamble_raw = full_text[: matches[0].start()] if matches else full_text
    preamble, preamble_chapter = split_trailing_structure(preamble_raw)
    if preamble_chapter:
        current_chapter = preamble_chapter
    if preamble and not is_title_only(preamble):
        chunks.append(
            Document(
                page_content=preamble,
                metadata={
                    "source": source_name,
                    "article": "",
                    "chapter": "",
                    "chunk_id": "preamble",
                },
            )
        )

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body, next_chapter = split_trailing_structure(full_text[start:end])
        if not body or is_title_only(body):
            if next_chapter:
                current_chapter = next_chapter
            continue

        article = match.group(1)
        try:
            chunk_id = f"art-{cn_to_int(article):03d}"
        except ValueError:
            chunk_id = f"art-{article}"

        chunks.append(
            Document(
                page_content=body,
                metadata={
                    "source": source_name,
                    "article": article,
                    "chapter": current_chapter,
                    "chunk_id": chunk_id,
                },
            )
        )
        if next_chapter:
            current_chapter = next_chapter

    print(f"✅ 切分完成，共 {len(chunks)} 个片段")
    return chunks


if __name__ == "__main__":
    import sys

    default_pdf = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "中华人民共和国治安管理处罚法.pdf"
    )
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else str(default_pdf)

    if not Path(pdf_path).exists():
        print(f"文件不存在: {pdf_path}")
        sys.exit(1)

    chunks = process_legal_document(pdf_path)
    print(f"\n共 {len(chunks)} 个片段，前 3 个预览：")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"\n--- 片段 {i} metadata={chunk.metadata} ---")
        text = chunk.page_content
        print(text[:300] + "..." if len(text) > 300 else text)
