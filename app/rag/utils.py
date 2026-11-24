"""文本清洗工具"""
import re


def clean_legal_text(text: str) -> str:
    """清洗法律文档中的噪音，包括页码和残留的破折号页码。"""
    # 1. 统一处理特殊空格（如 \xa0 全角空格）
    text = text.replace("\xa0", " ")

    # 2. 成对页码标识：如 "- 1 -", "— 1 —", "-1-", "—1—"
    text = re.sub(r"[-—–]\s*\d+\s*[-—–]", "", text)

    # 3. 单侧残留页码：PDF 换页后常见「—10」或行首的「— 10」
    text = re.sub(r"(?m)^[ \t]*[—–-]\s*\d+[ \t]*$", "", text)
    text = re.sub(r"[—–-]\s*\d+[ \t]*(?=\n|$)", "", text)

    # 4. 压缩连续的空行（包括带有空格的空行）
    text = re.sub(r"\n\s*\n+", "\n", text)

    return text.strip()
