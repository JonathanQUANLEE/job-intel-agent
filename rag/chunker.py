"""切分模块：把岗位描述切成适合检索的片段（chunk）。

切分策略直接决定检索质量，是 RAG 的第一决定因素：
- 切太大：一个块里塞满无关信息，检索命中但不精准，还浪费上下文窗口；
- 切太小：语义被切碎，检索到半句话，回答缺上下文；
- 硬按固定字符数切：可能把一句话从中间拦腰砍断。

策略（借鉴开源项目 sre-bible 的实践）：
  1. 先按句子自然边界切（中文靠 。！？； 和换行，英文靠句号）；
  2. 再贪心合并成块：凑够目标长度就成块，绝不硬切。
"""

import re

TARGET_CHUNK = 900   # 目标块长（字符）。约等于一段 JD 核心要求，检索粒度正好
MIN_CHUNK = 450      # 低于这个长度不单独成块，并到上一块，避免产生垃圾块

# 中文没有空格分词，句子边界只能靠标点。用 lookbehind 保留标点本身
_SENTENCE_END = re.compile(r"(?<=[。！？；；\n])")


def chunk_text(text: str) -> list[str]:
    """把一个岗位描述切成若干片段。空文本返回空列表。

    两步走：
    1. split 出最小语义单元（以标点/换行结尾的一句话或一行）；
    2. 按 TARGET_CHUNK 贪心合并。
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []

    units = [u.strip() for u in _SENTENCE_END.split(text) if u.strip()]

    chunks: list[str] = []
    buf = ""
    for unit in units:
        buf += unit
        if len(buf) >= TARGET_CHUNK:
            chunks.append(buf)
            buf = ""

    # 收尾：够长就成块，不够长并进上一块（或单独成块）
    if buf:
        if len(buf) >= MIN_CHUNK:
            chunks.append(buf)
        elif chunks:
            chunks[-1] += buf
        else:
            chunks.append(buf)
    return chunks
