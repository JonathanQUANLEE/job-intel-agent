"""回答生成：把检索到的片段拼进 prompt，让 LLM 基于片段作答并给引用。

为什么强制"带引用编号"？
- 防幻觉：模型只能基于检索结果说话，不知道就说不知道；
- 可追溯：每条结论都能点回 JD 原文核对——这是 RAG 相对纯大模型的
  核心优势，也是面试演示时最能体现工程感的一环。
"""

import requests

from .config import API_KEY, BASE_URL, LLM_MODEL

_SYSTEM_PROMPT = (
    "你是「Agent 岗位情报站」的求职顾问，负责根据真实岗位信息回答求职问题。\n"
    "回答规则：\n"
    "1. 只能依据用户提供的岗位片段回答，片段里没有的信息就说不知道，绝不编造；\n"
    "2. 每个具体结论末尾标注来源编号，格式如（来源[2]）；\n"
    "3. 用中文回答，先给结论，再给依据，语言简洁。"
)


def ask(
    question: str,
    hits: list[dict],
) -> str:
    """hits: 检索结果，每个元素形如
    {"job_id": ..., "title": ..., "location": ..., "url": ..., "text": ...}

    片段按编号 [1][2]... 拼进 user prompt，模型回答必须引用这些编号。
    """
    if not API_KEY:
        raise RuntimeError("没有找到 SILICONFLOW_API_KEY（看 config.py 的提示）")
    if not hits:
        return "库里没有检索到相关内容。请先运行 python -m rag --index 建立索引。"

    snippets = "\n\n".join(
        f"[{i + 1}] 岗位：{h['title']}（{h['location']}，{h['job_id']}）\n{h['text']}"
        for i, h in enumerate(hits)
    )
    user_content = f"用户问题：{question}\n\n以下是检索到的真实岗位片段，请据此回答：\n{snippets}"

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,  # 检索问答任务求稳，温度压低减少胡说
        },
        timeout=(5, 120),
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
