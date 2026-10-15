"""回答生成：把检索到的片段拼进 prompt，让 LLM 基于片段作答并给引用。

为什么强制"带引用编号"？
- 防幻觉：模型只能基于检索结果说话，不知道就说不知道；
- 可追溯：每条结论都能点回 JD 原文核对——这是 RAG 相对纯大模型的
  核心优势，也是面试演示时最能体现工程感的一环。

无 Key 时不报错、不返回空——走本地降级作答（见 _fallback_answer），
把检索到的岗位组织成可读清单，让产品在"零配置"状态下依然有产出。
"""

import requests

from .config import API_KEY, BASE_URL, LLM_MODEL, has_key

_SYSTEM_PROMPT = (
    "你是「Agent 岗位情报站」的求职顾问，负责根据真实岗位信息回答求职问题。\n"
    "回答规则：\n"
    "1. 只能依据用户提供的岗位片段回答，片段里没有的信息就说不知道，绝不编造；\n"
    "2. 每个具体结论末尾标注来源编号，格式如（来源[2]）；\n"
    "3. 用户问'我会xx适合什么岗'时，逐条对比技能要求与用户背景，指出匹配点与缺口；\n"
    "4. 用中文回答，先给结论，再给依据，语言简洁，不要复述整个 JD。"
)


def _fallback_answer(hits: list[dict]) -> str:
    """没有 Key 时的本地作答：把检索结果组织成岗位清单。

    诚实声明当前模式，同时给出真正有用的信息（岗位名/城市/相关度/链接），
    并引导用户配 Key 获得 AI 分析——把"降级"变成"转化引导"。
    """
    if not hits:
        return "库里没有检索到相关内容。请先运行 python -m crawler 采集岗位数据。"
    lines = ["当前是**本地关键词检索模式**（未配置 API Key，无法调用大模型做深度分析）。", ""]
    lines.append(f"按相关度为你找到 {len(hits)} 个岗位：")
    lines.append("")
    for i, h in enumerate(hits, 1):
        extra = "，".join(x for x in [h.get("location", ""), h.get("experience", "")] if x)
        lines.append(f"[{i}] {h['title']}（{extra}）相关度 {h['score']:.2f}")
        if h.get("url"):
            lines.append(f"    {h['url']}")
    lines.append("")
    lines.append("---")
    lines.append(
        "想要 AI 逐条对比你的技能与岗位要求？在项目根目录建 .env 写入 "
        "`SILICONFLOW_API_KEY=sk-xxx` 后重跑即可。"
    )
    return "\n".join(lines)


def ask(
    question: str,
    hits: list[dict],
) -> str:
    """hits: 检索结果，每个元素形如
    {"job_id": ..., "title": ..., "location": ..., "url": ..., "text": ...}

    片段按编号 [1][2]... 拼进 user prompt，模型回答必须引用这些编号。
    没有 Key 时走本地降级，而不是抛错——产品要能在零配置下可用。
    """
    if not hits:
        return "库里没有检索到相关内容。请先运行 python -m crawler 采集数据。"
    if not has_key():
        return _fallback_answer(hits)

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
