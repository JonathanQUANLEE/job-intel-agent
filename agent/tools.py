"""工具层：search_jobs / match_skills。

Function Calling 的核心思想：把"函数"的名称、说明、参数格式（JSON Schema）
告诉模型，模型用自然语言理解问题后，自己决定调哪个函数、传什么参数。
模型只输出"我要调 search_jobs，参数是 xxx"，真正执行的是我们的代码。

两个工具共享同一个检索底座（rag 包的 embed + 余弦检索），
但语义入口不同——这正是"底座复用、语义分层"的最小体现：
  search_jobs(question)  按问题语义找岗位（"有哪些 Agent 岗？"）
  match_skills(skills)   按用户技能画像找匹配岗位，附带技能差距对比提示
"""

import json
import sqlite3
from contextlib import closing

from rag.embedder import embed_texts
from rag.store import search

# 暴露给模型的工具清单（OpenAI tools 格式）。description 是模型选工具的唯一依据，
# 写得越准，模型选择越靠谱——这就是 Prompt Engineering 落到工具层的形态。
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "在岗位库里按自然语言问题检索岗位。当用户询问有哪些岗位、某方向/城市的岗位时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "用于检索的问题或描述，如'深圳的 Agent 开发岗位'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回岗位数量，默认 5，最多 10",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "match_skills",
            "description": "用户描述自己的技能背景，找出与其技能最匹配的岗位，并对比技能差距。当用户问'我会xx，适合什么岗位'时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "skills": {
                        "type": "string",
                        "description": "用户的技能描述，如'熟悉 PyTorch 和分布式训练'",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回岗位数量，默认 5，最多 10",
                    },
                },
                "required": ["skills"],
            },
        },
    },
]


def _job_meta(db_path: str, job_ids: list[str]) -> dict[str, dict]:
    """job_id → 标题/地点/链接。工具输出要带这些展示字段，回答才有引用价值。"""
    if not job_ids:
        return {}
    placeholders = ", ".join("?" * len(job_ids))
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            f"SELECT job_id, title, location, url FROM jobs WHERE job_id IN ({placeholders})",
            job_ids,
        ).fetchall()
    return {r[0]: {"title": r[1], "location": r[2], "url": r[3]} for r in rows}


def _dedupe_by_job(raw_hits: list[tuple], top_k: int) -> list[tuple]:
    """同岗位命中多片段时只留相似度最高的一段（与 rag CLI 同策略，保证多样性）。"""
    best: dict[str, tuple] = {}
    for score, job_id, text in raw_hits:
        if job_id not in best or score > best[job_id][0]:
            best[job_id] = (score, job_id, text)
    ordered = sorted(best.values(), key=lambda h: h[0], reverse=True)
    return ordered[:top_k]


def _format_hits(hits: list[tuple], db_path: str) -> str:
    """把检索结果整理成模型易读的文本。带编号，方便模型在回答里引用来源。"""
    meta = _job_meta(db_path, [job_id for _, job_id, _ in hits])
    parts = []
    for i, (score, job_id, text) in enumerate(hits, 1):
        m = meta.get(job_id, {})
        parts.append(
            f"[{i}] {m.get('title', job_id)}（{m.get('location', '未知')}，编号 {job_id}）\n"
            f"相关度 {score:.3f}\n{text}"
        )
    return "\n\n".join(parts) if parts else "没有检索到相关岗位。"


def search_jobs(question: str, top_k: int = 5, db_path: str = "jobs.db") -> str:
    """按问题语义检索岗位。失败时返回错误文本而不是抛异常——
    工具失败让模型"看见"，它才能决定换关键词重查还是直接告诉用户，
    这就是工具失败降级（简历里"工具失败降级"的实体）。"""
    try:
        top_k = max(1, min(int(top_k), 10))
        vec = embed_texts([question])[0]
        hits = _dedupe_by_job(search(vec, db_path, top_k=top_k * 2), top_k)
        return _format_hits(hits, db_path)
    except Exception as e:  # noqa: BLE001 —— 边界在这里：工具错误要反馈给模型
        return f"工具执行失败：{type(e).__name__}: {e}。请尝试换个说法再查，或直接告知用户检索暂不可用。"


def match_skills(skills: str, top_k: int = 5, db_path: str = "jobs.db") -> str:
    """按用户技能画像检索最匹配的岗位片段，并引导模型做技能差距对比。
    与 search_jobs 的区别：入口语义是"人"而不是"问题"，
    返回里明确要求模型逐条对比岗位要求与用户已有技能。"""
    try:
        top_k = max(1, min(int(top_k), 10))
        vec = embed_texts([skills])[0]
        hits = _dedupe_by_job(search(vec, db_path, top_k=top_k * 2), top_k)
        body = _format_hits(hits, db_path)
        return (
            "以下是与用户技能最相关的岗位片段。请逐条对比岗位要求与用户已有技能，"
            "指出：匹配点、缺失技能、建议补齐的方向。\n\n" + body
        )
    except Exception as e:  # noqa: BLE001
        return f"工具执行失败：{type(e).__name__}: {e}。请尝试换个说法再查，或直接告知用户检索暂不可用。"


# 函数名 → 实现。graph.py 的工具节点按这个表分发（模型只会给名字和参数）。
TOOL_REGISTRY = {"search_jobs": search_jobs, "match_skills": match_skills}


def execute_tool_call(name: str, arguments_json: str, db_path: str = "jobs.db") -> str:
    """执行一次工具调用。模型给的 arguments 是 JSON 字符串，要自己解析——
    解析失败同样降级为文本反馈，而不是让整个循环崩掉。"""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"未知工具：{name}。可用工具：{', '.join(TOOL_REGISTRY)}"
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as e:
        return f"工具参数不是合法 JSON：{e}。请重新调用并给出正确参数。"
    return fn(db_path=db_path, **args)
