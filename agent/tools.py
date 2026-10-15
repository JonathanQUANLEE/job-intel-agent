"""工具层：search_jobs / match_skills。

Function Calling 的核心思想：把"函数"的名称、说明、参数格式（JSON Schema）
告诉模型，模型用自然语言理解问题后，自己决定调哪个函数、传什么参数。
模型只输出"我要调 search_jobs，参数是 xxx"，真正执行的是我们的代码。

两个工具共享同一个检索底座（rag.retriever 的统一检索入口），
但语义入口不同——这正是"底座复用、语义分层"的最小体现：
  search_jobs(question)  按问题语义找岗位（"有哪些 Agent 岗？"）
  match_skills(skills)   按用户技能画像找匹配岗位，附带技能差距对比提示

底座换成 retriever 之后，Agent 自动获得了降级能力：
有 Key 走向量，没 Key 走 BM25，工具链不会因为缺 Key 而整体失效。
"""

import json

from rag.retriever import MODE_OFFLINE, retrieve

# 暴露给模型的工具清单（OpenAI tools 格式）。description 是模型选工具的唯一依据，
# 写得越准，模型选择越靠谱——这就是 Prompt Engineering 落到工具层的形态。
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": (
                "在岗位库里按自然语言问题检索岗位。当用户询问有哪些岗位、"
                "某方向/某城市有哪些岗位时使用。例如'深圳的 Agent 开发岗位'。"
            ),
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
            "description": (
                "用户描述自己的技能背景，找出与其技能最匹配的岗位，并对比技能差距。"
                "当用户问'我会xx，适合什么岗位'时使用。"
            ),
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


def _format_hits(hits: list[dict]) -> str:
    """把统一检索结果整理成模型易读、可引用的文本。

    每条都带：编号、标题、城市、事业群/经验、job_id（模型回答里引用它）、相关度、正文片段。
    job_id 一定要给——模型引用它，用户才能拿这个编号点回原文核对。
    """
    if not hits:
        return "没有检索到相关岗位。"
    parts = []
    for i, h in enumerate(hits, 1):
        extra = "，".join(x for x in [h.get("department", ""), h.get("experience", "")] if x)
        head = f"[{i}] {h['title']}（{h['location']}"
        head += f"，{extra}" if extra else ""
        head += f"，编号 {h['job_id']}）\n相关度 {h['score']:.3f}\n{h['text']}"
        parts.append(head)
    return "\n\n".join(parts)


def search_jobs(question: str, top_k: int = 5, db_path: str = "jobs.db") -> str:
    """按问题语义检索岗位。失败时返回错误文本而不是抛异常——
    工具失败让模型"看见"，它才能决定换关键词重查还是直接告诉用户，
    这就是工具失败降级（简历里"工具失败降级"的实体）。"""
    try:
        top_k = max(1, min(int(top_k), 10))
        hits = retrieve(question, db_path, top_k=top_k)
        mode_hint = "（当前为本地关键词检索模式）" if (hits and hits[0]["mode"] == MODE_OFFLINE) else ""
        return mode_hint + _format_hits(hits)
    except Exception as e:  # noqa: BLE001 —— 边界在这里：工具错误要反馈给模型
        return f"工具执行失败：{type(e).__name__}: {e}。请尝试换个说法再查，或直接告知用户检索暂不可用。"


def match_skills(skills: str, top_k: int = 5, db_path: str = "jobs.db") -> str:
    """按用户技能画像检索最匹配的岗位片段，并引导模型做技能差距对比。
    与 search_jobs 的区别：入口语义是"人"而不是"问题"，
    返回里明确要求模型逐条对比岗位要求与用户已有技能。"""
    try:
        top_k = max(1, min(int(top_k), 10))
        hits = retrieve(skills, db_path, top_k=top_k)
        body = _format_hits(hits)
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
