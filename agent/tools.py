"""工具层：择岗顾问的"专业动作"。

这层回答"这个 agent 除了检索还能干什么"。通用编程 agent（Codex / OpenCode 那类）
强在"给我代码我改"，它们的工具是文件读写、命令执行；而择岗这件专业事需要的是
另一套动作：算匹配度、查技能差距、按硬条件过滤、对比两个岗位。工具集不同，
agent 的专业性边界就不同——这就是"场景化"落在代码上的样子。

工具分两组：
  A. 确定性工具（被图直接调用，不经过模型）
     score_and_rank / skill_gap / filter_by_conditions / compare_jobs
     这些是"决策计算"，必须结果稳定可复现，所以写死在代码里、图节点直接调。
  B. Function Calling 工具（暴露给模型，模型自己决定调不调）
     search_jobs / match_skills —— 保留 OpenAI tools 格式，
     用于"用户追问细节"的场景（如"再帮我看看深圳的"），模型按需触发。

为什么两组并存而不是全丢给模型？
择岗的核心结论（谁排第一、差哪项技能）不能让模型自由裁量——
它今天说 A 好明天说 B 好，产品就没法用。所以把"判断"沉淀成确定性代码，
把"表达"和"长尾意图理解"交给模型。这个分工是这一版最重要的设计取向。
"""

import json

from rag.retriever import MODE_OFFLINE, retrieve
from .scoring import rank_jobs, score_job, skill_gap_report

# —— A. 确定性工具：图节点直接调用，结果可复现 ——

def score_and_rank(query: str, profile, db_path: str = "jobs.db", top_k: int = 8) -> list[tuple]:
    """检索 + 打分排序，返回 [(job, Score)]。图里 evaluate 节点的核心动作。"""
    jobs = retrieve(query, db_path, top_k=15)
    return rank_jobs(jobs, profile, top_k=top_k)


def skill_gap(query: str, profile, db_path: str = "jobs.db") -> dict:
    """跨岗位汇总技能差距。compose 节点用它给"该补什么"的建议。"""
    jobs = retrieve(query, db_path, top_k=15)
    return skill_gap_report(jobs, profile)


def filter_by_conditions(jobs: list[dict], city: str = "", experience: str = "") -> list[dict]:
    """按硬条件过滤岗位。用户明确说"只看北京"或"要应届岗"时用。

    过滤放在打分之前：先去掉明显不满足硬条件的，避免它们占满推荐位。
    """
    out = jobs
    if city and city != "不限":
        out = [j for j in out if city in (j.get("location") or "")]
    if experience:
        # 经验是"不限"或未标注的岗位不筛掉——它们对用户仍然是机会
        out = [
            j for j in out
            if experience in (j.get("experience") or "")
            or not (j.get("experience") or "").strip()
            or "不限" in (j.get("experience") or "")
        ]
    return out


def compare_jobs(job_a: dict, job_b: dict, profile) -> dict:
    """对比两个岗位的评分明细，返回逐维度的差异。用户纠结"选哪个"时用。"""
    sa, sb = score_job(job_a, profile), score_job(job_b, profile)
    return {
        "a": {"job_id": job_a.get("job_id"), "title": job_a.get("title"), "total": sa.total},
        "b": {"job_id": job_b.get("job_id"), "title": job_b.get("title"), "total": sb.total},
        "winner": job_a.get("job_id") if sa.total >= sb.total else job_b.get("job_id"),
        "diff": {
            "技能": {"a": sa.matched_skills, "b": sb.matched_skills},
            "待补": {"a": sa.missing_skills, "b": sb.missing_skills},
        },
    }


# —— B. Function Calling 工具：暴露给模型，用于长尾追问 ——

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": (
                "在岗位库里按自然语言问题检索岗位。当用户追问'再找找xx方向的岗位'"
                "或需要补充检索时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "检索用的问题或描述"},
                    "top_k": {"type": "integer", "description": "返回数量，默认 5，最多 10"},
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
                "用户描述技能背景，找出最匹配的岗位并对比技能差距。"
                "当用户问'我会xx，适合什么岗'时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skills": {"type": "string", "description": "用户的技能描述"},
                    "top_k": {"type": "integer", "description": "返回数量，默认 5，最多 10"},
                },
                "required": ["skills"],
            },
        },
    },
]


def _format_hits(hits: list[dict]) -> str:
    """把检索结果整理成模型易读、可引用的文本。job_id 必须给——用户靠它回查原文。"""
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
    """按问题语义检索岗位。失败返回错误文本而非抛异常——
    工具失败让模型"看见"，它才能决定换关键词重查还是如实告诉用户。"""
    try:
        top_k = max(1, min(int(top_k), 10))
        hits = retrieve(question, db_path, top_k=top_k)
        mode_hint = "（当前为本地关键词检索模式）" if (hits and hits[0]["mode"] == MODE_OFFLINE) else ""
        return mode_hint + _format_hits(hits)
    except Exception as e:  # noqa: BLE001 —— 边界在这里：工具错误要反馈给模型
        return f"工具执行失败：{type(e).__name__}: {e}。请尝试换个说法再查，或直接告知用户检索暂不可用。"


def match_skills(skills: str, top_k: int = 5, db_path: str = "jobs.db") -> str:
    """按技能画像检索岗位。入口语义是"人"而不是"问题"。"""
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


# 函数名 → 实现。模型只给名字和参数，真正执行的是这里。
TOOL_REGISTRY = {"search_jobs": search_jobs, "match_skills": match_skills}


def execute_tool_call(name: str, arguments_json: str, db_path: str = "jobs.db") -> str:
    """执行一次工具调用。arguments 是 JSON 字符串——
    解析失败同样降级为文本反馈，而不是让整个流程崩掉。"""
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"未知工具：{name}。可用工具：{', '.join(TOOL_REGISTRY)}"
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as e:
        return f"工具参数不是合法 JSON：{e}。请重新调用并给出正确参数。"
    return fn(db_path=db_path, **args)
