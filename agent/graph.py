"""LangGraph 编排：把"择岗"画成一张有判断力的图。

旧版是简单的 ReAct 循环（chat ⇄ tools），单轮问答够用，但撑不起"择岗顾问"——
顾问的价值在于**先问清楚、再给排序、还能说清为什么**，这是有状态的决策流程，
不是一次检索能覆盖的。

新版图结构：

    START → intake ──┬─缺硬条件──→ clarify → END（追问，等用户下一轮）
                     │
                     └─信息够──→ recall → evaluate ──┬─信息不足──→ refine → evaluate（补搜循环）
                                                     │
                                                     └─信息够──→ compose → END

各节点职责（每个都是"可独立测试的小函数"，这才叫编排而不是一坨脚本）：
  intake    抽槽位 + 并入会话里累积的画像，判断硬条件够不够
  clarify   缺条件就生成一轮追问（确定性话术，不烧模型）
  recall    用画像拼检索查询，召回候选岗位
  evaluate  对候选逐条打分排序，同时判断"信息是否足够下结论"
  refine    信息不足时换更宽的查询补搜一次（有次数上限，防死循环）
  compose   把排序结果写成给人看的推荐（有 Key 走模型润色，没 Key 走模板）

为什么 evaluate 既能打分又能判断"够不够"？
打分需要候选，而"候选够不够/像不像"的判断跟打分用的是同一批数据——
放在一个节点里避免重复计算，也让"要不要再搜一轮"这个决策有明确依据。

护栏（生产级 agent 必须有的东西）：
  - refine 最多补搜 MAX_REFINE 次，超了就直接 compose（不能无限重查烧钱）
  - compose 全部走"模板兜底"：模型挂了/没 Key 也能出结果，绝不白屏
"""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .profile import (
    Profile,
    clarify_question,
    extract_slots,
    missing_hard_slots,
)
from .scoring import skill_gap_report
MAX_REFINE = 2  # 补搜上限：防"查不完"死循环，也是成本护栏
MIN_GOOD_HITS = 3  # 打分后达到"较匹配"以上的条数，少于它则考虑再搜一轮
GOOD_GRADE_THRESHOLD = 60  # 与 scoring.grade 的"较匹配"档对齐


class AgentState(TypedDict, total=False):
    """图里流动的状态。total=False 让各节点只声明自己写的那几个键。"""

    question: str  # 本轮用户输入
    profile: Profile  # 累积的择岗画像（跨轮保留）
    messages: list[dict]  # 全部对话历史，喂给模型润色时用
    candidates: list[dict]  # 检索到的候选岗位
    scored: list[tuple]  # [(job, Score)] 打分排序结果
    gap: dict  # 技能差距汇总
    need_clarify: bool  # 是否进入追问分支
    clarify_text: str  # 追问内容
    refine_count: int  # 已补搜次数
    answer: str  # 最终回答
    trace: list[str]  # 决策轨迹（给前端展示"它想了什么"，也便于调试）


def _trace(state: AgentState, msg: str) -> list[str]:
    """往轨迹里追加一行。轨迹是 agent"可解释性"的载体——面试时能说清每一步在干嘛。"""
    return list(state.get("trace") or []) + [msg]


def build_graph(db_path: str = "jobs.db") -> Any:
    """编译择岗顾问图。db_path 闭包进来，节点不用层层传参。"""
    from . import llm_client

    builder = StateGraph(AgentState)

    # —— 节点 1：ingest，抽槽位并入画像，判断信息是否足够 ——
    def intake_node(state: AgentState) -> dict:
        fresh = extract_slots(state["question"])
        profile: Profile = state.get("profile") or Profile()
        profile.merge(fresh)  # 本轮抽到的覆盖累积画像
        missing = missing_hard_slots(profile)
        if missing:
            return {
                "profile": profile,
                "need_clarify": True,
                "clarify_text": clarify_question(missing),
                "trace": _trace(
                    state, f"识别到缺失硬条件：{'、'.join(missing)}，先追问"
                ),
            }
        return {
            "profile": profile,
            "need_clarify": False,
            "trace": _trace(
                state, f"信息充足，画像：{profile.as_dict()}"
            ),
        }

    # —— 节点 2：recall，用画像拼查询召回候选 ——
    def recall_node(state: AgentState) -> dict:
        from rag.retriever import retrieve

        p: Profile = state["profile"]
        # 查询词按"方向 + 城市 + 技能"拼——把结构化槽位转回检索能用的自然语言
        parts = [p.direction.value if p.direction.filled else ""]
        parts.append(p.city.value if p.city.filled and p.city.value != "不限" else "")
        if p.skills.filled:
            parts.append(p.skills.value)
        query = " ".join(x for x in parts if x) or state["question"]

        jobs = retrieve(query, db_path, top_k=15)
        return {
            "candidates": jobs,
            "trace": _trace(
                state, f"检索「{query}」→ 召回 {len(jobs)} 条候选"
            ),
        }

    # —— 节点 3：evaluate，打分排序 + 判断信息是否足够 ——
    def evaluate_node(state: AgentState) -> dict:
        from .tools import score_and_rank

        p: Profile = state["profile"]
        jobs = state.get("candidates") or []
        if not jobs:
            return {
                "scored": [],
                "gap": {},
                "trace": _trace(state, "无候选岗位，无法打分"),
            }
        # 走工具层的确定性打分（与 CLI / 评测共用同一实现，数字才有可比性）
        ranked = score_and_rank(state["question"], p, db_path, top_k=8)
        gap = skill_gap_report(jobs, p)
        good = [t for t in ranked if t[1].total >= GOOD_GRADE_THRESHOLD]
        top_line = ""
        if ranked:
            best = ranked[0]
            top_line = f"，最高分 {best[1].total}（{best[0]['title'][:20]}）"
        return {
            "scored": ranked,
            "gap": gap,
            "trace": _trace(
                state,
                f"打分排序完成：{len(good)} 条达「较匹配」以上（阈值 {GOOD_GRADE_THRESHOLD}）{top_line}",
            ),
        }

    # —— 节点 4：refine，信息不足时放宽条件补搜 ——
    def refine_node(state: AgentState) -> dict:
        from rag.retriever import retrieve
        from .tools import filter_by_conditions

        p: Profile = state["profile"]
        # 补搜策略：先按硬条件过滤掉明显不符的（城市不符的岗位不该占推荐位），
        # 再放宽城市约束重搜一次——两步都用确定性工具，结果可复现。
        prev = state.get("candidates") or []
        filtered = filter_by_conditions(prev, city=p.city.value if p.city.filled else "")
        if filtered and len(filtered) < len(prev):
            n = state.get("refine_count", 0) + 1
            return {
                "candidates": filtered,
                "refine_count": n,
                "trace": _trace(
                    state,
                    f"按城市硬条件过滤：{len(prev)} → {len(filtered)} 条（去掉城市不符的）",
                ),
            }

        wide_parts = [p.direction.value if p.direction.filled else ""]
        if p.skills.filled:
            wide_parts.append(p.skills.value)
        query = " ".join(x for x in wide_parts if x) or state["question"]

        seen = {j["job_id"] for j in prev}
        fresh = [j for j in retrieve(query, db_path, top_k=15) if j["job_id"] not in seen]
        merged = prev + fresh
        n = state.get("refine_count", 0) + 1
        return {
            "candidates": merged,
            "refine_count": n,
            "trace": _trace(
                state,
                f"结果偏弱，第 {n} 次放宽条件补搜「{query}」→ 新增 {len(fresh)} 条",
            ),
        }

    # —— 节点 5：compose，成稿（有 Key 润色，无 Key 模板）——
    def compose_node(state: AgentState) -> dict:
        p: Profile = state["profile"]
        scored = state.get("scored") or []
        gap = state.get("gap") or {}

        # 先拼一份确定性的"骨架"文本，任何情况下都能交付
        base = _compose_template(p, scored, gap)

        # 有 Key 且确实有结果时，让模型基于骨架润色（不改变事实，只调语气和组织）
        if llm_client.has_key() and scored:
            try:
                polished = llm_client.polish_answer(
                    state["question"], base, state.get("messages") or []
                )
                if polished.strip():
                    return {
                        "answer": polished,
                        "trace": _trace(state, "模型润色完成"),
                    }
            except Exception as e:  # noqa: BLE001 —— 润色失败就用骨架，绝不白屏
                return {
                    "answer": base,
                    "trace": _trace(state, f"模型润色失败（{type(e).__name__}），改用模板成稿"),
                }

        return {
            "answer": base,
            "trace": _trace(state, "模板成稿（离线模式或无需润色）"),
        }

    # —— 条件边：intake 之后判断走追问还是继续 ——
    def route_after_intake(state: AgentState) -> str:
        return "clarify" if state.get("need_clarify") else "recall"

    # —— 条件边：evaluate 之后判断要不要补搜 ——
    def route_after_evaluate(state: AgentState) -> str:
        scored = state.get("scored") or []
        good = sum(1 for _, s in scored if s.total >= GOOD_GRADE_THRESHOLD)
        if good >= MIN_GOOD_HITS:
            return "compose"
        if state.get("refine_count", 0) >= MAX_REFINE:
            return "compose"  # 护栏：补搜次数耗尽，不再纠缠
        if not state.get("candidates"):
            return "compose"  # 一条都没有，补搜大概率也白搭
        return "refine"

    builder.add_node("intake", intake_node)
    builder.add_node("clarify", lambda s: {"answer": s["clarify_text"]})
    builder.add_node("recall", recall_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("refine", refine_node)
    builder.add_node("compose", compose_node)

    builder.add_edge(START, "intake")
    builder.add_conditional_edges("intake", route_after_intake, ["clarify", "recall"])
    builder.add_edge("clarify", END)
    builder.add_edge("recall", "evaluate")
    builder.add_conditional_edges(
        "evaluate", route_after_evaluate, ["refine", "compose"]
    )
    builder.add_edge("refine", "evaluate")  # 补搜后重新打分
    builder.add_edge("compose", END)

    return builder.compile()


def _compose_template(p: Profile, scored: list[tuple], gap: dict) -> str:
    """确定性成稿：不依赖任何模型，保证离线/故障时也有可交付内容。

    结构：一句话结论 → 推荐清单（带分数和理由）→ 技能差距建议。
    这个模板本身就是"评测基线"——模型润色后的输出不应偏离它的事实。
    """
    if not scored:
        return (
            "没有检索到符合条件的岗位。可以试试放宽城市限制、"
            "换一个方向关键词，或先运行采集脚本补充岗位数据。"
        )

    head_bits = []
    if p.direction.filled:
        head_bits.append(p.direction.value)
    if p.city.filled and p.city.value != "不限":
        head_bits.append(p.city.value)
    head = "、".join(head_bits) or "你关心的方向"

    lines = [f"按「{head}」为你排了 {len(scored)} 个岗位，按匹配度从高到低："]
    for i, (job, s) in enumerate(scored, 1):
        lines.append(
            f"\n{i}. {job['title']}｜{job['location']}｜匹配度 {s.total}（{s.grade()}）"
        )
        lines.append(f"   编号 {job['job_id']}")
        if s.matched_skills:
            lines.append(f"   ✅ 命中技能：{'、'.join(s.matched_skills)}")
        if s.missing_skills:
            lines.append(f"   ⚠️ 待补技能：{'、'.join(s.missing_skills)}")
        lines.append(f"   依据：{'；'.join(s.reasons[:2])}")

    top_missing = (gap or {}).get("top_missing") or []
    if top_missing:
        items = "、".join(f"{x['skill']}（{x['count']} 个岗位要求）" for x in top_missing[:3])
        lines.append(f"\n📌 最该补齐的技能：{items}")
        lines.append("（出现频次越高越接近通用门槛，建议优先补。）")

    lines.append("\n想细看哪个岗位，把编号发我。")
    return "\n".join(lines)


def run_agent(
    question: str,
    db_path: str = "jobs.db",
    profile: Profile | None = None,
    messages: list[dict] | None = None,
) -> AgentState:
    """跑一轮择岗，返回完整 state。

    profile / messages 可传入——这是"多轮"的关键：
    追问之后用户再发一句，调用方把上一轮的 profile 和对话历史传回来，
    画像得以累积（用户第一轮说"深圳"，第二轮说"Agent"，合起来才够）。
    """
    graph = build_graph(db_path=db_path)
    init: AgentState = {
        "question": question,
        "profile": profile or Profile(),
        "messages": (messages or []) + [{"role": "user", "content": question}],
        "refine_count": 0,
        "trace": [],
    }
    return graph.invoke(init)
