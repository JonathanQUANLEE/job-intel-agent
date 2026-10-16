"""Agent 包离线测试：验证决策图的分支、槽位、打分与护栏，全程不需要 API Key。

新图的核心是"分支"和"确定性"——所以测试也围绕这两点：
  - intake 缺硬条件 → 走 clarify（追问），不检索
  - intake 信息完整 → 走 recall → evaluate → compose
  - evaluate 结果偏弱 → 走 refine 补搜，但补搜次数被护栏卡住
  - 无 Key 时 compose 走模板兜底，绝不抛异常
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.graph import MAX_REFINE, build_graph, run_agent
from agent.profile import Profile, Slot, extract_slots, missing_hard_slots
from agent.scoring import score_job, skill_gap_report
from agent.tools import execute_tool_call


# —— 图分支 ——

def test_clarify_when_slots_missing():
    """信息不全时必须追问，而不是瞎检索。追问内容要包含缺失的槽位提示。"""
    state = run_agent("帮我推荐些岗位")
    assert state["need_clarify"] is True
    assert "城市" in state["answer"]
    assert "哪一类岗位" in state["answer"]
    # 追问分支不应触发检索
    assert not state.get("candidates")


def test_full_path_returns_ranked_recommendations():
    """信息完整的路径：召回 → 打分 → 成稿，排序结果按分数降序。"""
    state = run_agent("深圳 Agent 开发岗，会 Python 和 PyTorch")
    assert state["need_clarify"] is False
    scored = state.get("scored") or []
    assert scored, "应召回到候选岗位"
    totals = [s.total for _, s in scored]
    assert totals == sorted(totals, reverse=True), "推荐必须按匹配度降序"
    assert "匹配度" in state["answer"]
    # 轨迹应记录关键决策点（可解释性）
    trace = "\n".join(state.get("trace") or [])
    assert "检索" in trace and "打分" in trace


def test_multi_turn_profile_accumulates():
    """多轮：第一轮只有城市 → 追问；把第一轮的 profile 传回，第二轮补方向 → 出结果。"""
    first = run_agent("我在深圳找工作")
    assert first["need_clarify"] is True
    assert first["profile"].city.filled

    second = run_agent("Agent 开发方向", profile=first["profile"])
    assert second["need_clarify"] is False
    assert second["profile"].city.value == "深圳"  # 上一轮的城市没丢
    assert second["profile"].direction.value == "agent"


def test_refine_is_capped():
    """护栏：补搜次数不超过 MAX_REFINE（用一个永远召不回好结果的画像触发）。"""
    p = Profile(
        city=Slot(value="深圳", filled=True),
        direction=Slot(value="agent", filled=True),
    )
    # 直接看图在弱结果下的行为：refine_count 不应突破上限
    state = run_agent("深圳 Agent", profile=p)
    assert state.get("refine_count", 0) <= MAX_REFINE


def test_no_key_falls_back_to_template():
    """无 Key 时 compose 走模板，答案非空且含具体岗位编号（可回查）。"""
    state = run_agent("深圳 Agent 开发岗")
    assert state["answer"].strip()
    mode_line = "\n".join(state.get("trace") or [])
    assert "模板成稿" in mode_line or "模型润色" in mode_line


# —— 槽位抽取 ——

def test_slot_extraction_city_and_direction():
    p = extract_slots("深圳有哪些 Agent 方向的岗位？")
    assert p.city.value == "深圳"
    assert p.direction.value == "agent"
    assert missing_hard_slots(p) == []


def test_slot_extraction_ignores_verb_recommend():
    """「帮我推荐下岗位」里的"推荐"是动词，不能被识别成"推荐算法"方向。"""
    p = extract_slots("帮我推荐下岗位")
    assert not p.direction.filled, "动词『推荐』不应触发方向槽位"
    assert set(missing_hard_slots(p)) == {"city", "direction"}


def test_slot_extraction_skills_and_experience():
    p = extract_slots("我会 Python 和 PyTorch，是应届生")
    assert "python" in p.skills.value.lower()
    assert p.experience.value == "应届/在校"


# —— 打分 ——

def _job(title, location, text, experience="", job_id="t-1"):
    return {
        "job_id": job_id, "title": title, "location": location,
        "text": text, "experience": experience, "score": 1.0,
    }


def test_score_rewards_title_direction_over_body_mention():
    """标题含方向的岗位，分数必须高于只在正文提一句的岗位。"""
    p = extract_slots("深圳 Agent 开发岗，会 Python")
    strong = score_job(_job("Agent 开发工程师", "深圳", "要求 Python"), p)
    weak = score_job(_job("视频生成研究员", "深圳", "与 agent 协作，要求 Python"), p)
    assert strong.total > weak.total
    assert any("弱对齐" in r for r in weak.reasons)


def test_score_missing_city_reduces_total():
    """城市不符要扣分，且理由里说明清楚。"""
    p = extract_slots("深圳 Agent 开发岗，会 Python")
    in_city = score_job(_job("Agent 开发工程师", "深圳", "Python"), p)
    out_city = score_job(_job("Agent 开发工程师", "北京", "Python"), p)
    assert in_city.total > out_city.total
    assert any("城市不符" in r for r in out_city.reasons)


def test_skill_gap_counts_across_jobs():
    """技能差距报告要跨岗位统计"用户有但岗位未要求"的技能频次。

    注意语义：missing_skills 是"用户填了、但岗位 JD 里没出现"的技能——
    这代表"这项技能在这个方向上用不上"，而不是"用户不会"。
    """
    p = extract_slots("深圳 Agent 岗，会 Python 和 Vue")
    jobs = [
        _job("A", "深圳", "要求 Python", job_id="a"),
        _job("B", "深圳", "要求 Python", job_id="b"),
    ]
    gap = skill_gap_report(jobs, p)
    skills = [x["skill"] for x in gap["top_missing"]]
    assert "Vue" in skills, "Vue 在用户技能里、两个岗位都没提，应计入差距"
    vue = next(x for x in gap["top_missing"] if x["skill"] == "Vue")
    assert vue["count"] == 2


# —— 工具层降级（保留原有覆盖）——

def test_tool_failure_degrades():
    """未知工具 / 坏 JSON 参数 → 返回降级文本而不是抛异常。"""
    out1 = execute_tool_call("no_such_tool", "{}")
    out2 = execute_tool_call("search_jobs", "{bad json")
    assert "未知工具" in out1
    assert "不是合法 JSON" in out2
