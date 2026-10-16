"""决策质量评测：验证择岗顾问的"行为"而不只是"召回"。

和 run_eval.py 的分工：
  run_eval.py     → 召回评测：该找到的岗位找到了吗（hit@k / MRR）
  run_decision.py → 决策评测：agent 的行为对不对（该追问时追问了吗、
                    条件继承了吗、推荐排序合理吗、轨迹完整吗）

为什么需要这一层？
召回高不等于体验好。一个"召回 100% 但从不追问、把北京岗位推给深圳用户"
的 agent，召回指标一样漂亮，却完全没法用。决策评测把"行为契约"写成
确定性断言，任何一次改动（改词表、改打分权重、改图结构）都能立刻发现
"行为回退了"。这是 agent 工程里比召回更贴近产品的一层回归测试。

为什么断言用规则而不用 LLM 判卷？
追问/继承/排序这些行为是可以用规则精确判断的——用 LLM 判反而引入不确定性，
还可能放过明显的逻辑错误。能用规则判的绝不用模型，这是工程判断。

用法：python -m evals.decision [--db jobs.db]
"""

import argparse
import json
from pathlib import Path

from agent.graph import run_agent
from agent.profile import Profile, Slot

_SET = Path(__file__).with_name("decision_set.json")


def _load() -> list[dict]:
    return json.loads(_SET.read_text(encoding="utf-8"))["cases"]


def _profile_from(d: dict) -> Profile:
    """把用例里的 profile_in 转成 Profile。"""
    p = Profile()
    for k, v in (d or {}).items():
        setattr(p, k, Slot(value=str(v), filled=True, source="user"))
    return p


def _recommendations(state: dict) -> list[dict]:
    """复用 API 层的结构，保证"评测看到的"和"用户看到的"是同一份数据。"""
    out = []
    for job, s in state.get("scored") or []:
        out.append({
            "job_id": job["job_id"], "title": job["title"],
            "location": job["location"], "score": s.total,
            "grade": s.grade(), "reasons": s.reasons,
        })
    return out


def check_case(case: dict, db_path: str) -> tuple[bool, list[str]]:
    """跑一条用例，返回 (是否全过, 失败原因列表)。

    每条断言独立检查、独立报错——失败时能精确指出"哪个契约被破坏"，
    而不是笼统地说"这条没过"。
    """
    exp = case.get("expect") or {}
    state = run_agent(
        case["question"], db_path,
        profile=_profile_from(case.get("profile_in") or {}),
    )
    failures: list[str] = []

    # —— 追问行为 ——
    if "need_clarify" in exp:
        got = bool(state.get("need_clarify"))
        if got != exp["need_clarify"]:
            failures.append(f"need_clarify 期望 {exp['need_clarify']}，实际 {got}")

    answer = state.get("answer") or ""
    for kw in exp.get("must_ask") or []:
        # 追问关键词既要出现在回答里，也要确实走了追问分支
        if kw not in answer or not state.get("need_clarify"):
            failures.append(f"追问里缺少「{kw}」")
    for kw in exp.get("must_not_ask") or []:
        if kw in answer and state.get("need_clarify"):
            failures.append(f"不该问却问了「{kw}」")

    # —— 检索与推荐 ——
    cands = state.get("candidates") or []
    if exp.get("no_candidates") and cands:
        failures.append(f"追问场景不应检索，但召回了 {len(cands)} 条")
    if "min_candidates" in exp and len(cands) < exp["min_candidates"]:
        failures.append(f"候选数 {len(cands)} < 期望 {exp['min_candidates']}")

    recs = _recommendations(state)
    if "min_recommendations" in exp and len(recs) < exp["min_recommendations"]:
        failures.append(f"推荐数 {len(recs)} < 期望 {exp['min_recommendations']}")

    if exp.get("sorted_desc") and recs:
        scores = [r["score"] for r in recs]
        if scores != sorted(scores, reverse=True):
            failures.append(f"推荐未按分数降序：{scores}")

    if exp.get("recommendations_have_fields"):
        for r in recs:
            for f in exp["recommendations_have_fields"]:
                if not r.get(f):
                    failures.append(f"推荐项缺少字段 {f}（{r.get('title','?')}）")
                    break

    if "top1_city_is" in exp and recs:
        if exp["top1_city_is"] not in (recs[0]["location"] or ""):
            failures.append(
                f"Top1 城市为 {recs[0]['location']}，期望含 {exp['top1_city_is']}"
            )

    # —— 画像 ——
    prof = state["profile"].as_dict()
    for k, v in (exp.get("profile_contains") or {}).items():
        if prof.get(k) != v:
            failures.append(f"画像 {k} 期望 {v}，实际 {prof.get(k)}")
    for k in exp.get("profile_not_contains") or []:
        if k in prof:
            failures.append(f"画像不该含 {k}（实际 {prof.get(k)}）")

    # —— 技能差距 ——
    if exp.get("has_gap_report"):
        gap = state.get("gap") or {}
        if not gap.get("top_missing"):
            failures.append("期望有技能差距报告，但为空")

    # —— 推理轨迹 ——
    trace = "\n".join(state.get("trace") or [])
    for kw in exp.get("trace_contains") or []:
        if kw not in trace:
            failures.append(f"轨迹里缺少「{kw}」")

    return (not failures), failures


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals.decision", description="择岗顾问决策质量评测")
    parser.add_argument("--db", default="jobs.db")
    args = parser.parse_args()

    cases = _load()
    passed = 0
    details = []
    print(f"决策质量评测 · {len(cases)} 条用例\n" + "─" * 58)

    for case in cases:
        ok, failures = check_case(case, args.db)
        passed += int(ok)
        mark = "✓" if ok else "✗"
        print(f"{mark} [{case['id']}] {case['desc']}")
        if not ok:
            for f in failures:
                print(f"     └─ {f}")
        details.append({"id": case["id"], "desc": case["desc"], "pass": ok, "failures": failures})

    rate = passed / len(cases)
    print("─" * 58)
    print(f"通过 {passed}/{len(cases)}  通过率 {rate:.0%}")

    out = Path(__file__).with_name("decision_report.json")
    out.write_text(
        json.dumps(
            {"n_cases": len(cases), "passed": passed, "pass_rate": round(rate, 3), "details": details},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"明细已写入 {out}")


if __name__ == "__main__":
    main()
