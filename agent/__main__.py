"""CLI 入口：python -m agent --ask "..." [--profile-json '{...}']

打印完整的决策轨迹（抽槽 → 追问/检索 → 打分 → 补搜 → 成稿），
面试演示时评委能看到"agent 在每一步想什么、为什么这么走"，
而不是只看到一个最终答案——这才是"编排"和"调 API"的区别。

--profile-json 支持跨轮：第一轮追问后，把上一轮输出的画像存下来传回来，
下一轮就能接着用（命令行里手动演示多轮）。
"""

import argparse
import json

from . import __version__
from .graph import run_agent
from .profile import Profile, Slot


def _load_profile(raw: str) -> Profile:
    """从 JSON 还原画像（命令行传参用）。格式：{"city": "深圳", "direction": "agent"}"""
    p = Profile()
    if not raw:
        return p
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print("[警告] --profile-json 不是合法 JSON，已忽略")
        return p
    for name in ("city", "direction", "experience", "skills", "salary"):
        if data.get(name):
            setattr(p, name, Slot(value=str(data[name]), filled=True, source="user"))
    return p


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="Agent 岗位情报站 · 择岗顾问")
    parser.add_argument("--ask", metavar="问题", required=True, help="求职问题")
    parser.add_argument("--db", default="jobs.db")
    parser.add_argument("--profile-json", default="", help="跨轮画像，如 '{\"city\":\"深圳\"}'")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    print(f"问题：{args.ask}")
    print("─" * 56)

    state = run_agent(args.ask, db_path=args.db, profile=_load_profile(args.profile_json))

    # 决策轨迹：每一步都在回答"agent 为什么这么做"
    print("决策轨迹：")
    for i, step in enumerate(state.get("trace") or [], 1):
        print(f"  {i}. {step}")

    print("─" * 56)
    print("\n回答：\n")
    print(state.get("answer") or "（未产生回答——检查数据与索引）")

    # 追问场景下把画像打出来，方便用户带着它再问一轮
    if state.get("need_clarify"):
        print("\n" + "─" * 56)
        print("下一轮可带上已识别的画像继续问：")
        print(f"  python -m agent --ask \"你的补充信息\" --profile-json '{json.dumps(state['profile'].as_dict(), ensure_ascii=False)}'")


if __name__ == "__main__":
    main()
