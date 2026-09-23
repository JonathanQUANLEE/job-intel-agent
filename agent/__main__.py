"""CLI 入口：python -m agent --ask "..."

打印完整 ReAct 过程（模型决定 → 工具执行 → 继续思考），面试演示时
评委能看到"决策权在模型手里"这件事本身，而不是只看到最终答案。
"""

import argparse
import json

from . import __version__
from .graph import run_agent
from .llm_client import tool_call_summary


def _print_trace(messages: list[dict]) -> str:
    """回放 messages，展示循环过程。返回最终回答文本。"""
    final = ""
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            print(f"\n  [模型] 决定调用工具：{tool_call_summary(m)}")
        elif role == "assistant" and m.get("content"):
            final = m["content"]
        elif role == "tool":
            preview = (m.get("content") or "")[:120].replace("\n", " ")
            print(f"  [工具] 返回：{preview}...")
    return final


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="Agent 岗位情报站 · 智能体")
    parser.add_argument("--ask", metavar="问题", required=True, help="求职问题")
    parser.add_argument("--db", default="jobs.db")
    parser.add_argument("--max-steps", type=int, default=6, help="工具调用轮数上限（护栏）")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    print(f"问题：{args.ask}")
    print("─" * 50)
    state = run_agent(args.ask, db_path=args.db, max_steps=args.max_steps)

    answer = _print_trace(state["messages"])
    print("─" * 50)
    print("\n回答：")
    print(answer if answer else "（未产生回答——检查 API Key 与索引）")
    print(f"\n本轮工具调用 {state['steps']} 次（上限 {args.max_steps}）")


if __name__ == "__main__":
    main()
