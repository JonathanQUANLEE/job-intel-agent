"""Agent 包离线测试：用 FakeLLM 验证图结构与护栏，不需要 API Key。

FakeLLM 是注入的 chat_fn——这正是 build_graph(chat_fn=...) 参数存在的意义：
生产接真 API，测试接假模型，图的逻辑本身离线可测。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.graph import build_graph
from agent.tools import execute_tool_call

SYSTEM = {"role": "system", "content": "s"}


def _tool_call_msg(name="search_jobs", args='{"question": "test"}'):
    """构造一条带 tool_calls 的 assistant 消息（OpenAI 格式）。"""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    }


def test_graph_loops_then_ends():
    """第一轮返回 tool_calls（进 tools），第二轮返回纯文本（到 END）。"""
    responses = [_tool_call_msg(), {"role": "assistant", "content": "最终回答"}]
    calls = {"n": 0}

    def fake_chat(messages):
        msg = responses[calls["n"]]
        calls["n"] += 1
        return msg

    graph = build_graph(chat_fn=fake_chat, max_steps=6, db_path="jobs.db")
    state = graph.invoke({"messages": [SYSTEM, {"role": "user", "content": "q"}], "steps": 0})

    assert calls["n"] == 2                       # chat 节点跑了 2 轮
    assert state["steps"] == 1                    # 工具执行了 1 轮
    assert state["messages"][-1]["content"] == "最终回答"
    # messages 里应有 role=tool 的回填（无 Key 时是降级错误文本，也证明链路通）
    assert any(m.get("role") == "tool" for m in state["messages"])


def test_guardrail_forces_summarize():
    """模型永远要调工具 → 步数耗尽 → 走 summarize 护栏节点收尾。"""

    def endless_chat(messages):
        return _tool_call_msg()

    def fake_summarize(messages):
        return "已基于已有信息回答"

    import agent.graph as g

    orig = g.summarize_without_tools
    g.summarize_without_tools = fake_summarize  # 护栏收尾同样注入假实现
    try:
        graph = build_graph(chat_fn=endless_chat, max_steps=2, db_path="jobs.db")
        state = graph.invoke({"messages": [SYSTEM, {"role": "user", "content": "q"}], "steps": 0})
    finally:
        g.summarize_without_tools = orig

    assert state["steps"] == 2                    # 工具轮数被拦在上限
    assert state["messages"][-1]["content"] == "已基于已有信息回答"


def test_tool_failure_degrades():
    """未知工具 / 坏 JSON 参数 → 返回降级文本而不是抛异常。"""
    out1 = execute_tool_call("no_such_tool", "{}")
    out2 = execute_tool_call("search_jobs", "{bad json")
    assert "未知工具" in out1
    assert "不是合法 JSON" in out2
