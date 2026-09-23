"""LangGraph 编排：把 Agent 循环画成一张图。

图结构（Workflow 的骨架 + 模型掌握决策权 = Agent）：

    START → chat ──无 tool_calls──→ END（直接回答）
              │
              ├─有 tool_calls 且未超步数──→ tools → 回到 chat（循环）
              │
              └─有 tool_calls 但步数耗尽──→ summarize → END（护栏降级）

为什么用 LangGraph 而不是 while 循环？
- 状态流转显式化：节点/边/条件边就是流程图本身，可检查、可 checkpoint；
- 通用骨架：换成"先检索再评估质量再决定重写"等多步图，只加节点不动框架。
（小数据量下 while 也能跑——选 LangGraph 是为了练工业界主流的编排表达。）
"""

from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from .llm_client import chat_with_tools, summarize_without_tools
from .tools import execute_tool_call

MAX_STEPS = 6  # 护栏：工具轮数上限。防"查不完"死循环烧钱


class AgentState(TypedDict):
    """所有节点共享的'传菜窗口'。messages 是 OpenAI 格式的对话历史。"""
    messages: list[dict]
    steps: int


def build_graph(
    chat_fn: Callable[[list[dict]], dict] = chat_with_tools,
    max_steps: int = MAX_STEPS,
    db_path: str = "jobs.db",
) -> Any:
    """编译好的 Agent 图。chat_fn 可注入——单元测试用假 LLM，
    生产用 llm_client.chat_with_tools（依赖注入让逻辑可离线验证）。"""
    builder = StateGraph(AgentState)

    def chat_node(state: AgentState) -> dict:
        message = chat_fn(state["messages"])
        return {"messages": state["messages"] + [message]}

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        out = list(state["messages"])
        for call in last.get("tool_calls") or []:
            result = execute_tool_call(
                call["function"]["name"], call["function"]["arguments"], db_path
            )
            # role=tool + tool_call_id：OpenAI 协议里工具结果的回填格式
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result,
                }
            )
        return {"messages": out, "steps": state["steps"] + 1}

    def summarize_node(state: AgentState) -> dict:
        content = summarize_without_tools(state["messages"])
        return {"messages": state["messages"] + [{"role": "assistant", "content": content}]}

    def route_after_chat(state: AgentState) -> str:
        last = state["messages"][-1]
        if last.get("tool_calls"):
            return "tools" if state["steps"] < max_steps else "summarize"
        return END

    # 节点与边：与文件顶部的图一一对应
    builder.add_node("chat", chat_node)
    builder.add_node("tools", tools_node)
    builder.add_node("summarize", summarize_node)
    builder.add_edge(START, "chat")
    builder.add_edge("tools", "chat")
    builder.add_edge("summarize", END)
    builder.add_conditional_edges("chat", route_after_chat, ["tools", "summarize", END])
    return builder.compile()


def run_agent(question: str, db_path: str = "jobs.db", max_steps: int = MAX_STEPS) -> dict:
    """跑一轮完整 Agent 问答，返回最终 state（messages 里有全过程）。

    CLI/评测/API 都从这一个入口走——单一入口，行为可预测。
    """
    from .llm_client import SYSTEM_PROMPT  # 局部导入避免循环依赖

    graph = build_graph(max_steps=max_steps, db_path=db_path)
    init: AgentState = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "steps": 0,
    }
    return graph.invoke(init)
