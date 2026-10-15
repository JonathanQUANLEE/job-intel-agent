"""LLM 客户端（Agent 版）：带 tools 参数的对话调用。

与 rag/llm.py 的 ask() 区别：那是不带工具的"读片段答题"；
这里是把 TOOLS_SPEC 一起发给模型，模型可以返回 tool_calls
（"我要调 search_jobs，参数如下"），由 graph.py 执行后回填。
"""

import json

import requests

from rag.config import API_KEY, BASE_URL, LLM_MODEL
from .tools import TOOLS_SPEC

SYSTEM_PROMPT = (
    "你是「Agent 岗位情报站」的求职顾问。岗位库里有真实采集的大模型/算法方向招聘岗位。\n"
    "规则：\n"
    "1. 优先调用工具检索岗位库，基于工具返回的真实片段回答，绝不凭常识编造岗位；\n"
    "2. 工具没查到相关信息就直说查不到，不要硬凑；\n"
    "3. 引用岗位时给出岗位名称、城市和「编号」（如 tencent-120062），用户靠编号回查原文；\n"
    "4. 用户问'我会xx适合什么岗'时，逐条对比：先说匹配点，再指出缺失技能和补齐方向，"
    "不要泛泛而谈；\n"
    "5. 如果工具返回里提示'本地关键词检索模式'，说明当前没有配 API Key，"
    "可以顺带告诉用户配置 Key 后能做更细的语义匹配；\n"
    "6. 用中文，先结论后依据，简洁，不要复述整个 JD。"
)


def chat_with_tools(messages: list[dict]) -> dict:
    """一次带工具定义的 LLM 调用，返回原始 message（content 和/或 tool_calls）。

    messages 遵循 OpenAI 格式，工具结果用 role=tool 回填。
    温度 0.2：Agent 场景要的是稳定的决策，不是创意。
    """
    if not API_KEY:
        raise RuntimeError("没有找到 SILICONFLOW_API_KEY（看 rag/config.py 的提示）")
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "tools": TOOLS_SPEC,
            "temperature": 0.2,
        },
        timeout=(5, 120),
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


def summarize_without_tools(messages: list[dict]) -> str:
    """降级总结：步数耗尽后禁用工具，逼模型基于已有信息收尾。
    这是"护栏"的后半段——前半段是步数上限本身。"""
    if not API_KEY:
        raise RuntimeError("没有找到 SILICONFLOW_API_KEY")
    final_messages = messages + [
        {
            "role": "system",
            "content": "已达到工具调用步数上限。请立即基于以上已获取的信息回答用户，不要再调用工具。若信息不足，说明不足之处。",
        }
    ]
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": LLM_MODEL, "messages": final_messages, "temperature": 0.2},
        timeout=(5, 120),
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def tool_call_summary(message: dict) -> str:
    """把模型的 tool_calls 转成一行可读文本（CLI 打印过程用）。"""
    calls = message.get("tool_calls") or []
    parts = []
    for c in calls:
        try:
            args = json.loads(c["function"]["arguments"])
            brief = ", ".join(f"{k}={v}" for k, v in args.items())
        except (json.JSONDecodeError, KeyError):
            brief = "?"
        parts.append(f"{c['function']['name']}({brief})")
    return "；".join(parts)
