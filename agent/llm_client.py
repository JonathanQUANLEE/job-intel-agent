"""LLM 客户端（Agent 版）：带 tools 参数的对话调用。

与 rag/llm.py 的 ask() 区别：那是不带工具的"读片段答题"；
这里是把 TOOLS_SPEC 一起发给模型，模型可以返回 tool_calls
（"我要调 search_jobs，参数如下"），由 graph.py 执行后回填。
"""

import requests

from rag.config import API_KEY, BASE_URL, LLM_MODEL


def has_key() -> bool:
    """有没有配 Key。图里的 compose 节点靠它决定走模型润色还是模板兜底。"""
    return bool((API_KEY or "").strip())


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

# 润色阶段的系统提示：只重组织、不改事实。
# 为什么强调"不许改事实"？骨架里每个岗位、每项技能、每个分数都来自真实检索和
# 确定性打分，模型一旦"自由发挥"就会引入幻觉——这正好是 RAG 要防的东西。
_POLISH_PROMPT = (
    "你是求职顾问，下面是系统已经算好的岗位推荐结果（岗位、城市、匹配度分数、"
    "命中/缺失技能都来自真实检索与打分，**全部准确，不得增删改任何岗位、分数或技能**）。\n"
    "你的任务只是把它改写成更自然、更有顾问口吻的一段话：\n"
    "- 开头一句话点明结论（最推荐哪个、为什么）\n"
    "- 保留所有岗位编号（用户要靠它回查）\n"
    "- 保留匹配度分数和技能对比\n"
    "- 结尾给一句行动建议\n"
    "不要新增任何原始结果里没有的岗位或信息。"
)


def polish_answer(question: str, skeleton: str, history: list[dict]) -> str:
    """把确定性骨架润色成顾问口吻的回答。只改组织方式，不改事实。

    这是新版图 compose 节点用的函数，替代了旧的 chat_with_tools ReAct 循环——
    因为"择岗"的专业性来自确定性的检索+打分，模型负责的是表达而不是决策。
    这个分工（规则管事实、模型管表达）本身就是要展示的设计取向。
    """
    if not has_key():
        raise RuntimeError("没有找到 SILICONFLOW_API_KEY")

    messages = [
        {"role": "system", "content": _POLISH_PROMPT},
        {"role": "user", "content": f"用户问题：{question}\n\n系统算好的推荐结果：\n{skeleton}"},
    ]
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": LLM_MODEL, "messages": messages, "temperature": 0.3},
        timeout=(5, 90),
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
