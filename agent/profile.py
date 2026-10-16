"""偏好槽位（slots）：把"择岗"这件事里真正影响决策的几个维度显式化。

为什么需要它？
通用编程 agent（Codex / OpenCode 那类）不做择岗，因为它们没有"决策框架"——
它们擅长的是"给我一段代码我能改"，而不是"我该投哪个岗"。
择岗这种专业场景里，真正稀缺的不是信息（岗位 JD 到处都是），而是
**决策所需的信息是否齐全**。用户问"帮我看看有什么岗位"时，
他心里的约束（城市、技术栈、经验、想要什么类型的活）是隐式的，
agent 的专业性就体现在：识别出这些隐式约束、缺什么主动问、问到就锁定。

设计取舍：为什么用固定槽位而不是让模型自由抽？
固定槽位让"缺什么该问什么"变成可枚举、可测试的确定逻辑，
模型只负责"从一句话里填槽"，判断"槽满没满"是确定性代码。
这样多轮追问的行为不会被模型随机性带偏——面试时能讲清楚边界在哪。

槽位分两类：
  HARD（硬条件）：缺了就一定追问——城市、岗位方向。缺这两样，推荐等于瞎猜。
  SOFT（软偏好）：有就用、没有就用默认值——经验、技术栈、薪资倾向。
  软偏好缺失不追问，避免第一轮就把用户问烦。
"""

from dataclasses import dataclass, field

# 城市词表：命中即视为填上了 city 槽位。
# 与 rag/offline.py 的城市表同源思路（那里给检索加权，这里给槽位识别）。
_CITIES = [
    "北京", "上海", "深圳", "广州", "杭州", "成都", "武汉", "西安",
    "南京", "苏州", "重庆", "长沙", "天津", "郑州", "青岛", "厦门",
    "合肥", "济南", "福州", "大连", "哈尔滨", "沈阳", "中国香港",
    "新加坡", "东京", "首尔", "伦敦", "巴黎", "纽约", "西雅图",
]

# 方向词表：把"想做哪一类工作"归一成少量标签，便于后续过滤和追问判断。
# 注意：不放"推荐""搜索"这类单字歧义词——"帮我推荐下岗位"里的"推荐"
# 是动词不是方向，误命中会让 agent 把"随便看看"当成"要做推荐算法"，这类误伤
# 必须靠词表精度挡掉（踩过这个坑：早期版本把"推荐下"判成了推荐算法方向）。
_DIRECTIONS = {
    "agent": ["agent", "智能体", "harness", "multi-agent", "工作流"],
    "llm": ["大模型", "llm", "语言模型", "大模型推理", "微调", "后训练"],
    "算法": ["算法", "机器学习", "深度学习", "推荐算法", "搜索算法", "nlp", "cv", "语音"],
    "工程": ["后端", "服务端", "工程", "架构", "平台", "infra", "基础设施", "存储"],
    "产品": ["产品经理", "产品", "pm"],
    "数据": ["数据", "数仓", "分析", "bi"],
    "测试": ["测试", "qa", "质量"],
    "前端": ["前端", "web", "客户端"],
}

# 经验档位：把自由文本归一成三档，用于硬条件过滤时对齐。
_EXPERIENCE = {
    "应届/在校": ["应届", "在校", "实习生", "实习", "校招", "无经验", "0 年"],
    "1-3 年": ["一年", "两年", "三年", "1年", "2年", "3年", "1-3"],
    "3 年以上": ["三年以上", "五年", "5年", "资深", "高级", "专家", "架构师"],
}


@dataclass
class Slot:
    """一个槽位的当前状态。value 为空 = 未填。"""

    value: str = ""
    filled: bool = False
    source: str = "default"  # default | inferred | user


@dataclass
class Profile:
    """用户择岗画像。所有槽位集中在这里，图里的节点读写同一个对象。"""

    city: Slot = field(default_factory=Slot)
    direction: Slot = field(default_factory=Slot)
    experience: Slot = field(default_factory=Slot)
    skills: Slot = field(default_factory=Slot)
    salary: Slot = field(default_factory=Slot)

    def as_dict(self) -> dict:
        """转成纯字典，便于塞进 SSE 事件给前端展示"已了解的信息"。"""
        out = {}
        for name in ("city", "direction", "experience", "skills", "salary"):
            s = getattr(self, name)
            if s.filled:
                out[name] = s.value
        return out

    def merge(self, other: "Profile") -> None:
        """把另一份画像里已填的槽位并进来（新一轮用户回答覆盖旧值）。"""
        for name in ("city", "direction", "experience", "skills", "salary"):
            mine = getattr(self, name)
            theirs = getattr(other, name)
            if theirs.filled:
                mine.value = theirs.value
                mine.filled = True
                mine.source = theirs.source


def _first_hit(text: str, table: dict) -> str:
    """在一张"标签 → 关键词列表"的表里找命中的标签，按关键词长度优先（长词更具体）。"""
    low = text.lower()
    best, best_len = "", 0
    for label, words in table.items():
        for w in words:
            if w in low and len(w) > best_len:
                best, best_len = label, len(w)
    return best


def extract_slots(text: str) -> Profile:
    """从一段自然语言里抽槽位。纯规则、离线可跑、结果确定。

    为什么不直接丢给模型抽？
    模型抽取会漏、会漂，而且无 Key 时完全不可用。规则抽取覆盖了
    "深圳有哪些 Agent 岗"这类高频表述（这恰好是离线模式下用户的主要问法），
    模型只用来兜住规则覆盖不到的长尾表达。
    """
    p = Profile()
    low = text.lower()

    # 城市：先看有没有明确的"不限"表态（用户常见说法），再匹配具体城市名。
    # 顺序很重要——"城市不限"里没有城市名，只能靠"不限/都可以/哪都行"这类词捕获。
    if any(k in text for k in ["不限", "都可以", "哪都行", "都行", "无所谓城市", "全国"]):
        p.city = Slot(value="不限", filled=True, source="inferred")
    else:
        for c in _CITIES:
            if c in text:
                p.city = Slot(value=c, filled=True, source="inferred")
                break

    # 方向：用归一表
    d = _first_hit(low, _DIRECTIONS)
    if d:
        p.direction = Slot(value=d, filled=True, source="inferred")

    # 经验：档位归一
    e = _first_hit(text, _EXPERIENCE)
    if e:
        p.experience = Slot(value=e, filled=True, source="inferred")

    # 技术栈：常见技能词命中即记，作为 skills 槽位的粗粒度值
    skill_words = [
        "python", "pytorch", "tensorflow", "langchain", "langgraph", "rag",
        "java", "go", "c++", "sql", "分布式", "推荐算法", "微调", "transformer",
        "fastapi", "docker", "k8s", "vue", "react", "node",
    ]
    hit_skills = [w for w in skill_words if w in low]
    if hit_skills:
        p.skills = Slot(value="、".join(hit_skills), filled=True, source="inferred")

    # 薪资倾向：粗粒度识别
    for kw in ["薪资", "待遇", "多少钱", "税前", "包吃", "双休", "965", "wlb"]:
        if kw in low:
            p.salary = Slot(value=text.strip()[:60], filled=True, source="inferred")
            break

    return p


# 硬条件槽位：缺任一个都必须追问，否则推荐没有意义
HARD_SLOTS = ("city", "direction")


def missing_hard_slots(p: Profile) -> list[str]:
    """返回还没填的硬条件槽位名（按追问优先级排序）。"""
    return [name for name in HARD_SLOTS if not getattr(p, name).filled]


# 追问话术：每个硬槽位一句，避免模型自由发挥时问出一堆碎问题。
_CLARIFY_TEXT = {
    "city": "你希望工作城市是哪里？（比如深圳、北京、杭州；也可以说「不限」）",
    "direction": "你想找哪一类岗位？（比如 Agent 开发、大模型算法、推荐算法、后端工程、产品）",
}


def clarify_question(missing: list[str]) -> str:
    """把缺失槽位拼成一轮简洁追问。一次只问硬条件，最多两问，不啰嗦。"""
    parts = [_CLARIFY_TEXT[m] for m in missing if m in _CLARIFY_TEXT]
    head = "为了给你靠谱的推荐，还差两个信息：" if len(parts) > 1 else "还差一个信息："
    return head + "\n" + "\n".join(f"- {t}" for t in parts)
