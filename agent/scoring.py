"""匹配评分：把"这个岗位适不适合我"从感觉变成可解释的数字。

为什么这一步是"专业化"的关键？
通用 agent 能帮你读 JD，但它不会作答"我该投哪个"。因为打分需要两样它没有的东西：
  1) 一套**显式**的评分维度（技能命中 / 城市 / 方向 / 经验对齐）
  2) **可解释**的证据（命中哪几项技能、缺哪几项、依据是 JD 里哪句话）
打分逻辑写成确定性规则而不是交给模型，理由：一是可复现、可测试、可回归，
二是面试时能逐项讲清楚每个分是怎么来的——黑盒打分经不起追问。

分数结构（总分 100）：
  技能命中   0-55   用户技能在 JD 正文里出现，每命中一项加权
  方向对齐   0-20   岗位方向与用户目标方向一致
  城市命中   0-15   岗位所在城市与用户期望一致（说"不限"则给中位分）
  经验对齐   0-10   岗位经验要求与用户经验档位匹配
每一项都返回命中/缺失明细，前端可以逐条展开。
"""

from dataclasses import dataclass, field

# 技能词典：把用户技能描述和 JD 正文都往这套标准词上归一。
# 用"标准词 → 别名列表"的形式，命中任一别名就算命中该标准技能。
_SKILL_ALIASES: dict[str, list[str]] = {
    "Python": ["python"],
    "PyTorch": ["pytorch", "torch"],
    "TensorFlow": ["tensorflow", "tf"],
    "LangChain": ["langchain"],
    "LangGraph": ["langgraph"],
    "RAG": ["rag", "检索增强"],
    "FastAPI": ["fastapi"],
    "SQL": ["sql", "数据库", "mysql", "sqlite"],
    "Java": ["java"],
    "Go": ["golang", "go 语言", "go语言"],
    "C++": ["c++", "cpp"],
    "分布式": ["分布式", "多机", "并行训练", "分布式训练"],
    "机器学习": ["机器学习", "深度学习", "神经网络"],
    "大模型": ["大模型", "llm", "语言模型"],
    "微调": ["微调", "sft", "lora", "后训练"],
    "推理优化": ["推理优化", "推理加速", "量化", "vllm", "tensorrt"],
    "Transformer": ["transformer", "注意力机制"],
    "推荐算法": ["推荐算法", "召回", "排序", "ctr"],
    "Docker": ["docker", "容器"],
    "K8s": ["k8s", "kubernetes"],
    "Vue": ["vue"],
    "React": ["react"],
    "Node.js": ["node", "nodejs"],
    "前端": ["前端", "javascript", "typescript", "html", "css"],
}

# 方向词 → JD 里的典型特征词，用于"方向对齐"这一项的判定
_DIRECTION_HINTS = {
    "agent": ["agent", "智能体", "harness"],
    "llm": ["大模型", "llm", "语言模型", "推理", "训练", "微调"],
    "算法": ["算法", "机器学习", "深度学习", "推荐", "搜索", "nlp", "cv"],
    "工程": ["后端", "服务", "架构", "平台", "工程", "infra"],
    "产品": ["产品", "pm", "需求"],
    "数据": ["数据", "数仓", "分析"],
    "测试": ["测试", "qa", "质量"],
    "前端": ["前端", "web", "客户端"],
}

# 各维度满分
_W_SKILL, _W_DIRECTION, _W_CITY, _W_EXP = 55, 20, 15, 10


@dataclass
class Score:
    """一个岗位的评分结果，含可展开的证据明细。"""

    job_id: str
    total: float
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def grade(self) -> str:
        """把总分归成四档，前端用不同颜色展示。"""
        if self.total >= 80:
            return "强匹配"
        if self.total >= 60:
            return "较匹配"
        if self.total >= 40:
            return "可尝试"
        return "勉强相关"


def _norm_skills(profile_skills: str) -> dict[str, list[str]]:
    """把用户技能文本归一成 {标准技能: [命中的别名]}。"""
    low = profile_skills.lower()
    hit: dict[str, list[str]] = {}
    for canon, aliases in _SKILL_ALIASES.items():
        found = [a for a in aliases if a in low]
        if found:
            hit[canon] = found
    return hit


def score_job(job: dict, profile) -> Score:
    """给单个岗位打分。job 是 retriever 返回的 dict；profile 是 agent.profile.Profile。"""
    text = (job.get("text") or "").lower()
    title = (job.get("title") or "").lower()
    loc = job.get("location") or ""
    body = f"{title}\n{text}"

    matched, missing, reasons = [], [], []
    total = 0.0

    # —— 维度 1：技能命中（0-55）——
    user_skills = _norm_skills(profile.skills.value) if profile.skills.filled else {}
    if user_skills:
        for canon in user_skills:
            if any(a in body for a in _SKILL_ALIASES[canon]):
                matched.append(canon)
            else:
                missing.append(canon)
        if matched:
            # 命中率开方：命中越多收益递减，避免"技能堆得多就必高分"
            ratio = (len(matched) / len(user_skills)) ** 0.5
            pts = _W_SKILL * ratio
            total += pts
            reasons.append(f"命中 {len(matched)}/{len(user_skills)} 项技能（+{pts:.0f}）")
    else:
        # 用户没给技能时该项给中位分，不惩罚也不奖励
        total += _W_SKILL * 0.5
        reasons.append("未提供技能信息，技能项按中位计分")

    # —— 维度 2：方向对齐（0-20）——
    # 关键设计：标题命中 ≠ 正文命中。
    # 踩过的坑：只看"正文里有没有出现 agent"，会让"视频生成研究员"这种岗位
    # 靠 JD 里一句"与 agent 协作"混进推荐——它根本不是 Agent 岗。
    # 所以拆成两档：标题命中给满 20，仅正文命中给 8，标题完全不含方向词给 0。
    if profile.direction.filled:
        hints = _DIRECTION_HINTS.get(profile.direction.value, [])
        if any(h in title for h in hints):
            total += _W_DIRECTION
            reasons.append(f"方向「{profile.direction.value}」在标题体现（+{_W_DIRECTION}）")
        elif any(h in text for h in hints):
            weak = round(_W_DIRECTION * 0.4)
            total += weak
            reasons.append(f"方向「{profile.direction.value}」仅在正文提及（+{weak}，弱对齐）")
        else:
            reasons.append(f"方向「{profile.direction.value}」未体现（+0）")
    else:
        total += _W_DIRECTION * 0.5
        reasons.append("未指定方向，方向项按中位计分")

    # —— 维度 3：城市命中（0-15）——
    if profile.city.filled:
        want = profile.city.value
        if want in ("不限", ""):
            total += _W_CITY * 0.6
            reasons.append("城市不限，按中位计分")
        elif want in loc:
            total += _W_CITY
            reasons.append(f"城市命中（{loc}，+{_W_CITY}）")
        else:
            reasons.append(f"城市不符（期望 {want}，实际 {loc or '未知'}，+0）")
    else:
        total += _W_CITY * 0.5
        reasons.append("未指定城市，按中位计分")

    # —— 维度 4：经验对齐（0-10）——
    exp_text = (job.get("experience") or "").lower()
    if profile.experience.filled:
        want = profile.experience.value
        # 简单对齐规则：档位词直接命中，或岗位写明"不限经验"
        if want in exp_text or "不限" in exp_text or "经验不限" in exp_text:
            total += _W_EXP
            reasons.append(f"经验对齐（{exp_text or '不限'}，+{_W_EXP}）")
        else:
            reasons.append(f"经验档位可能不匹配（期望 {want}，岗位 {exp_text or '未标注'}，+0）")
    else:
        total += _W_EXP * 0.5
        reasons.append("未指定经验，按中位计分")

    return Score(
        job_id=job.get("job_id", ""),
        total=round(min(total, 100), 1),
        matched_skills=matched,
        missing_skills=missing,
        reasons=reasons,
    )


def rank_jobs(jobs: list[dict], profile, top_k: int = 5) -> list[tuple[dict, Score]]:
    """按总分排序返回 (job, score)。总分相同则按检索原始分兜底，保证稳定排序。"""
    scored = [(j, score_job(j, profile)) for j in jobs]
    scored.sort(key=lambda t: (t[1].total, t[0].get("score", 0)), reverse=True)
    return scored[:top_k]


def skill_gap_report(jobs: list[dict], profile) -> dict:
    """跨多个岗位汇总"最该补的技能"：统计缺失技能出现频次，频次高 = 通用门槛。"""
    from collections import Counter

    counter: Counter = Counter()
    for j in jobs:
        s = score_job(j, profile)
        counter.update(s.missing_skills)
    top = counter.most_common(5)
    return {
        "top_missing": [{"skill": k, "count": v} for k, v in top],
        "note": "出现频次越高，说明越接近该方向的通用门槛，优先级越高。",
    }
