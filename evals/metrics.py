"""指标计算：纯函数，不联网，可离线单元测试。

为什么指标函数必须纯（输入列表、输出数字）？
评测脚本、单测、未来的 CI 全都复用同一份计算逻辑，
调参对比时数字才有可比性——指标一旦有两份实现，评测就失去意义。
"""


def hit_rate_at_k(retrieved: list[str], expected: list[str], k: int = 5) -> int:
    """命中率@k：top-k 检索结果里命中任一期望岗位记 1，否则 0。

    用"命中任一"而不是"全中"：一个方向常有多个正确岗位，
    检索找到其中一个即视为成功，这是召回评测的宽松共识。
    """
    top_k = retrieved[:k]
    return int(any(job_id in top_k for job_id in expected))


def reciprocal_rank(retrieved: list[str], expected: list[str]) -> float:
    """MRR 的单条分量：第一个命中期望岗位的排名的倒数。

    排第 1 得 1.0，排第 3 得 0.333，没命中得 0。
    比 hit_rate 更细：它在乎"找到的岗位排得靠不靠前"。
    """
    for rank, job_id in enumerate(retrieved, 1):
        if job_id in expected:
            return 1.0 / rank
    return 0.0


def average(values: list[float]) -> float:
    """平均分。空列表返回 0（无样本时给中性值，避免除零）。"""
    return round(sum(values) / len(values), 4) if values else 0.0
