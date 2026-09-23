"""清洗模块：把不同来源的原始字段 → 统一 schema。

为什么要统一 schema？
后面 RAG 和 Agent 只想面对一种数据形状。
"多源异构数据整合"正是面试常问的"不同源字段不一致怎么办"的答案主体。
"""

from datetime import datetime

# 统一 schema 的字段一览（README 里有说明表）
FIELDS = ["job_id", "source", "title", "location", "category", "url", "raw_text", "fetched_at"]


def _base(source: str) -> dict:
    """先给全字段默认值，避免缺字段时 KeyError / None 乱窜。"""
    row = {f: "" for f in FIELDS}
    row["source"] = source
    row["fetched_at"] = datetime.now().isoformat(timespec="seconds")
    return row


def normalize_tencent(post: dict) -> dict:
    row = _base("tencent")
    row["job_id"] = f"tencent-{post.get('RecruitPostId', '')}"
    row["title"] = (post.get("RecruitPostName") or "").strip()
    row["location"] = post.get("LocationName") or "未知"
    row["category"] = post.get("CategoryName") or ""
    # PostURL 常见是相对路径（//careers.tencent.com/...），补全协议
    url = post.get("PostURL") or ""
    row["url"] = url if url.startswith("http") else f"https:{url}" if url else ""
    row["raw_text"] = (post.get("Responsibility") or "").strip()
    return row


def normalize_hackernews(item: dict) -> dict:
    """HN 不是招聘数据，但字段映射逻辑一样：挑有用的，其余丢弃。"""
    row = _base("hackernews")
    row["job_id"] = f"hn-{item.get('id', '')}"
    row["title"] = (item.get("title") or "").strip()
    row["url"] = item.get("url") or ""
    row["category"] = "tech-news"
    row["raw_text"] = (item.get("text") or "").strip()
    return row


NORMALIZERS = {
    "tencent": normalize_tencent,
    "hackernews": normalize_hackernews,
}


def normalize(source: str, raw_posts: list[dict]) -> list[dict]:
    """入口：来源名 + 原始记录 → 统一 schema 记录列表。"""
    normalizer = NORMALIZERS[source]
    rows, skipped = [], 0
    for raw in raw_posts:
        row = normalizer(raw)
        if not row["job_id"] or row["job_id"].endswith("-"):
            skipped += 1  # 没有唯一 ID 的脏数据，宁可丢弃也不能入库
            continue
        if not row["title"]:
            skipped += 1
            continue
        rows.append(row)
    return rows
