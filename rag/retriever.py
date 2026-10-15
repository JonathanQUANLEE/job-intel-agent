"""统一检索层：把"向量检索"和"离线检索"收进一个入口，自动降级。

为什么要多这一层？
产品里有三个调用方（CLI / Agent 工具 / API），如果每个地方都写一遍
"有 Key 用向量、没 Key 用 BM25、出错再降级"的判断，逻辑会散落三处、
迟早不一致。收在 retriever 里，调用方只管问 "给我最相关的岗位"。

降级链：
  1. 有 Key 且索引已建 → 向量语义检索（效果最好）
  2. 无 Key / 未建索引 / 调用失败 → 本地 BM25 检索（永远可用）
  3. 都拿不到 → 返回空列表，由上层给出"先采集数据"的提示
"""

from .config import has_key
from .offline import search_offline
from .store import chunk_stats, search as vector_search

# 检索通道标识，前端用来提示用户"当前是语义模式还是离线模式"
MODE_VECTOR = "vector"
MODE_OFFLINE = "offline"


def retrieve(
    query: str,
    db_path: str = "jobs.db",
    top_k: int = 5,
    want_mode: bool = False,
) -> list[dict] | tuple[list[dict], str]:
    """检索岗位，返回去重后的结果列表。

    每个元素：{job_id, title, location, company, url, text, score, mode}
    - 同岗位命中多片段时只保留最高分（多样性，避免上下文被一个岗位占满）
    - want_mode=True 时额外返回本次实际使用的通道名（前端提示用）
    """
    import sqlite3
    from contextlib import closing

    raw: list[tuple] = []
    mode = MODE_OFFLINE

    # 通道 1：向量检索（需要 Key + 已建索引）
    if has_key():
        jobs_indexed, _ = chunk_stats(db_path)
        if jobs_indexed > 0:
            try:
                from .embedder import embed_texts

                vec = embed_texts([query])[0]
                raw = vector_search(vec, db_path, top_k=top_k * 3)
                if raw:
                    mode = MODE_VECTOR
            except Exception:  # noqa: BLE001 —— 云端抽风就降级，不能让产品白屏
                raw = []

    # 通道 2：本地离线检索（无 Key / 未建索引 / 上个通道失败都走这里）
    if not raw:
        raw = search_offline(query, db_path, top_k=top_k * 3)
        mode = MODE_OFFLINE

    # 按岗位去重：同一岗位只留最高分片段
    best: dict[str, tuple] = {}
    for score, job_id, text in raw:
        if job_id not in best or score > best[job_id][0]:
            best[job_id] = (score, job_id, text)
    ordered = sorted(best.values(), key=lambda h: h[0], reverse=True)[:top_k]

    # 补全展示元数据
    meta: dict[str, dict] = {}
    ids = [job_id for _, job_id, _ in ordered]
    if ids:
        placeholders = ", ".join("?" * len(ids))
        with closing(sqlite3.connect(db_path)) as conn:
            rows = conn.execute(
                f"SELECT job_id, title, location, company, department, experience, url "
                f"FROM jobs WHERE job_id IN ({placeholders})",
                ids,
            ).fetchall()
        meta = {
            r[0]: {
                "title": r[1], "location": r[2], "company": r[3],
                "department": r[4], "experience": r[5], "url": r[6],
            }
            for r in rows
        }

    results = [
        {
            "job_id": job_id,
            "title": meta.get(job_id, {}).get("title", job_id),
            "location": meta.get(job_id, {}).get("location", "未知"),
            "company": meta.get(job_id, {}).get("company", ""),
            "department": meta.get(job_id, {}).get("department", ""),
            "experience": meta.get(job_id, {}).get("experience", ""),
            "url": meta.get(job_id, {}).get("url", ""),
            "text": text,
            "score": round(float(score), 3),
            "mode": mode,
        }
        for score, job_id, text in ordered
    ]
    return (results, mode) if want_mode else results
