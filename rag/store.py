"""RAG 存储与检索：向量也放 SQLite，检索时用 numpy 暴力余弦。

为什么不用 ChromaDB / FAISS / pgvector？
- 数据量几百条，numpy 暴力余弦毫秒级出结果，方案零依赖；
- 少一个组件 = 少一个要部署、要维护、要出错、要在面试里解释的点；
- 面试标准答案："数据量到万级、QPS 上来后，把 search() 内部换成
  FAISS/pgvector 即可，接口不变"——这就是留好抽象边界的价值。

表结构：
  jd_chunks(id, job_id, chunk_index, chunk_text, embedding BLOB)
  向量以 numpy float32 二进制存 BLOB，比存 JSON 省约 4 倍空间、载入更快。
"""

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jd_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    embedding   BLOB,
    UNIQUE(job_id, chunk_index)
);
"""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def _pack(vec: list[float]) -> bytes:
    """向量 → float32 二进制。存之前顺手转 numpy，后面检索直接用。"""
    return np.asarray(vec, dtype=np.float32).tobytes()


def reset_chunks(db_path: str | Path = "jobs.db") -> None:
    """清空所有 chunks（--force 全量重建时用）。"""
    with closing(_connect(db_path)) as conn:
        conn.execute("DELETE FROM jd_chunks")


def save_chunks(
    job_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
    db_path: str | Path = "jobs.db",
) -> None:
    """一个岗位的全部片段入库。INSERT OR REPLACE：重跑同一岗位 = 覆盖，幂等。"""
    if not chunks:
        return
    rows = [
        (job_id, i, text, _pack(embeddings[i]))
        for i, text in enumerate(chunks)
    ]
    with closing(_connect(db_path)) as conn, conn:
        conn.executemany(
            "INSERT OR REPLACE INTO jd_chunks (job_id, chunk_index, chunk_text, embedding) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )


def pending_jobs(db_path: str | Path = "jobs.db", force: bool = False) -> list[tuple]:
    """找出需要建索引的岗位。

    增量逻辑：已有 chunk 的岗位跳过（幂等，重复跑不浪费 API 额度）。
    force=True 时返回全部岗位（配合 reset_chunks 全量重建）。
    """
    with closing(_connect(db_path)) as conn:
        if force:
            return conn.execute(
                "SELECT job_id, raw_text FROM jobs WHERE raw_text != '' ORDER BY job_id"
            ).fetchall()
        return conn.execute(
            """
            SELECT j.job_id, j.raw_text FROM jobs j
            LEFT JOIN jd_chunks c ON c.job_id = j.job_id
            WHERE j.raw_text != '' AND c.id IS NULL
            ORDER BY j.job_id
            """
        ).fetchall()


def chunk_stats(db_path: str | Path = "jobs.db") -> tuple[int, int]:
    """(已建索引的岗位数, 片段总数)。"""
    with closing(_connect(db_path)) as conn:
        jobs = conn.execute(
            "SELECT COUNT(DISTINCT job_id) FROM jd_chunks"
        ).fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM jd_chunks").fetchone()[0]
    return jobs, chunks


def search(
    query_vec: list[float],
    db_path: str | Path = "jobs.db",
    top_k: int = 5,
) -> list[tuple[float, str, str]]:
    """余弦检索：返回 [(相似度, job_id, chunk_text), ...] 按相似度降序。

    原理三步：
    1. 把库里所有向量叠成矩阵（已归一化，余弦 = 点积）；
    2. 查询向量也归一化；
    3. 矩阵乘向量，一次算出与所有片段的相似度，取 top_k。
    """
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT job_id, chunk_text, embedding FROM jd_chunks WHERE embedding IS NOT NULL"
        ).fetchall()
    if not rows:
        return []

    mat = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    mat = mat / np.linalg.norm(mat, axis=1, keepdims=True)  # 每行归一化
    q = np.asarray(query_vec, dtype=np.float32)
    q = q / np.linalg.norm(q)

    scores = mat @ q
    top = np.argsort(-scores)[:top_k]
    return [(float(scores[i]), rows[i][0], rows[i][1]) for i in top]
