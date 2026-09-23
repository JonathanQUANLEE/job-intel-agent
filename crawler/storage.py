"""存储模块：SQLite 入库，主键去重，支持增量更新。

为什么用 SQLite 而不是 CSV / MySQL？
- 比 CSV：能查询、有主键约束、并发写更安全，零配置单文件；
- 比 MySQL：本项目单机学习场景，不值得引入数据库服务。
"""

import sqlite3
from contextlib import closing
from pathlib import Path

from .normalize import FIELDS

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id     TEXT PRIMARY KEY,
    source     TEXT,
    title      TEXT,
    location   TEXT,
    category   TEXT,
    url        TEXT,
    raw_text   TEXT,
    fetched_at TEXT
);
"""


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    return conn


def save_jobs(rows: list[dict], db_path: str | Path = "jobs.db") -> tuple[int, int]:
    """批量入库。INSERT OR IGNORE：重复 job_id 直接忽略 → 幂等，重复跑也不会插重。

    executemany 一次性提交，比逐条 insert 快一个数量级。
    返回 (新增条数, 跳过的重复条数)。
    注意：closing(...) 确保连接一定关闭 —— sqlite3 的 with 只提交事务，不关连接，
    Windows 上不关连接会导致 db 文件被锁住删不掉。
    """
    if not rows:
        return 0, 0
    placeholders = ", ".join(f":{f}" for f in FIELDS)
    sql = f"INSERT OR IGNORE INTO jobs ({', '.join(FIELDS)}) VALUES ({placeholders})"
    with closing(_connect(db_path)) as conn, conn:  # 内层 with 管提交，closing 管关闭
        before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        conn.executemany(sql, rows)
        after = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    return after - before, len(rows) - (after - before)


def stats(db_path: str | Path = "jobs.db") -> list[tuple]:
    """按来源统计条数，给 CLI 的 --stats 用。"""
    with closing(_connect(db_path)) as conn:
        return conn.execute(
            "SELECT source, COUNT(*) FROM jobs GROUP BY source ORDER BY COUNT(*) DESC"
        ).fetchall()


def sample(db_path: str | Path = "jobs.db", limit: int = 5) -> list[tuple]:
    """抽样看几条，确认数据没存歪。"""
    with closing(_connect(db_path)) as conn:
        return conn.execute(
            "SELECT job_id, title, location FROM jobs LIMIT ?", (limit,)
        ).fetchall()
