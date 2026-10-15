"""存储模块：SQLite 入库，主键去重，支持增量更新。

为什么用 SQLite 而不是 CSV / MySQL？
- 比 CSV：能查询、有主键约束、并发写更安全，零配置单文件；
- 比 MySQL：本项目单机学习场景，不值得引入数据库服务。

本模块还负责两件"产品化"的事：
1. 库结构自动迁移（老库缺列自动补，不用删库重来）；
2. 给 Web 层提供列表 / 筛选 / 城市统计查询，前端不直接写 SQL。
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
    company    TEXT,
    department TEXT,
    experience TEXT,
    url        TEXT,
    raw_text   TEXT,
    fetched_at TEXT
);
"""

# 建索引：前端按城市/来源筛选、按标题搜索都要走这些列
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_jobs_location ON jobs(location)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_source   ON jobs(source)",
    "CREATE INDEX IF NOT EXISTS idx_jobs_category ON jobs(category)",
]


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """建连接 + 建表 + 迁移。返回的连接由调用方负责关闭。

    连接参数说明：
    - timeout=15：等锁最多 15 秒，而不是默认 5 秒就抛 "database is locked"；
    - isolation_level=None：关掉 Python 的隐式事务管理，DDL/DML 立即生效。
      默认模式下 sqlite3 会自作主张开事务，配合 executescript（它会隐式提交）
      容易让"刚写入的数据"在下一次连接里读不到，跨连接一致性问题很隐蔽。
      改为手动管理后，每次 execute/executemany 落盘行为可预期。
    - journal_mode=WAL：读写并发更友好（浏览页查询不会被采集写入阻塞）。
    """
    conn = sqlite3.connect(db_path, timeout=15, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(_SCHEMA)
    _migrate(conn)
    for sql in _INDEXES:
        conn.execute(sql)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """轻量迁移：给已存在的老库补上新增的列。

    早期版本没有 company / department 列，用户升级代码后如果直接跑，
    旧库会因缺列而报错。这里对比实际列名，缺什么补什么（SQLite 支持 ADD COLUMN）。
    """
    existing = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    for field in FIELDS:
        if field not in existing:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {field} TEXT")


def save_jobs(rows: list[dict], db_path: str | Path = "jobs.db") -> tuple[int, int]:
    """批量入库。INSERT OR IGNORE：重复 job_id 直接忽略 → 幂等，重复跑也不会插重。

    executemany 一次性提交，比逐条 insert 快一个数量级。
    返回 (新增条数, 跳过的重复条数)。

    对已存在的记录再做一次"正文更优则更新"：同一个岗位如果这次拿到了更完整的
    JD（详情接口补全过），就把正文更新掉。这样重跑爬虫既能去重，又能逐步补全。

    连接管理：显式 commit + close。sqlite3 的 `with conn:` 只提交事务不关连接，
    Windows 上残留连接会把 db 文件锁住（删不掉、后续连接读不到新数据），
    所以这里用 try/finally 保证一定关掉。
    """
    if not rows:
        return 0, 0
    placeholders = ", ".join(f":{f}" for f in FIELDS)
    sql = f"INSERT OR IGNORE INTO jobs ({', '.join(FIELDS)}) VALUES ({placeholders})"

    conn = _connect(db_path)
    try:
        before = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        conn.executemany(sql, rows)
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]

        # 已有记录：正文明显更长就更新（详情补全场景），其余字段不动
        for row in rows:
            conn.execute(
                "UPDATE jobs SET raw_text = :raw_text, company = :company, "
                "department = :department, experience = :experience "
                "WHERE job_id = :job_id AND LENGTH(:raw_text) > LENGTH(COALESCE(raw_text, ''))",
                row,
            )
        conn.commit()
    finally:
        conn.close()
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


# ---- 给 Web / API 层用的查询接口（前端不写 SQL）----

def count_jobs(db_path: str | Path = "jobs.db") -> int:
    with closing(_connect(db_path)) as conn:
        return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


def locations(db_path: str | Path = "jobs.db", limit: int = 30) -> list[dict]:
    """各城市岗位数，降序。前端用它做城市筛选下拉。"""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT location, COUNT(*) c FROM jobs WHERE location != '' "
            "GROUP BY location ORDER BY c DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"location": r[0], "count": r[1]} for r in rows]


def list_jobs(
    db_path: str | Path = "jobs.db",
    keyword: str = "",
    location: str = "",
    limit: int = 20,
    offset: int = 0,
) -> list[dict]:
    """条件列表查询：给"浏览岗位"用。

    关键词走 LIKE（title + raw_text + company），城市走等值匹配。
    这不是检索主路径（检索走向量），而是"让我翻翻有哪些岗位"的浏览路径，
    LIKE 在本项目几百条规模下完全够用，且零依赖。
    """
    where, params = ["1=1"], []
    if keyword:
        where.append("(title LIKE ? OR raw_text LIKE ? OR company LIKE ?)")
        like = f"%{keyword}%"
        params += [like, like, like]
    if location:
        where.append("location = ?")
        params.append(location)
    params += [limit, offset]

    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            f"SELECT job_id, title, location, category, company, department, experience, url, "
            f"SUBSTR(raw_text, 1, 400) FROM jobs WHERE {' AND '.join(where)} "
            f"ORDER BY fetched_at DESC, job_id LIMIT ? OFFSET ?",
            params,
        ).fetchall()
    return [
        {
            "job_id": r[0], "title": r[1], "location": r[2], "category": r[3],
            "company": r[4], "department": r[5], "experience": r[6],
            "url": r[7], "preview": r[8],
        }
        for r in rows
    ]


def get_job(db_path: str | Path, job_id: str) -> dict | None:
    """取单个岗位完整信息（详情弹窗用）。"""
    with closing(_connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    return dict(row) if row else None
