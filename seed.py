"""数据种子：把仓库里的 CSV 快照导入 SQLite。

为什么需要它？
用户 clone 下来第一件事是想"看看它是什么"，但如果岗位库是空的，
界面打开就是一片空白 —— 体验直接崩掉。
有了这个脚本，用户不必先跑爬虫，就能立刻在界面上浏览真实岗位。

用法：
    python -m seed                      # 从 data/jobs_export.csv 导入
    python -m seed --csv 其他文件.csv    # 指定文件
    python -m seed --db my.db           # 指定库
重复运行安全（主键去重）。
"""

import argparse
import csv
from pathlib import Path

from crawler.normalize import FIELDS
from crawler.storage import save_jobs

DEFAULT_CSV = "data/jobs_export.csv"


def load_csv(path: str) -> list[dict]:
    """读 CSV 成 record 列表。空字段补默认值，缺列也不炸。"""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for raw in csv.DictReader(f):
            row = {f: (raw.get(f) or "").strip() for f in FIELDS}
            if row["job_id"]:
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(prog="seed", description="把 CSV 快照导入 SQLite")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--db", default="jobs.db")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"找不到 {args.csv}。请先运行 python -m crawler 采集数据。")
        return

    rows = load_csv(args.csv)
    inserted, skipped = save_jobs(rows, args.db)
    print(f"从 {args.csv} 读取 {len(rows)} 条 → 新增 {inserted} 条（跳过 {skipped} 条重复）")
    print("下一步：python -m api   然后打开 http://127.0.0.1:8000")


if __name__ == "__main__":
    main()
