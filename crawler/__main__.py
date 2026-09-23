"""CLI 入口：python -m crawler --source tencent --keyword "大模型"

argparse 三个子功能：
  默认      抓取并入库
  --stats   看各来源条数
  --sample  抽样看数据
"""

import argparse

from . import __version__
from .http_client import build_session
from .normalize import normalize
from .sources import fetch_hackernews_items, fetch_tencent_jobs
from .storage import sample, save_jobs, stats


def main() -> None:
    parser = argparse.ArgumentParser(prog="crawler", description="岗位情报采集器")
    parser.add_argument("--source", choices=["tencent", "hackernews"], default="tencent")
    parser.add_argument("--keyword", default="大模型", help="腾讯招聘搜索关键词")
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=20, help="HN 拉多少条")
    parser.add_argument("--db", default="jobs.db")
    parser.add_argument("--stats", action="store_true", help="查看入库统计")
    parser.add_argument("--sample", type=int, default=0, metavar="N", help="抽样看 N 条数据")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    if args.stats:
        for source, count in stats(args.db):
            print(f"{source:<12} {count:>6} 条")
        return

    if args.sample:
        for job_id, title, location in sample(args.db, args.sample):
            print(f"{job_id:<18} {location:<10} {title}")
        return

    session = build_session()
    if args.source == "tencent":
        raw = fetch_tencent_jobs(session, args.keyword, args.max_pages)
    else:
        raw = fetch_hackernews_items(session, args.max_items)

    rows = normalize(args.source, raw)
    inserted, skipped = save_jobs(rows, args.db)
    print(f"[{args.source}] 原始 {len(raw)} 条 → 有效 {len(rows)} 条 → 新增 {inserted} 条（重复/跳过 {skipped} 条）")
    print(f"下一步：python -m crawler --stats   或   python -m crawler --sample 5")


if __name__ == "__main__":
    main()
