"""CLI 入口：python -m rag --index / --ask "..." / --stats

三步走通 RAG：
  --index   把 jobs 表里的岗位描述切块 + 嵌入 + 入库（增量，重复跑不浪费额度）
  --ask     问题 → 嵌入 → 检索 → LLM 带引用回答
  --stats   看已建索引的岗位数和片段总数
"""

import argparse
import sqlite3
from contextlib import closing

from . import __version__
from .chunker import chunk_text
from .embedder import embed_texts
from .llm import ask
from .store import chunk_stats, pending_jobs, reset_chunks, save_chunks, search


def _job_meta(db_path: str, job_ids: list[str]) -> dict[str, dict]:
    """把 job_id 补全成展示信息（标题/地点/链接），给回答和来源列表用。"""
    if not job_ids:
        return {}
    placeholders = ", ".join("?" * len(job_ids))
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            f"SELECT job_id, title, location, url FROM jobs WHERE job_id IN ({placeholders})",
            job_ids,
        ).fetchall()
    return {r[0]: {"title": r[1], "location": r[2], "url": r[3]} for r in rows}


def cmd_index(args) -> None:
    if args.force:
        reset_chunks(args.db)

    jobs = pending_jobs(args.db, force=args.force)
    if not jobs:
        jobs_done, _ = chunk_stats(args.db)
        print(f"没有需要建索引的岗位（已索引 {jobs_done} 个岗位）。加 --force 全量重建。")
        return

    print(f"开始为 {len(jobs)} 个岗位建索引...")
    for n, (job_id, raw_text) in enumerate(jobs, 1):
        chunks = chunk_text(raw_text)
        if not chunks:
            print(f"[{n}/{len(jobs)}] {job_id:<20} 描述为空，跳过")
            continue
        embeddings = embed_texts(chunks)
        save_chunks(job_id, chunks, embeddings, args.db)
        print(f"[{n}/{len(jobs)}] {job_id:<20} {len(chunks)} 块")

    jobs_done, total_chunks = chunk_stats(args.db)
    print(f"完成：{jobs_done} 个岗位 / {total_chunks} 个片段。下一步：python -m rag --ask \"...\"")


def cmd_ask(args) -> None:
    query_vec = embed_texts([args.question])[0]  # 问题也要先嵌入，才能比相似度
    raw_hits = search(query_vec, args.db, top_k=args.top_k * 2)

    # 同一岗位可能命中多个片段：按 job_id 去重，只留相似度最高的一段，
    # 避免上下文被同一个岗位占满（这也是"多样性"重排的最小实现）
    best: dict[str, tuple[float, str]] = {}
    for score, job_id, text in raw_hits:
        if job_id not in best or score > best[job_id][0]:
            best[job_id] = (score, text)

    ordered = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[: args.top_k]
    meta = _job_meta(args.db, [job_id for job_id, _ in ordered])

    hits = [
        {
            "job_id": job_id,
            "title": meta.get(job_id, {}).get("title", ""),
            "location": meta.get(job_id, {}).get("location", ""),
            "url": meta.get(job_id, {}).get("url", ""),
            "text": text,
        }
        for job_id, (_, text) in ordered
    ]

    print(f"\n问题：{args.question}\n")
    print(ask(args.question, hits))

    print("\n" + "=" * 40)
    print("=== 检索到的来源（RAG 可追溯性）===")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['title']}（{h['location']}）{h['url'] or ''}")


def cmd_stats(args) -> None:
    jobs_done, total_chunks = chunk_stats(args.db)
    print(f"已建索引岗位：{jobs_done} 个")
    print(f"片段总数：{total_chunks} 个")
    if jobs_done == 0:
        print("提示：先运行 python -m rag --index 建索引")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="Agent 岗位情报站 · RAG 知识库")
    parser.add_argument("--index", action="store_true", help="切块+嵌入+入库（增量）")
    parser.add_argument("--force", action="store_true", help="与 --index 连用：清空后全量重建")
    parser.add_argument("--ask", metavar="问题", help="问答模式，传入求职问题")
    parser.add_argument("--top-k", type=int, default=3, help="--ask 时返回几个岗位")
    parser.add_argument("--stats", action="store_true", help="看索引规模")
    parser.add_argument("--db", default="jobs.db")
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args()

    if args.stats:
        cmd_stats(args)
    elif args.index:
        cmd_index(args)
    elif args.ask:
        cmd_ask(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
