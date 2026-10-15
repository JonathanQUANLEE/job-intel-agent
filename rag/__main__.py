"""CLI 入口：python -m rag --index / --ask "..." / --stats

三步走通 RAG：
  --index   把 jobs 表里的岗位描述切块 + 嵌入 + 入库（增量，重复跑不浪费额度）
  --ask     问题 → 检索 → LLM 带引用回答（无 Key 时自动降级本地检索）
  --stats   看已建索引的岗位数和片段总数
"""

import argparse

from . import __version__
from .chunker import chunk_text
from .config import has_key
from .embedder import embed_texts
from .llm import ask
from .retriever import MODE_OFFLINE, retrieve
from .store import chunk_stats, pending_jobs, reset_chunks, save_chunks


def cmd_index(args) -> None:
    if not has_key():
        print(
            "没有检测到 SILICONFLOW_API_KEY —— 建立向量索引需要调嵌入接口。\n"
            "跳过建索引也不影响使用：问答会自动走本地关键词检索。\n"
            "想启用语义检索，请在项目根目录 .env 里写入 SILICONFLOW_API_KEY=sk-xxx。"
        )
        return

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
    question = args.ask
    hits, mode = retrieve(question, args.db, top_k=args.top_k, want_mode=True)

    if not hits:
        print("没检索到相关岗位。可能是岗位库为空——先运行 python -m crawler 采集数据。")
        return

    if mode == MODE_OFFLINE:
        why = "未配置 API Key" if not has_key() else "向量索引未建立或调用失败"
        print(f"[提示] 当前为本地关键词检索模式（{why}）。配 Key + 建索引可获得语义检索。\n")

    print(f"\n问题：{question}\n")
    print(ask(question, hits))

    print("\n" + "=" * 40)
    print("=== 检索到的来源（RAG 可追溯性）===")
    for i, h in enumerate(hits, 1):
        extra = "，".join(x for x in [h.get("department", ""), h.get("experience", "")] if x)
        suffix = f"（{extra}）" if extra else ""
        print(f"[{i}] {h['title']}（{h['location']}{suffix}）相关度 {h['score']:.2f}")
        print(f"    {h['url'] or '（无链接）'}")


def cmd_stats(args) -> None:
    jobs_done, total_chunks = chunk_stats(args.db)
    print(f"已建索引岗位：{jobs_done} 个")
    print(f"片段总数：{total_chunks} 个")
    if jobs_done == 0:
        print("提示：先运行 python -m rag --index 建索引（没有 Key 也能用本地检索）")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag", description="Agent 岗位情报站 · RAG 知识库")
    parser.add_argument("--index", action="store_true", help="切块+嵌入+入库（增量）")
    parser.add_argument("--force", action="store_true", help="与 --index 连用：清空后全量重建")
    parser.add_argument("--ask", metavar="问题", help="问答模式，传入求职问题")
    parser.add_argument("--top-k", type=int, default=5, help="--ask 时返回几个岗位")
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
