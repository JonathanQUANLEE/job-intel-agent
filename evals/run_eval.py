"""评测脚本：python -m evals [--top-k 5] [--skip-faithful]

流程：30 条评测问题 → 逐条嵌入检索 → 召回指标（hit@k / MRR，纯计算）
→ 可选忠实度（LLM-as-judge：判定回答是否全部有依据）→ 汇总报告。

忠实度为什么用 LLM 判卷而不是规则？
"回答是否忠于片段"是语义判断，关键词规则误伤太多（模型换种说法就算不忠）。
LLM-as-judge 是业界评测忠实度的主流做法，判卷 prompt 里同时给出原文，
只回答"是/否 + 理由"，输出限定 JSON 便于程序解析。
"""

import argparse
import json
import time
from pathlib import Path

import requests

from rag.config import API_KEY, BASE_URL, LLM_MODEL
from rag.embedder import embed_texts
from rag.llm import ask
from rag.store import search
from .metrics import average, hit_rate_at_k, reciprocal_rank

_DATASET = Path(__file__).with_name("dataset.json")

_JUDGE_PROMPT = (
    "你是评测裁判。判断'回答'中的每个具体事实（岗位名、城市、技能要求等）"
    "是否都能在'检索片段'中找到依据。\n"
    "只输出 JSON：{\"faithful\": true/false, \"reason\": \"一句话理由\"}\n"
    "回答里出现片段没有的岗位或要求 = false；说'没找到/信息不足'不算不忠实。\n\n"
    "【检索片段】\n{snippets}\n\n【回答】\n{answer}"
)


def load_cases() -> list[dict]:
    """载入评测集。数据集与代码同目录（打包简单、路径稳定）。"""
    data = json.loads(_DATASET.read_text(encoding="utf-8"))
    return data["cases"]


def judge_faithfulness(snippets: str, answer: str) -> dict:
    """LLM 判卷一次，返回 {faithful, reason}。判卷失败按不忠实计（从严）。"""
    if not API_KEY:
        return {"faithful": False, "reason": "判卷器未配置 Key，按不忠实计"}
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "user", "content": _JUDGE_PROMPT.format(snippets=snippets, answer=answer)}
                ],
                "temperature": 0,
            },
            timeout=(5, 60),
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
        # 模型偶尔在 JSON 外包裹说明文字，抠出第一个 {...} 再解析
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start : end + 1])
    except Exception as e:  # noqa: BLE001 —— 判卷失败不中断整个评测
        return {"faithful": False, "reason": f"判卷异常：{type(e).__name__}"}


def run(top_k: int, skip_faithful: bool, db_path: str) -> dict:
    """跑全量评测，返回报告 dict（也由 CLI 打印、写盘）。"""
    cases = load_cases()
    hits, rrs, faiths, details = [], [], [], []

    for n, case in enumerate(cases, 1):
        vec = embed_texts([case["question"]])[0]
        raw = search(vec, db_path, top_k=top_k * 2)

        # 与线上同策略：按岗位去重取 top_k（评测和产品用同一套检索语义，数字才有意义）
        best: dict[str, tuple] = {}
        for score, job_id, text in raw:
            if job_id not in best or score > best[job_id][0]:
                best[job_id] = (score, job_id, text)
        ordered = sorted(best.values(), key=lambda h: h[0], reverse=True)[:top_k]
        retrieved_ids = [job_id for _, job_id, _ in ordered]

        hit = hit_rate_at_k(retrieved_ids, case["expect_job_ids"], top_k)
        rr = reciprocal_rank(retrieved_ids, case["expect_job_ids"])
        hits.append(hit)
        rrs.append(rr)

        row = {
            "id": case["id"],
            "question": case["question"],
            "hit": hit,
            "first_hit_rank": None if rr == 0 else round(1 / rr),
            "retrieved": retrieved_ids,
        }

        if not skip_faithful:
            snippets = "\n\n".join(f"[{i}] {t}" for i, (_, _, t) in enumerate(ordered, 1))
            answer = ask(case["question"], [
                {"job_id": j, "title": "", "location": "", "url": "", "text": t}
                for _, j, t in ordered
            ])
            verdict = judge_faithfulness(snippets, answer)
            faiths.append(int(bool(verdict.get("faithful"))))
            row["faithful"] = verdict.get("faithful")
            row["judge_reason"] = verdict.get("reason", "")

        details.append(row)
        print(f"[{n:>2}/{len(cases)}] hit={hit} rr={rr:.2f}  {case['question']}")

    report = {
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cases": len(cases),
        "top_k": top_k,
        "hit_rate_at_k": average(hits),
        "mrr": average(rrs),
        "faithfulness": None if skip_faithful else average(faiths),
        "details": details,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="evals", description="RAG 评测：召回 + 忠实度")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-faithful", action="store_true", help="跳过忠实度判卷（省额度/快速回归）")
    parser.add_argument("--db", default="jobs.db")
    parser.add_argument("--out", default="evals/report.json")
    args = parser.parse_args()

    report = run(args.top_k, args.skip_faithful, args.db)

    print("\n" + "=" * 46)
    print(f"评测集 {report['n_cases']} 条  top_k={report['top_k']}")
    print(f"召回 hit_rate@{report['top_k']} : {report['hit_rate_at_k']}")
    print(f"MRR                 : {report['mrr']}")
    if report["faithfulness"] is not None:
        print(f"忠实度 faithfulness  : {report['faithfulness']}")

    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"明细已写入 {args.out}")


if __name__ == "__main__":
    main()
