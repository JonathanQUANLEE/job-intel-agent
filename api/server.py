"""FastAPI 服务：POST /ask 返回 SSE 流式问答。

为什么流式（SSE）而不是一次性 JSON？
生成一个完整回答要几秒，"打字机效果"把首字延迟从'整答完成'降到
'第一个 token'，这是所有 AI 产品标配体验的原因。
SSE = Server-Sent Events：HTTP 长连接上服务端单向推 event，
比 WebSocket 简单（问答场景不需要双向）。

事件协议（前端按 event 类型分流）：
  event: sources  → 检索到的岗位列表（回答前先亮证据）
  event: token    → 逐段回答文本
  event: done     → 结束标记
  event: error    → 出错信息
"""

import json
import sqlite3
from contextlib import closing

import requests
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag.config import API_KEY, BASE_URL, LLM_MODEL
from rag.embedder import embed_texts
from rag.llm import _SYSTEM_PROMPT
from rag.store import search

app = FastAPI(title="Agent 岗位情报站", version="0.1.0")


class AskRequest(BaseModel):
    """请求体。Pydantic 在边界上做校验（缺字段/类型错 422 自动返回）。"""
    question: str = Field(min_length=2, max_length=200)
    top_k: int = Field(default=3, ge=1, le=10)


def _retrieve(question: str, top_k: int) -> list[dict]:
    """检索 + 岗位去重 + 元数据补全（与 CLI 同策略，保证产品内行为一致）。"""
    vec = embed_texts([question])[0]
    raw = search(vec, "jobs.db", top_k=top_k * 2)
    best: dict[str, tuple] = {}
    for score, job_id, text in raw:
        if job_id not in best or score > best[job_id][0]:
            best[job_id] = (score, job_id, text)
    ordered = sorted(best.values(), key=lambda h: h[0], reverse=True)[:top_k]

    ids = [job_id for _, job_id, _ in ordered]
    meta: dict[str, dict] = {}
    if ids:
        placeholders = ", ".join("?" * len(ids))
        with closing(sqlite3.connect("jobs.db")) as conn:
            rows = conn.execute(
                f"SELECT job_id, title, location, url FROM jobs WHERE job_id IN ({placeholders})",
                ids,
            ).fetchall()
        meta = {r[0]: {"title": r[1], "location": r[2], "url": r[3]} for r in rows}

    return [
        {
            "job_id": job_id,
            "score": round(score, 3),
            "title": meta.get(job_id, {}).get("title", ""),
            "location": meta.get(job_id, {}).get("location", ""),
            "url": meta.get(job_id, {}).get("url", ""),
            "text": text,
        }
        for score, job_id, text in ordered
    ]


def _sse(event: str, data) -> str:
    """SSE 一帧：event 行 + data 行。data 序列化为 JSON。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
def health() -> dict:
    """健康检查：部署后先打这个确认服务活着。"""
    return {"status": "ok", "llm_key": bool(API_KEY)}


@app.post("/ask")
def ask(req: AskRequest) -> StreamingResponse:
    """流式问答。generator 内 yield SSE 帧——
    FastAPI 的 StreamingResponse 会把 generator 的产出持续推给客户端。"""

    def gen():
        try:
            if not API_KEY:
                yield _sse("error", {"message": "服务端未配置 SILICONFLOW_API_KEY"})
                yield _sse("done", {})
                return

            hits = _retrieve(req.question, req.top_k)
            if not hits:
                yield _sse("sources", [])
                yield _sse("token", "岗位库为空，请先运行采集与索引。")
                yield _sse("done", {})
                return

            # 先推来源：让前端在等待第一个字时就看到"依据"
            yield _sse(
                "sources",
                [
                    {
                        "job_id": h["job_id"],
                        "title": h["title"],
                        "location": h["location"],
                        "url": h["url"],
                        "score": h["score"],
                    }
                    for h in hits
                ],
            )

            snippets = "\n\n".join(
                f"[{i + 1}] 岗位：{h['title']}（{h['location']}，编号 {h['job_id']}）\n{h['text']}"
                for i, h in enumerate(hits)
            )
            user_content = (
                f"用户问题：{req.question}\n\n以下是检索到的真实岗位片段，请据此回答：\n{snippets}"
            )

            # 流式调用：stream=True，逐行读 SSE，把每个增量 token 推出去
            with requests.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.3,
                    "stream": True,
                },
                timeout=(5, 120),
                stream=True,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield _sse("token", {"text": delta})

            yield _sse("done", {})
        except Exception as e:  # noqa: BLE001 —— SSE 已开头，异常只能走事件通道
            yield _sse("error", {"message": f"{type(e).__name__}: {e}"})
            yield _sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream")
