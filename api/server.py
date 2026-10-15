"""FastAPI 服务：SSE 流式问答 + 岗位浏览 + 页面托管。

接口一览：
  GET  /                 直接打开聊天界面（单文件 HTML，免构建）
  POST /ask              SSE 流式问答（打字机效果）
  GET  /jobs             岗位列表（关键词 + 城市筛选，浏览页用）
  GET  /jobs/{job_id}    单个岗位完整 JD（详情弹窗用）
  GET  /meta             城市分布、岗位总数、是否已配 Key（前端初始化用）

为什么要流式（SSE）而不是一次性 JSON？
生成一个完整回答要几秒，"打字机效果"把首字延迟从'整答完成'降到
'第一个 token'，这是所有 AI 产品标配体验的原因。
SSE = Server-Sent Events：HTTP 长连接上服务端单向推 event，
比 WebSocket 简单（问答场景不需要双向）。

事件协议（前端按 event 类型分流）：
  event: meta     → 本次检索走的通道（vector/offline）+ 来源条数
  event: sources  → 检索到的岗位列表（回答前先亮证据）
  event: token    → 逐段回答文本
  event: done     → 结束标记
  event: error    → 出错信息
"""

import json
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from crawler.storage import count_jobs, get_job, list_jobs, locations
from rag import config as rag_config
from rag.config import BASE_URL, LLM_MODEL, has_key
from rag.llm import _SYSTEM_PROMPT
from rag.retriever import retrieve

app = FastAPI(title="Agent 岗位情报站", version="0.2.0")

_INDEX_HTML = Path(__file__).with_name("index.html")


class AskRequest(BaseModel):
    """请求体。Pydantic 在边界上做校验（缺字段/类型错 422 自动返回）。"""
    question: str = Field(min_length=2, max_length=200)
    top_k: int = Field(default=5, ge=1, le=10)


def _sse(event: str, data) -> str:
    """SSE 一帧：event 行 + data 行。data 序列化为 JSON。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/")
def index() -> FileResponse:
    """把聊天界面直接托管出去——用户打开 http://127.0.0.1:8000 就能用。"""
    if not _INDEX_HTML.exists():
        raise HTTPException(404, "index.html 缺失")
    return FileResponse(_INDEX_HTML)


@app.get("/health")
def health() -> dict:
    """健康检查：部署后先打这个确认服务活着。"""
    return {"status": "ok", "llm_key": has_key(), "jobs": count_jobs()}


@app.get("/meta")
def meta() -> dict:
    """前端初始化：岗位总数、城市分布、是否有 Key（决定界面提示文案）。"""
    return {"jobs": count_jobs(), "locations": locations(), "has_key": has_key()}


@app.get("/jobs")
def jobs(
    keyword: str = Query("", max_length=60),
    location: str = Query("", max_length=30),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> dict:
    """岗位浏览：条件筛选 + 分页。"""
    items = list_jobs(keyword=keyword, location=location, limit=limit, offset=offset)
    return {"items": items, "limit": limit, "offset": offset, "count": len(items)}


@app.get("/jobs/{job_id}")
def job_detail(job_id: str) -> dict:
    """单个岗位完整信息（详情弹窗）。"""
    job = get_job("jobs.db", job_id)
    if not job:
        raise HTTPException(404, "岗位不存在")
    return job


def _fallback_answer(hits: list[dict]) -> str:
    """没有 Key 时的本地作答：把检索结果组织成可读的推荐清单。

    这不是"假回答"——它诚实声明"当前是本地离线模式"，
    并逐条列出岗位、地点、相关度，对用户仍有真实价值（确实能定位岗位），
    同时明确引导"配上 Key 可获得 AI 分析"，把降级变成转化引导。
    """
    if not hits:
        return (
            "没有检索到相关岗位。\n\n"
            "可能原因：岗位库为空（先运行 `python -m crawler --source tencent "
            "--keyword \"大模型\" --max-pages 3` 采集数据），或者换个说法再问一次。"
        )
    lines = [
        "当前是**本地离线检索模式**（未配置 API Key，无法调用大模型做分析）。",
        f"我按关键词相关度为你找到 {len(hits)} 个可能匹配的岗位：\n",
    ]
    for i, h in enumerate(hits, 1):
        lines.append(
            f"**{i}. {h['title']}**（{h['location']}｜相关度 {h['score']:.2f}）\n"
            f"　{h['text'][:160].replace(chr(10), ' ')}…"
        )
    lines.append(
        "\n---\n"
        "想要 AI 逐条分析匹配度、指出技能差距？在项目根目录新建 `.env` 写入 "
        "`SILICONFLOW_API_KEY=sk-xxx` 后重启服务即可。"
    )
    return "\n".join(lines)


@app.post("/ask")
def ask(req: AskRequest) -> StreamingResponse:
    """流式问答。generator 内 yield SSE 帧——
    FastAPI 的 StreamingResponse 会把 generator 的产出持续推给客户端。"""

    def gen():
        try:
            hits, mode = retrieve(req.question, "jobs.db", top_k=req.top_k, want_mode=True)

            yield _sse("meta", {
                "mode": mode, "has_key": has_key(), "n_sources": len(hits),
            })

            if not hits:
                yield _sse("sources", [])
                yield _sse("token", {"text": "岗位库为空或没匹配到内容。"})
                yield _sse("done", {})
                return

            # 先推来源：让前端在等第一个字时就看到"依据"
            yield _sse("sources", [
                {
                    "job_id": h["job_id"], "title": h["title"], "location": h["location"],
                    "company": h["company"], "department": h["department"],
                    "experience": h["experience"], "url": h["url"], "score": h["score"],
                }
                for h in hits
            ])

            # 无 Key：走本地降级作答，把检索结果组织成清单推给前端
            if not has_key():
                text = _fallback_answer(hits)
                # 按小块推，复刻打字机节奏（本地模式没有 token 流，这里造一个）
                for i in range(0, len(text), 12):
                    yield _sse("token", {"text": text[i : i + 12]})
                yield _sse("done", {})
                return

            # 有 Key：正常走 LLM 流式生成
            snippets = "\n\n".join(
                f"[{i + 1}] 岗位：{h['title']}（{h['location']}，编号 {h['job_id']}）\n{h['text']}"
                for i, h in enumerate(hits)
            )
            user_content = (
                f"用户问题：{req.question}\n\n以下是检索到的真实岗位片段，请据此回答：\n{snippets}"
            )

            with requests.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {rag_config.API_KEY}",
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
