"""FastAPI 服务：SSE 流式问答 + 岗位浏览 + 页面托管。

接口一览：
  GET  /                 直接打开聊天界面（单文件 HTML，免构建）
  POST /ask              SSE 流式问答（打字机效果 + 推理轨迹）
  GET  /jobs             岗位列表（关键词 + 城市筛选，浏览页用）
  GET  /jobs/{job_id}    单个岗位完整 JD（详情弹窗用）
  GET  /meta             城市分布、岗位总数、是否已配 Key（前端初始化用）

为什么是"推理轨迹"而不只是"回答"？
择岗顾问的价值不只是给结论，还要说清"为什么是这几个"。所以 SSE 除了推回答文本，
还推 plan 事件（agent 每一步的决策）和 recommendations 事件（带分数和技能明细的
结构化推荐）。前端把轨迹渲染成一条时间线，用户能看到 agent 先抽了什么条件、
检索了多少候选、谁排第一、差哪项技能——这是通用编程 agent 给不了的透明度。

事件协议（前端按 event 类型分流）：
  event: meta            → 本次检索走的通道（vector/offline）
  event: profile         → 从问题里抽出的择岗画像（城市/方向/技能…）
  event: plan            → 推理轨迹，逐步推送（抽槽→检索→打分→成稿）
  event: clarify         → 信息不足，agent 在追问（前端高亮显示）
  event: recommendations → 结构化推荐：每条含匹配分、命中/缺失技能、依据
  event: token           → 逐段回答文本
  event: done            → 结束标记
  event: error           → 出错信息（含降级提示）
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.graph import run_agent
from agent.profile import Profile, Slot
from crawler.storage import count_jobs, get_job, list_jobs, locations
from rag.config import has_key
from rag.retriever import MODE_OFFLINE

app = FastAPI(title="Agent 岗位情报站", version="0.3.0")

_INDEX_HTML = Path(__file__).with_name("index.html")


class AskRequest(BaseModel):
    """请求体。Pydantic 在边界上做校验（缺字段/类型错 422 自动返回）。

    profile 字段支持多轮：前端把上一轮返回的画像存着，下一轮带上，
    agent 就能累积条件（第一次说"深圳"、第二次说"Agent"，合起来才够）。
    """
    question: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=5, ge=1, le=10)
    profile: dict = Field(default_factory=dict)


def _sse(event: str, data) -> str:
    """SSE 一帧：event 行 + data 行。data 序列化为 JSON。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _profile_from_dict(d: dict) -> Profile:
    """把前端传来的画像 dict 还原成 Profile（跨轮沿用）。"""
    p = Profile()
    for name in ("city", "direction", "experience", "skills", "salary"):
        if d.get(name):
            setattr(p, name, Slot(value=str(d[name]), filled=True, source="user"))
    return p


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


def _recommendations(state: dict) -> list[dict]:
    """把图输出的 (job, Score) 转成前端能直接渲染的结构化推荐。"""
    out = []
    for job, s in state.get("scored") or []:
        out.append({
            "job_id": job["job_id"],
            "title": job["title"],
            "location": job["location"],
            "company": job.get("company", ""),
            "department": job.get("department", ""),
            "experience": job.get("experience", ""),
            "url": job.get("url", ""),
            "score": s.total,
            "grade": s.grade(),
            "matched_skills": s.matched_skills,
            "missing_skills": s.missing_skills,
            "reasons": s.reasons,
        })
    return out


def _chunks(text: str, size: int = 14):
    """把文本切成小块，复刻打字机节奏（模型润色已自带流，模板成稿靠这个造节奏）。"""
    for i in range(0, len(text), size):
        yield text[i : i + size]


@app.post("/ask")
def ask(req: AskRequest) -> StreamingResponse:
    """流式择岗。跑完整决策图，把轨迹和结构化推荐逐步推给前端。

    注意：这里**先跑图再推流**，而不是边跑边推——因为要拿到完整的 trace
    才能保证"轨迹顺序 = 实际执行顺序"，不会出现"先看到成稿再看到检索"的错乱。
    图的单个节点很快（毫秒级检索 + 规则打分），只有最后的模型润色是慢的，
    但润色不走流式（一次性生成），所以整体延迟仍然可控。
    """

    def gen():
        try:
            profile = _profile_from_dict(req.profile or {})
            state = run_agent(req.question, "jobs.db", profile=profile)

            yield _sse("meta", {
                "has_key": has_key(),
                "mode": "vector" if has_key() else MODE_OFFLINE,
                "n_sources": len(state.get("candidates") or []),
            })

            # 选岗画像：让用户看到 agent 理解了什么（理解错了能当场纠正）
            yield _sse("profile", state["profile"].as_dict())

            # 推理轨迹：逐步推送，前端渲染成时间线
            for i, step in enumerate(state.get("trace") or [], 1):
                yield _sse("plan", {"index": i, "text": step})

            # 追问分支：前端高亮提示 + 留住画像
            if state.get("need_clarify"):
                yield _sse("clarify", {
                    "text": state["answer"],
                    "profile": state["profile"].as_dict(),
                })
                for piece in _chunks(state["answer"]):
                    yield _sse("token", {"text": piece})
                yield _sse("done", {})
                return

            recs = _recommendations(state)
            if recs:
                yield _sse("recommendations", recs)
                yield _sse("gap", state.get("gap") or {})

            for piece in _chunks(state["answer"]):
                yield _sse("token", {"text": piece})
            yield _sse("done", {})
        except Exception as e:  # noqa: BLE001 —— SSE 已开头，异常只能走事件通道
            yield _sse("error", {"message": f"{type(e).__name__}: {e}"})
            yield _sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream")
