"""离线检索层测试：BM25 分词、检索、以及与统一检索入口的降级衔接。

这一层是"没有 API Key 也能用"的保障，必须离线可测、不联网。
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crawler.normalize import FIELDS
from crawler.storage import save_jobs
from rag.offline import BM25Index, extract_city, search_offline, tokenize
from rag.retriever import MODE_OFFLINE, retrieve


def _row(job_id: str, title: str, raw_text: str = "x", **kw) -> dict:
    base = {f: "" for f in FIELDS}
    base.update({
        "job_id": job_id, "source": "tencent", "title": title, "location": "深圳",
        "category": "技术", "company": "腾讯", "department": "TEG",
        "experience": "", "url": "", "raw_text": raw_text,
        "fetched_at": "2026-09-18T00:00:00",
    })
    base.update(kw)
    return base


def test_tokenize_chinese_and_english():
    """中文出单字+双字、英文出整词，停用词被过滤。"""
    tokens = tokenize("我会 Python 和 LangChain，想做 Agent 开发")
    assert "python" in tokens
    assert "langchain" in tokens
    assert "agent" in tokens
    assert "大模型" not in tokens          # 没出现的词当然不该有
    assert "的" not in tokens              # 停用词过滤
    bigrams = tokenize("大模型推理")
    assert "大模" in bigrams and "模型" in bigrams   # 中文双字组合生效


def test_bm25_prefers_relevant_doc():
    """相关文档得分应显著高于无关文档。"""
    docs = [
        ("a", "深圳", "大模型 推理 优化 工程师，熟悉 vLLM 和 TensorRT"),
        ("b", "北京", "前端 工程师，熟悉 React 和 TypeScript"),
    ]
    idx = BM25Index(docs)
    hits = idx.search("大模型推理优化")
    assert hits, "应至少命中一条"
    assert hits[0][1] == "a", "相关文档应排第一"


def test_extract_city():
    assert extract_city("深圳有哪些大模型岗位？") == "深圳"
    assert extract_city("北京有 NLP 岗位吗") == "北京"
    assert extract_city("有哪些 Agent 岗位") == ""


def test_city_boost_prefers_target_city():
    """问题点名城市时，该城市的岗位应优先于外地的同名岗位。"""
    docs = [
        ("bj", "北京", "大模型推理工程师 大模型推理优化"),   # 正文更相关
        ("sz", "深圳", "大模型推理工程师"),                  # 相关度略低但在目标城市
    ]
    idx = BM25Index(docs)
    hits = idx.search("深圳的大模型推理岗位")
    assert hits[0][1] == "sz", "城市加权后，深圳岗位应排第一"


def test_search_offline_end_to_end():
    """离线检索入口：写库 → 搜关键词 → 命中的是目标岗位。"""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "t.db")
        save_jobs([
            _row("j1", "大模型推理工程师", "负责大模型推理性能优化，熟悉 vLLM"),
            _row("j2", "前端工程师", "负责 React 页面开发"),
            _row("j3", "数据分析师", "负责业务数据报表"),
        ], db)
        hits = search_offline("大模型推理", db, top_k=3)
        assert hits, "应该有命中"
        assert hits[0][1] == "j1"
        assert 0 < hits[0][0] <= 1.0, "分数应归一化到 0~1"


def test_retriever_falls_back_to_offline():
    """没有 API Key 时，统一检索入口必须自动降级到离线通道，而不是报错。"""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "t.db")
        save_jobs([
            _row("j1", "Agent 开发工程师", "负责 Agent 框架与工具链开发，熟悉 Python"),
            _row("j2", "测试工程师", "负责用例编写"),
        ], db)
        results, mode = retrieve("Agent 开发", db, top_k=2, want_mode=True)
        assert mode == MODE_OFFLINE
        assert results, "离线通道也要能出结果"
        # 统一检索结果必须带全展示字段，前端据此渲染卡片
        top = results[0]
        for key in ("job_id", "title", "location", "company", "url", "text", "score", "mode"):
            assert key in top, f"缺少字段 {key}"


if __name__ == "__main__":
    test_tokenize_chinese_and_english()
    test_bm25_prefers_relevant_doc()
    test_extract_city()
    test_city_boost_prefers_target_city()
    test_search_offline_end_to_end()
    test_retriever_falls_back_to_offline()
    print("6 tests passed ✓")
