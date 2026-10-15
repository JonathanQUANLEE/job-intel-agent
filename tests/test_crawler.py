"""最小测试集：不联网也能跑，验证清洗与存储的核心逻辑。

运行：python -m pytest tests/ -v
（没装 pytest 的话，直接 python tests/test_crawler.py 也行）
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crawler.normalize import FIELDS, normalize_tencent
from crawler.storage import get_job, list_jobs, locations, save_jobs, stats


def _row(job_id: str = "tencent-1", **kw) -> dict:
    """构造一条符合当前 schema 的完整记录（字段变了不用逐个测试文件改）。"""
    base = {f: "" for f in FIELDS}
    base.update({
        "job_id": job_id, "source": "tencent", "title": "A", "location": "深圳",
        "category": "技术", "company": "腾讯", "department": "TEG",
        "experience": "三年以上工作经验", "url": "", "raw_text": "x",
        "fetched_at": "2026-09-18T00:00:00",
    })
    base.update(kw)
    return base


def test_normalize_tencent_basic():
    row = normalize_tencent({
        "RecruitPostId": 12345,
        "RecruitPostName": " 大模型应用工程师 ",
        "LocationName": "深圳",
        "CategoryName": "技术类",
        "BGName": "TEG",
        "PostURL": "//careers.tencent.com/job/12345.html",
        "Responsibility": "负责大模型应用研发",
    })
    assert row["job_id"] == "tencent-12345"
    assert row["title"] == "大模型应用工程师"      # 已去空格
    assert row["url"].startswith("https:")        # 相对协议已补全
    assert row["source"] == "tencent"
    assert row["department"] == "TEG"


def test_normalize_merges_requirement():
    """详情接口补来的 Requirement 要拼进正文——技能匹配靠它。"""
    row = normalize_tencent({
        "RecruitPostId": 1, "RecruitPostName": "算法工程师",
        "Responsibility": "负责模型训练", "Requirement": "熟悉 PyTorch",
    })
    assert "【任职要求】" in row["raw_text"]
    assert "PyTorch" in row["raw_text"]


def test_save_jobs_dedup():
    rows = [_row()]
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        inserted, _ = save_jobs(rows, db)
        again_inserted, skipped = save_jobs(rows, db)          # 同一批再存一遍
        assert (inserted, again_inserted, skipped) == (1, 0, 1)  # 主键去重生效
        assert stats(db)[0][1] == 1


def test_save_jobs_updates_longer_text():
    """重爬拿到更完整的 JD 时，应更新正文，而不是被 IGNORE 掉。"""
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        save_jobs([_row(raw_text="短")], db)
        save_jobs([_row(raw_text="这是一段明显更长的岗位描述，用于验证覆盖逻辑")], db)
        assert get_job(db, "tencent-1")["raw_text"].startswith("这是一段明显更长")


def test_list_and_locations():
    """浏览页依赖的查询接口：关键词过滤 + 城市统计。

    注意：save_jobs 必须显式传 db —— 漏传会写进默认的 jobs.db，
    既污染真实数据库，又让本测试读不到数据（踩过一次）。
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        save_jobs([
            _row("tencent-1", title="大模型工程师", location="深圳"),
            _row("tencent-2", title="前端工程师", location="北京"),
        ], db)
        matched = list_jobs(db, keyword="大模型")
        assert len(matched) == 1
        assert matched[0]["job_id"] == "tencent-1"
        assert len(list_jobs(db, location="北京")) == 1
        assert {l["location"] for l in locations(db)} == {"深圳", "北京"}


if __name__ == "__main__":
    test_normalize_tencent_basic()
    test_normalize_merges_requirement()
    test_save_jobs_dedup()
    test_save_jobs_updates_longer_text()
    test_list_and_locations()
    print("5 tests passed ✓")
