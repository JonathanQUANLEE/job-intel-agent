"""最小测试集：不联网也能跑，验证清洗与存储的核心逻辑。

运行：cd starter_project && python -m pytest tests/ -v
（没装 pytest 的话，直接 python tests/test_crawler.py 也行）
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crawler.normalize import normalize_tencent
from crawler.storage import save_jobs, stats


def test_normalize_tencent_basic():
    row = normalize_tencent({
        "RecruitPostId": 12345,
        "RecruitPostName": " 大模型应用工程师 ",
        "LocationName": "深圳",
        "CategoryName": "技术类",
        "PostURL": "//careers.tencent.com/job/12345.html",
        "Responsibility": "负责大模型应用研发",
    })
    assert row["job_id"] == "tencent-12345"
    assert row["title"] == "大模型应用工程师"      # 已去空格
    assert row["url"].startswith("https:")        # 相对协议已补全
    assert row["source"] == "tencent"


def test_save_jobs_dedup():
    rows = [{
        "job_id": "tencent-1", "source": "tencent", "title": "A", "location": "深圳",
        "category": "技术", "url": "", "raw_text": "x", "fetched_at": "2026-09-18T00:00:00",
    }]
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "test.db")
        inserted, _ = save_jobs(rows, db)
        again_inserted, skipped = save_jobs(rows, db)          # 同一批再存一遍
        assert (inserted, again_inserted, skipped) == (1, 0, 1)  # 主键去重生效
        assert stats(db)[0][1] == 1


if __name__ == "__main__":
    test_normalize_tencent_basic()
    test_save_jobs_dedup()
    print("2 tests passed ✓")
