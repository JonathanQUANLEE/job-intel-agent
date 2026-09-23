"""evals 离线测试：指标函数与评测集自身的完整性（不需要 API Key）。"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evals.metrics import average, hit_rate_at_k, reciprocal_rank


def test_hit_rate():
    assert hit_rate_at_k(["a", "b", "c"], ["b"], 5) == 1
    assert hit_rate_at_k(["a", "b", "c"], ["z"], 5) == 0
    assert hit_rate_at_k(["a", "b", "c", "d", "e", "f"], ["f"], 5) == 0  # k 截断生效


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert abs(reciprocal_rank(["x", "b"], ["b"]) - 0.5) < 1e-9
    assert reciprocal_rank(["x", "y"], ["z"]) == 0.0


def test_average_empty():
    assert average([]) == 0


def test_dataset_integrity():
    """评测集自身质检：30 条、字段齐全、期望岗位都真实存在于库中。"""
    data = json.loads((Path(__file__).with_name("..") / "evals" / "dataset.json").read_text(encoding="utf-8"))
    cases = data["cases"]
    assert len(cases) == 30, f"评测集应为 30 条，实际 {len(cases)}"
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == 30, "评测集 id 有重复"

    import sqlite3
    conn = sqlite3.connect(Path(__file__).with_name("..") / "jobs.db")
    known = {r[0] for r in conn.execute("SELECT job_id FROM jobs")}
    conn.close()
    for c in cases:
        assert c["question"].strip(), f"case {c['id']} 问题为空"
        assert c["expect_job_ids"], f"case {c['id']} 没有期望岗位"
        missing = [j for j in c["expect_job_ids"] if j not in known]
        assert not missing, f"case {c['id']} 期望岗位不在库里: {missing}"
