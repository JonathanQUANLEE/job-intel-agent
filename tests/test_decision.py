"""决策质量评测的单元测试包装：把 15 条行为契约纳入 pytest。

为什么要在 pytest 里再跑一遍 evals.decision？
evals.decision 是给人看的报告（打印 + 写 JSON），pytest 是给 CI/提交前用的门禁。
把同一个用例集包成测试，提交前 `pytest` 就能拦住"行为回退"——
比人工记得去跑评测可靠得多。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from evals.decision import _load, check_case


@pytest.mark.parametrize("case", _load(), ids=lambda c: c["id"])
def test_decision_behaviour(case):
    """每条决策用例都是独立断言，失败时 pytest 会精确指出是哪条契约破了。"""
    ok, failures = check_case(case, "jobs.db")
    assert ok, f"{case['desc']} → " + "；".join(failures)
