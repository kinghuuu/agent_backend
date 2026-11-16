"""终止策略单测（Day 05 练习 1）：纯逻辑模块，不需要数据库与模型。

运行：``python -m pytest tests/test_termination.py -p no:cacheprovider``
（在工作目录 agent_backend 下执行；``-p no:cacheprovider`` 避免生成 .pytest_cache）
"""

import pytest

from app.modules.study.services import termination
from app.modules.study.services.termination import (
    LoopGuard,
    StopReason,
    action_fp,
    extract_entities,
    looks_grounded,
    progress_fp,
)


def _action(name: str, args: dict) -> dict:
    """构造一条动作记录（与主循环喂给 guard 的结构一致）。"""
    return {"name": name, "args": args, "fp": action_fp(name, args), "ok": True}


def test_repeated_action_triggers_on_fourth_call() -> None:
    """(a) 同一动作第 4 次触发 REPEATED_ACTION（max_same_action=3）。"""
    guard = LoopGuard(max_same_action=3)
    action = _action("search_orders", {"user_id": "U000123"})
    # 前三次结果都在变，确保不是因为"无进展"提前退出
    for i in range(3):
        reason = guard.check([action], [{"ok": True, "total": i}])
        assert reason is None, f"第 {i + 1} 次不应触发"

    reason = guard.check([action], [{"ok": True, "total": 99}])
    assert reason is StopReason.REPEATED_ACTION


def test_no_progress_triggers_on_three_identical_results() -> None:
    """(b) 连续三步结果指纹相同 → NO_PROGRESS（动作各不相同也照样抓住）。"""
    guard = LoopGuard(max_no_progress=2)
    # 变体自旋：动作参数不同（动作指纹不同），但结果完全相同
    empty_result = {"ok": True, "total": 0, "orders": [], "hint": "确实没有订单"}
    for i, keyword in enumerate(["上周", "最近7天", "近一周"]):
        reason = guard.check([_action("search_orders", {"range": keyword})], [empty_result])
        if i < 2:
            assert reason is None
        else:
            assert reason is StopReason.NO_PROGRESS


def test_repeatable_whitelist_allows_polling() -> None:
    """(c) 白名单内的轮询工具重复 10 次（结果一直 running）不触发，且退避指数封顶。

    这是课程练习里最值得构造的场景：构造"结果一直不变"的轮询数据，才能验证
    白名单确实同时豁免了动作计数与无进展判定——只豁免前者的话，第 3 次就误杀。
    """
    guard = LoopGuard(repeatable={"get_task_status"})
    action = _action("get_task_status", {"task_id": "T8821"})
    backoffs: list[float] = []

    for i in range(10):
        reason = guard.check([action], [{"ok": True, "task_id": "T8821", "status": "running"}])
        assert reason is None, f"轮询第 {i + 1} 次不应被终止"
        backoffs.append(guard.backoff_seconds("get_task_status"))

    # 退避序列 1,2,4,8,8,... —— 指数增长且封顶 8s
    assert backoffs[:4] == [1.0, 2.0, 4.0, 8.0]
    assert backoffs[-1] == 8.0


def test_duplicate_action_in_same_step_triggers() -> None:
    """单步内模型吐了两个完全相同的调用 → 立即触发（纯浪费，零容忍）。"""
    guard = LoopGuard()
    action = _action("current_time", {})
    reason = guard.check([action, dict(action)], [{"ok": True, "now": "x"}])
    assert reason is StopReason.REPEATED_ACTION


def test_looks_grounded() -> None:
    """第 4 层：答案引用实体才算有事实支撑；无实体的场景不判（避免误杀闲聊）。"""
    results = [{"ok": True, "total": 1, "orders": [{"order_id": "O20260012", "date": "2026-11-05"}]}]

    assert looks_grounded("订单 O20260012 已于 2026-11-05 下单", results) is True
    # 只提到金额但没引用任何硬实体 → 判为无支撑
    assert looks_grounded("您有一笔 268 元的订单", results) is False
    # 工具没有可引用实体（纯聊天）→ 不判
    assert looks_grounded("你好呀", [{"ok": True, "now": "12:00"}]) is True


def test_progress_fp_ignores_order() -> None:
    """进展指纹只看结果集合，与结果顺序无关（并行调用顺序不固定）。"""
    a = [{"ok": True, "total": 1}, {"ok": True, "now": "12:00"}]
    b = [{"ok": True, "now": "12:00"}, {"ok": True, "total": 1}]
    assert progress_fp(a) == progress_fp(b)
    assert progress_fp(a) != progress_fp([{"ok": True, "total": 2}, {"ok": True, "now": "12:00"}])


def test_extract_entities_dedupes() -> None:
    """实体抽取去重且覆盖订单号/用户号/日期。"""
    entities = extract_entities(
        [
            {"order_id": "O20260012", "user_id": "U000123", "date": "2026-11-05"},
            {"order_id": "O20260012"},
        ]
    )
    assert entities == ["O20260012", "U000123", "2026-11-05"]


@pytest.mark.parametrize(
    ("profile", "expected_steps"),
    [("simple", 6), ("complex", 20)],
)
def test_task_budget_profiles(profile: str, expected_steps: int) -> None:
    """练习 2：预算档位步数符合预期（阈值来源应是历史 trace 的 P95，这里是起点）。"""
    assert termination.TASK_BUDGETS[profile].max_steps == expected_steps
