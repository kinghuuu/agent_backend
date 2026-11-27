"""Day 20 单测：可靠性（分类重试 / 降级阶梯 / 优雅退出 / 容量 / SLO / 混沌演练）。

运行：``python -m pytest tests/test_day20_reliability.py -p no:cacheprovider -q``

四个原则：

1. **不调模型、不起服务、不写文件**：重试的等待是模拟的、压测是排队模拟、
   SLO 的 trace 是合成的，整份单测秒级跑完；
2. **反面对照必须有断言**：确定性失败不许重试、备模型不支持 FC 必须跳过该级、
   降级答案必须不写缓存、无 checkpointer 的 run 必须被标记撕碎；
3. **数学必须可复算**：三道天花板取最小、Little's Law 的并发、烧毁率阈值，
   断言具体数字而不是「大概对」；
4. **告警与演练的判定要能翻转**：同一份数据，改了样本量 / 开关就必须换结论——
   否则这些规则只是摆设。
"""

import json

from app.modules.study.schemas.study import (
    Day20CapacityRequest,
    Day20ChaosRequest,
    Day20LadderRequest,
    Day20RetryRequest,
    Day20ShutdownRequest,
    Day20ShutdownRun,
    Day20SloRequest,
)
from app.modules.study.services import day20_reliability as day20


# ---------- 概念表 ----------


def test_mapping_covers_course_sections() -> None:
    """概念表覆盖 §1–§8：四分类 / 三要素 / 阶梯 / 退出六步 / 三道天花板 / SLO / 告警 / 红线。"""
    data = day20.mapping()

    assert len(data["error_classes"]) == 4
    assert [row["action"] for row in data["error_classes"]] == ["retry", "fail", "fallback", "fix_prompt"]
    assert len(data["retry_rules"]) == 3
    assert [row["level"] for row in data["ladder_levels"]] == ["L0", "L1", "L2", "L3"]
    assert len(data["shutdown_steps"]) == 6
    assert len(data["capacity_ceilings"]) == 3
    assert len(data["slo_table"]) == 5
    assert len(data["alert_principles"]) == 3
    assert len(data["chaos_redlines"]) == 3
    assert len(data["pitfalls"]) == 7
    assert len(data["cheatsheet"]) == 13
    assert data["versions"]["fastapi"]


def test_static_tables_have_no_blank_cells() -> None:
    """前端直接渲染表格，空单元格会变成难看的空白列。"""
    for row in day20.CHEATSHEET:
        assert row["topic"] and row["content"] and row["key"]
    for row in day20.PITFALLS:
        assert row["no"] and row["title"] and row["symptom"] and row["fix"]
    for row in day20.SHUTDOWN_STEPS:
        assert row["no"] and row["stage"] and row["detail"] and row["if_skipped"]
    for row in day20.SLO_TABLE:
        assert row["sli"] and row["slo"] and row["source"] and row["why"]
    for key, items in day20.RUNBOOK.items():
        assert key in {"error_burn", "degraded_burn"} and len(items) == 3


# ---------- 故障分层与重试 ----------


def test_classify_maps_every_fault_to_an_action() -> None:
    """四类错误 → 四种动作；未知错误 fail-closed。"""
    assert day20.classify(day20.RateLimitError("429")) == "retry"
    assert day20.classify(day20.TransientError("timeout")) == "retry"
    assert day20.classify(day20.PermanentError("400")) == "fail"
    assert day20.classify(day20.SchemaValidationError("bad json")) == "fix_prompt"
    assert day20.classify(ValueError("???")) == "fail"


def test_backoff_grows_exponentially_and_stays_within_jitter_bounds() -> None:
    """第 n 次等待 = min(base·2^(n-1), cap) × [0.5, 1.5)。"""
    import random

    rng = random.Random(1)
    base, cap = 500, 8000
    for attempt, raw in [(1, 500), (2, 1000), (3, 2000), (6, 8000), (9, 8000)]:
        for _ in range(20):
            wait = day20.backoff_ms(attempt, base, cap, rng)
            assert raw * 0.5 - 1 <= wait <= raw * 1.5 + 1


def test_retry_transient_heals_within_budget() -> None:
    """瞬态错误重试 2 次后成功：最终 ok，等待时间只在失败的尝试上产生。"""
    result = day20.retry_demo(
        Day20RetryRequest(scenario="transient", fail_times=2, max_attempts=3, seed=3)
    )

    assert result["summary"]["succeeded"] is True
    assert result["summary"]["final"] == "ok"
    assert [row["outcome"] for row in result["timeline"]] == ["fail", "fail", "ok"]
    assert result["timeline"][0]["wait_ms"] > 0
    assert result["timeline"][2]["wait_ms"] == 0
    assert result["classification"]["action"] == "retry"


def test_retry_permanent_fails_fast_without_waiting() -> None:
    """确定性失败：只尝试 1 次、不等待、不烧钱。"""
    result = day20.retry_demo(
        Day20RetryRequest(scenario="permanent", fail_times=5, max_attempts=3)
    )

    assert len(result["timeline"]) == 1
    assert result["summary"]["final"] == "fail"
    assert result["summary"]["total_wait_ms"] == 0


def test_retry_invalid_output_patches_prompt_then_gives_up() -> None:
    """解析失败：每次重试都带 prompt 回填；耗尽后走降级出口而不是抛原始异常。"""
    result = day20.retry_demo(
        Day20RetryRequest(scenario="invalid_output", fail_times=6, max_attempts=3)
    )

    assert result["summary"]["final"] == "fallback"
    assert all("prompt_patch" in row for row in result["timeline"][:2])
    assert result["timeline"][-1]["detail"].startswith("重试次数耗尽")


def test_retry_respects_retry_after_header() -> None:
    """429 的等待时间必须 ≥ Retry-After：否则重试风暴会让限流更狠。"""
    result = day20.retry_demo(
        Day20RetryRequest(
            scenario="rate_limit",
            fail_times=1,
            max_attempts=3,
            base_ms=50,
            retry_after_ms=5000,
        )
    )

    first = result["timeline"][0]
    assert first["retry_after_ms"] == 5000
    assert first["wait_ms"] >= 5000


def test_retry_budget_guard_catches_run_level_blowup() -> None:
    """单步 3 次重试都「合法」，但 run 级总闸必须能叫停。"""
    ok = day20.retry_demo(Day20RetryRequest(llm_calls=3, elapsed_s=10))
    bad = day20.retry_demo(Day20RetryRequest(llm_calls=29, elapsed_s=179))

    assert ok["budget"]["within_budget"] is True
    assert bad["budget"]["within_budget"] is False
    assert bad["budget"]["max_llm_calls"] == 30


def test_retry_pulse_shows_jitter_flattens_the_sync_spike() -> None:
    """裸重试的 50 个并发 run 全部落在同一个等待值；带抖动后峰值被打散。"""
    result = day20.retry_demo(Day20RetryRequest(scenario="transient", fail_times=2, seed=5))
    pulse = result["pulse"]

    assert pulse["sync"]["peak_share"] == 1.0
    assert pulse["sync"]["spread_ms"] == 0
    assert pulse["jittered"]["peak_share"] < 1.0
    assert pulse["jittered"]["spread_ms"] > 0


# ---------- 降级阶梯 ----------


def test_ladder_primary_ok_keeps_quality_and_caches() -> None:
    """主路正常：level 0、非降级、答案可以回写缓存。"""
    result = day20.ladder_demo(Day20LadderRequest(primary_ok=True))

    assert result["final"]["level"] == 0
    assert result["final"]["degraded"] is False
    assert result["cache"]["written"] is True
    assert result["cache"]["store"] == [Day20LadderRequest().query]


def test_ladder_falls_back_and_never_caches_degraded_answer() -> None:
    """L1 接管：degraded=True、reason 指向备模型、缓存铁律生效。"""
    result = day20.ladder_demo(
        Day20LadderRequest(primary_ok=False, fallback_ok=True, cache_hit=True)
    )

    assert result["final"]["level"] == 1
    assert result["final"]["degraded"] is True
    assert result["final"]["reason"] == "fallback_to_gpt-4.1-mini"
    assert result["cache"]["written"] is False
    assert result["cache"]["store"] == []


def test_ladder_skips_fallback_that_cannot_call_tools() -> None:
    """能力对齐：备模型不支持 FC 时必须跳过，不能把瞬态错误换成 400。"""
    result = day20.ladder_demo(
        Day20LadderRequest(
            needs_tools=True,
            fallback_supports_tools=False,
            fallback_ok=True,
            cache_hit=False,
        )
    )

    l1 = result["attempts"][1]
    assert l1["result"] == "skipped"
    assert "能力对齐" in l1["detail"]
    # 跳级后没有缓存 → 直接 L3
    assert result["final"]["level"] == 3


def test_ladder_cache_hit_and_final_apology_are_distinguishable() -> None:
    """L2 命中与 L3 认输都必须带 reason，前端才能分开展示。"""
    cached = day20.ladder_demo(Day20LadderRequest(fallback_ok=False, cache_hit=True))
    apology = day20.ladder_demo(Day20LadderRequest(fallback_ok=False, cache_hit=False))

    assert cached["final"]["level"] == 2
    assert cached["final"]["reason"] == "primary_unavailable_cache_hit"
    assert apology["final"]["level"] == 3
    assert apology["final"]["reason"] == "all_levels_failed"
    assert "抱歉" in apology["final"]["answer"]
    assert apology["cache"]["written"] is False


def test_probe_window_needs_consecutive_successes() -> None:
    """探活窗口：中间一次失败要清零；连续 3 次成功才回主路（防乒乓）。"""
    result = day20.ladder_demo(
        Day20LadderRequest(
            primary_ok=False,
            fallback_ok=False,
            recover_after=3,
            probe_sequence=[True, True, False, True, True, True],
        )
    )

    states = [row["degraded"] for row in result["probe"]]
    assert states == [True, True, True, True, True, False]


def test_probe_window_flips_back_on_failure() -> None:
    """回主路后再失败 → 立刻重新进入降级（不能只看历史成功率）。"""
    result = day20.ladder_demo(
        Day20LadderRequest(
            primary_ok=False,
            fallback_ok=False,
            recover_after=2,
            probe_sequence=[True, True, False],
        )
    )

    # 第 1 次成功只是连续计数的开始（1/2）；第 2 次才回主路；第 3 次失败立刻重新降级
    assert [row["degraded"] for row in result["probe"]] == [True, False, True]


# ---------- 优雅退出 ----------


def test_shutdown_classifies_three_run_outcomes() -> None:
    """跑完 / 落盘可续跑 / 被撕碎：三种结局的计数与可续跑标记都要对。"""
    result = day20.shutdown_demo(Day20ShutdownRequest())
    outcomes = {row["run_id"]: row for row in result["runs"]}

    assert outcomes["r-1"]["outcome"] == "completed"
    assert outcomes["r-2"]["outcome"] == "checkpointed"
    assert outcomes["r-2"]["resumable"] is True
    assert outcomes["r-3"]["outcome"] == "shattered"
    assert outcomes["r-3"]["resumable"] is False
    assert result["counters"]["rejected_503"] == 6
    assert "撕碎" in result["verdict"]


def test_shutdown_without_flush_loses_trace_tail() -> None:
    """忘 flush：最后一段 trace 随进程消失，复盘时恰好缺最关键的几秒。"""
    result = day20.shutdown_demo(Day20ShutdownRequest(flush_trace=False))

    stages = [row["stage"] for row in result["timeline"]]
    assert "flush 缺失" in stages
    assert "trace" in result["verdict"]


def test_shutdown_short_prestop_leaks_traffic() -> None:
    """preStop 比规则传播短：摘除瞬间的最后一波请求照样打到正在退出的副本。"""
    result = day20.shutdown_demo(
        Day20ShutdownRequest(prestop_sleep_s=1, lb_propagation_s=5)
    )

    assert result["counters"]["leaked"] == 3
    assert result["config"]["relation_ok"] is False
    assert "preStop" in result["verdict"]


def test_shutdown_all_green_passes_and_grace_period_is_longer() -> None:
    """全部跑完 + flush：判定通过；编排器宽限期必须比 drain 窗口长。"""
    result = day20.shutdown_demo(
        Day20ShutdownRequest(
            grace_seconds=25,
            runs=[
                Day20ShutdownRun(run_id="r-1", remaining_s=3, checkpointed=False),
                Day20ShutdownRun(run_id="r-2", remaining_s=12, checkpointed=True),
            ],
        )
    )

    assert result["verdict"].startswith("可通过")
    assert result["config"]["termination_grace_seconds"] > 25
    assert "terminationGracePeriodSeconds: 40" in result["k8s_snippet"]
    assert "sleep" in result["k8s_snippet"]


# ---------- 容量 ----------


def test_capacity_takes_the_min_of_three_ceilings() -> None:
    """默认参数下上游（3000 RPM ÷ 6 次）才是瓶颈，Semaphore ≈ 31。"""
    result = day20.capacity_demo(Day20CapacityRequest())

    assert result["ceilings"]["binding"] == "upstream"
    assert abs(result["ceilings"]["throughput_rps"] - 8.33) < 0.02
    assert result["semaphore"]["per_replica"] == 31
    assert "掐脖子" in result["semaphore"]["verdict"]


def test_capacity_money_can_become_the_bottleneck() -> None:
    """预算砍到 3000：钱先成为天花板（0.13 run/s），Semaphore 反而超额配置。"""
    result = day20.capacity_demo(Day20CapacityRequest(monthly_budget=3000))

    assert result["ceilings"]["binding"] == "money"
    assert result["ceilings"]["throughput_rps"] < 0.2
    assert "倍" in result["semaphore"]["verdict"]


def test_load_test_reads_503_when_arrival_exceeds_capacity() -> None:
    """恒定速率超过闸门：排队超时直接 503，判定指向「真实容量上限」。"""
    result = day20.capacity_demo(
        Day20CapacityRequest(arrival_rps=20, max_wait_s=2, duration_s=30)
    )

    assert result["load_test"]["rejected_503"] > 0
    assert result["decision"]["verdict"] == "出现 503"


def test_load_test_reads_p95_blowup_without_503() -> None:
    """闸门太大：不拒绝但慢速排队，p95 超标——该调小闸门而不是加机器。"""
    result = day20.capacity_demo(
        Day20CapacityRequest(arrival_rps=14, max_wait_s=300, duration_s=60)
    )

    assert result["load_test"]["rejected_503"] == 0
    assert result["load_test"]["p95"] > 30
    assert result["decision"]["verdict"] == "p95 超标但无 503"


def test_load_test_reads_headroom_at_low_arrival() -> None:
    """远低于容量：p95 达标、无 503，可以继续 +20% 试探。"""
    result = day20.capacity_demo(
        Day20CapacityRequest(arrival_rps=2, max_wait_s=5, duration_s=30)
    )

    assert result["load_test"]["rejected_503"] == 0
    assert result["load_test"]["p95"] <= 30
    assert result["decision"]["verdict"] == "p95 达标且无 503"


def test_queue_sim_is_deterministic_and_monotonic() -> None:
    """同一组参数反复跑结果一致；到达率升高，排队只会更久（不会变好）。"""
    first = day20._queue_sim(4, 10, 40, 60, 5)
    second = day20._queue_sim(4, 10, 40, 60, 5)
    heavier = day20._queue_sim(6, 10, 40, 60, 5)

    assert first == second
    assert heavier["avg_wait"] >= first["avg_wait"]
    assert heavier["rejected_503"] >= first["rejected_503"]


# ---------- SLO 与告警 ----------


def _span(trace_id: str, ts: int, status: str = "ok", event: str = "done", degraded: bool = False) -> dict:
    """构造一条最小 span（与 Day 15 JSONL 字段对齐）。"""
    return {
        "trace_id": trace_id,
        "ts": ts,
        "status": status,
        "event": event,
        "degraded": degraded,
        "cost": 0.001,
        "latency_ms": 100,
    }


def test_load_runs_aggregates_spans_by_trace() -> None:
    """首尾 span 求时长；任一 span 失败则整个 run 失败；retry 单独计数。"""
    spans = [
        _span("a", 100, event="start"),
        _span("a", 105, event="retry", degraded=True),
        _span("a", 110, status="error"),
        _span("b", 200, event="start"),
        _span("b", 208),
        {"ts": 300},  # 缺 trace_id：跳过
    ]
    runs = {run["trace_id"]: run for run in day20.load_runs(spans)}

    assert runs["a"]["ok"] is False
    assert runs["a"]["latency"] == 10
    assert runs["a"]["retries"] == 1
    assert runs["a"]["degraded"] is True
    assert runs["b"]["ok"] is True
    assert len(runs) == 2


def test_load_runs_jsonl_skips_corrupt_lines() -> None:
    """半行日志（进程被杀时写了一半）必须跳过而不是让告警评估崩掉。"""
    lines = [
        json.dumps(_span("a", 100, event="start")),
        '{"trace_id": "b", "ts": 1',
        json.dumps(_span("a", 106)),
    ]
    runs = day20.load_runs_jsonl(lines)

    assert len(runs) == 1
    assert runs[0]["trace_id"] == "a"


def test_window_metrics_guards_small_samples() -> None:
    """样本不足直接不出指标：深夜 10 个 run 错 1 个不是 10% 事故。"""
    runs = [{"trace_id": str(i), "ts": 1000 + i, "ok": True, "latency": 5, "cost": 0.01, "retries": 0, "degraded": False} for i in range(10)]
    metrics = day20.window_metrics(runs, window_s=60, min_sample=50)

    assert metrics["insufficient_sample"] is True
    assert day20.burn_rate(metrics) is None


def test_burn_rate_is_bad_rate_over_allowed_bad_rate() -> None:
    """成功率 90% 对 99% 的 SLO：错误烧毁率 = 10 倍。"""
    metrics = {"insufficient_sample": False, "success_rate": 0.90, "degraded_rate": 0.20}
    burn = day20.burn_rate(metrics)

    assert burn == {"error_burn": 10.0, "degraded_burn": 2.0}


def test_should_page_uses_short_and_long_windows() -> None:
    """短窗口 14.4x → page；只有长窗口 6x → 工单；都低 → 正常。"""
    assert day20.should_page({"error_burn": 20.0})["severity"] == "page"
    assert day20.should_page({"error_burn": 3.0}, {"error_burn": 7.0})["severity"] == "ticket"
    assert day20.should_page({"error_burn": 1.2}, {"error_burn": 2.0})["severity"] == "ok"


def test_slo_demo_pages_on_real_incident() -> None:
    """事故占满短窗口：错误烧毁率 ≥ 14.4x，触发页面告警并附 runbook。"""
    result = day20.slo_demo(Day20SloRequest())

    assert result["alert"]["severity"] == "page"
    assert len(result["alert"]["runbook"]) == 3
    short = result["windows"][0]
    assert short["metrics"]["n"] >= 50
    assert short["burn"]["error_burn"] >= 14.4


def test_slo_demo_is_quiet_when_error_rate_is_low() -> None:
    """失败率降到 1%：不到 page / ticket 阈值，不叫人起床。"""
    result = day20.slo_demo(
        Day20SloRequest(bad_rate=0.01, degraded_rate=0.02, incident_min=30)
    )

    assert result["alert"]["severity"] == "ok"


def test_slo_demo_sample_lines_include_a_corrupt_line() -> None:
    """样例里故意留半行：前端能直观看到消费者必须容忍坏行。"""
    result = day20.slo_demo(Day20SloRequest(runs=100))

    assert len(result["sample_lines"]) == 3
    with_bad = day20.load_runs_jsonl(result["sample_lines"])
    assert len(with_bad) <= 2


# ---------- 混沌演练 ----------


def test_chaos_off_short_circuits_injection() -> None:
    """总开关打开：注入全路径短路，一次故障都不该发生。"""
    result = day20.chaos_demo(Day20ChaosRequest(chaos_off=True, fallback_configured=True))

    sim = result["simulation"]
    assert sim["rate_limited"] == 0
    assert sim["retries"] == 0
    assert sim["degraded_rate"] == 0
    assert sim["bare_500"] == 0
    assert result["expectations"][0]["pass"] is True


def test_chaos_default_reproduces_first_drill_blind_spots() -> None:
    """首次演练的真相：备模型 key 没配 + 同工具死磕，两处落空。"""
    result = day20.chaos_demo(Day20ChaosRequest())

    rows = {row["no"]: row for row in result["expectations"]}
    assert rows[1]["pass"] is True  # 退避重试消化了大部分 429
    assert rows[2]["pass"] is False  # L1 key 没配
    assert rows[3]["pass"] is False  # 连续死磕同一个死工具
    assert rows[4]["pass"] is True  # flush 做了
    assert len(result["blind_spots"]) == 2


def test_chaos_fixed_blind_spots_turn_green() -> None:
    """补上三个工程前置：预期全绿、盲区清空。"""
    result = day20.chaos_demo(
        Day20ChaosRequest(
            fallback_configured=True,
            tool_short_circuit=True,
            trace_flush=True,
        )
    )

    assert all(row["pass"] for row in result["expectations"])
    assert result["blind_spots"] == []
    assert result["simulation"]["fallback_used"] >= 0


def test_chaos_tool_short_circuit_halves_wasted_rounds() -> None:
    """同工具连续失败 2 次就短路：浪费轮数从 3 降到 2。"""
    with_short = day20.chaos_demo(Day20ChaosRequest(tool_short_circuit=True))
    without = day20.chaos_demo(Day20ChaosRequest(tool_short_circuit=False))

    assert with_short["simulation"]["same_tool_attempts"] == 2
    assert without["simulation"]["same_tool_attempts"] == 3


def test_chaos_never_returns_bare_500_because_of_ladder() -> None:
    """阶梯存在时「没拿到答案」也是个体面的 200：裸 500 恒为 0。"""
    for fallback in (True, False):
        result = day20.chaos_demo(
            Day20ChaosRequest(fallback_configured=fallback, llm_429_prob=0.95)
        )
        assert result["simulation"]["bare_500"] == 0
        assert result["simulation"]["final_status"] == "ok"
