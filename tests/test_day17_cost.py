"""Day 17 单测：成本与延迟优化（账本 / 路由 / 压缩 / 缓存 / 并行 / 预算熔断 / A-B 投影）。

运行：``python -m pytest tests/test_day17_cost.py -p no:cacheprovider -q``

四个原则：
1. 纯逻辑优先：账本、路由、压缩、缓存、并行耗时模型、预算三闸、A/B 投影全都不碰模型；
2. 真实模型的接口（/day17/ledger 的 live、/day17/router 的 use_llm、/day17/budget-agent 的 live）
   在这里不跑——本文件只验证「不花钱也能跑的部分」，避免单测烧 token；
3. 落盘用例统一把 ``LEDGER_FILE`` / ``AB_FILE`` 指到 ``tmp_path``，绝不污染 var/day17；
4. 每个杠杆都要有「反面对照」断言：按条裁剪会产生孤儿 tool 消息、写操作不会被缓存、
   45 与 450 相似度 1.0 必须被实体校验拦下——这些是今天的教学核心，不能只测 happy path。
"""

import asyncio
import json

import pytest

from app.modules.study.schemas.study import (
    Day17AbRequest,
    Day17BudgetRequest,
    Day17CacheRequest,
    Day17CompressRequest,
    Day17LedgerRequest,
    Day17ParallelRequest,
    Day17RouterRequest,
)
from app.modules.study.services import day17_cost as day17


# ---------- 概念表与价格 ----------


def test_mapping_covers_course_sections() -> None:
    """概念表要覆盖 1.1 / 1.3 / 1.4 / 2.4 / 第 4 节 / 第 6 节。"""
    data = day17.mapping()

    assert len(data["cost_structure"]) == 3
    assert len(data["latency_breakdown"]) == 4
    assert len(data["levers"]) == 4
    assert len(data["cache_layers"]) == 3
    # 课程标题写「五个坑」，正文实际列了 6 条；本模块全部保留
    assert len(data["pitfalls"]) == 6
    assert len(data["cheatsheet"]) == 10
    assert len(data["checklist"]) == 10
    assert data["cost_structure"][1]["scope"] == "一次 run"
    assert "O(N²)" in data["cost_structure"][1]["note"]
    assert data["levers"][0]["priority"] == "1"
    assert data["thresholds"]["max_cost"] == 0.02
    assert data["thresholds"]["semantic_cache_threshold"] == 0.95


def test_cheatsheet_and_pitfalls_are_complete() -> None:
    """速查表与坑表不能有空格子（前端表格直接渲染）。"""
    for row in day17.CHEATSHEET:
        assert row["topic"] and row["content"] and row["key"]
    for row in day17.PITFALLS:
        assert row["no"] and row["title"] and row["symptom"] and row["fix"]


def test_price_of_matches_and_falls_back() -> None:
    """价格按名字模糊匹配；未知模型给兜底价，绝不 KeyError。"""
    assert day17.price_of("gpt-4o-mini") == (0.15, 0.60)
    assert day17.price_of("gpt-4o") == (2.50, 10.00)
    assert day17.price_of("deepseek-v4-flash") == day17.PRICE["deepseek-chat"]
    assert day17.price_of("deepseek-reasoner") == day17.PRICE["deepseek-reasoner"]
    assert day17.price_of("some-local-7b") == day17.PRICE["unknown"]

    # 便宜档单价必须真的更便宜，否则「分级省钱」这个结论不成立
    assert day17.step_cost("gpt-4o-mini", 1000, 100) < day17.step_cost("gpt-4o", 1000, 100)


# ---------- 账本 ----------


def test_ntok_is_monotonic() -> None:
    """token 估算：空串 0、长文本更多、中文不按字符除以 4。"""
    assert day17.ntok("") == 0
    short = day17.ntok("你好")
    long = day17.ntok("你好" * 50)
    assert short >= 1
    assert long > short
    # 中文 10 个字至少 5 个 token（退化实现是 1 字 1 token）
    assert day17.ntok("一二三四五六七八九十") >= 5


def test_simulate_run_grows_quadratically() -> None:
    """输入 token 每步递增（历史重放），放大率必须远超 1。"""
    ledger = day17.simulate_run(steps=5, tool_tokens=800)
    inputs = [step.in_tok for step in ledger.steps]

    assert len(ledger.steps) == 6  # 5 步决策 + 1 步汇总
    assert inputs == sorted(inputs)
    assert inputs[-1] > inputs[0] * 3

    summary = ledger.summary(baseline_in=900)
    assert summary["amplification"] > 5
    assert 0 < summary["waste_ratio"] < 1
    # 费用是累加的，且每一步都为正
    assert summary["cost"] > 0
    assert all(step.cost > 0 for step in ledger.steps)


def test_ledger_amplification_edge_cases() -> None:
    """基线为 0 时不要把放大率算成 ZeroDivisionError。"""
    ledger = day17.Ledger()
    ledger.add("gpt-4o-mini", 100, 20, 0.1)
    assert ledger.amplification(0) == 0.0
    assert ledger.waste_ratio(0) == 0.0
    assert ledger.summary(0)["steps"] == 1


def test_ledger_demo_variants_are_monotonic() -> None:
    """三个变体：剪裁工具结果、再瘦身 prompt，成本必须一档比一档低。"""
    result = day17.ledger_demo(Day17LedgerRequest(steps=5, tool_tokens=800), "deepseek-chat")
    costs = [variant["cost"] for variant in result["variants"]]

    assert len(result["variants"]) == 3
    assert costs == sorted(costs, reverse=True)
    assert costs[2] < costs[0] * 0.6
    assert result["totals"]["in_tok"] == result["variants"][0]["in_tok"]


def test_run_ledger_writes_report(tmp_path, monkeypatch) -> None:
    """账本落盘要指到临时目录：不允许污染 var/day17 的真实账本。"""
    target = tmp_path / "last_ledger.json"
    monkeypatch.setattr(day17, "LEDGER_FILE", target)

    result = asyncio.run(
        day17.run_ledger(None, Day17LedgerRequest(steps=3, tool_tokens=400, live=False))
    )

    assert result["live_probe"] is None
    assert json.loads(target.read_text(encoding="utf-8"))["model"] == result["model"]


# ---------- 杠杆一：分级路由 ----------


def test_route_by_rules_course_cases() -> None:
    """课程里那三个例子：简单 → small，对比 → main，长问题 → main。"""
    assert day17.route_by_rules("营业时间几点") is day17.Tier.SMALL
    assert day17.route_by_rules("帮我对比一下这两个套餐的性价比") is day17.Tier.MAIN
    long_question = (
        "我上个月买的耳机坏了，订单号 SO20261001，但我不确定是不是在保修期内，"
        "而且我想知道能不能直接换新而不是维修"
    )
    # 课程这条例子只有 58 字（< 60），按「纯长度」判据会被丢给便宜档；
    # 本模块补了「多约束」判据（分句 / 转折词 ≥2），所以仍然判 main。
    assert len(long_question) < 60
    assert day17.route_by_rules(long_question) is day17.Tier.MAIN
    assert "多约束" in day17.explain_route(long_question)["reason"]
    # 轮次够多也要升级（上下文变长，小模型容易断片）
    assert day17.route_by_rules("你好", history_rounds=5) is day17.Tier.MAIN


def test_explain_route_has_reason() -> None:
    """路由结论必须能解释——课堂上最容易问「为什么判成 main」。"""
    data = day17.explain_route("帮我分析一下这个方案")
    assert data["tier"] == "main"
    assert "困难词" in data["reason"]

    simple = day17.explain_route("营业时间几点")
    assert simple["tier"] == "small"
    assert "简单词" in simple["reason"] or "短问题" in simple["reason"]


def test_escalator_signals_and_cap() -> None:
    """三信号升级 + BIG 封顶 + 成功清零。"""
    escalator = day17.Escalator()
    assert escalator.next_tier(day17.Tier.SMALL, schema_error=True) is day17.Tier.MAIN
    assert escalator.next_tier(day17.Tier.SMALL, empty_answer=True) is day17.Tier.MAIN
    assert escalator.next_tier(day17.Tier.SMALL) is day17.Tier.SMALL

    escalator.note_tool_error()
    assert escalator.next_tier(day17.Tier.SMALL) is day17.Tier.SMALL
    escalator.note_tool_error()
    assert escalator.next_tier(day17.Tier.SMALL) is day17.Tier.MAIN

    # 封顶：BIG 再失败也不升（该走降级交付 + 转人工）
    assert escalator.next_tier(day17.Tier.BIG, schema_error=True) is day17.Tier.BIG

    chain = day17.simulate_escalation(["tool_error", "tool_error", "ok", "schema_error"])
    tiers = [hop["tier"] for hop in chain]
    assert tiers[0] == "small"
    assert tiers[2] == "main"          # 连续两次工具报错 → 升级
    assert tiers[3] == "main"          # ok 只清零计数，不降级
    assert tiers[4] == "big"           # 结构化失败 → 再升一档
    assert "升级" in chain[2]["trigger"]


def test_mix_saving_is_positive() -> None:
    """分流必须比全走主力便宜，且细节行能对上总账。"""
    data = day17.mix_saving({"small": 4, "main": 5, "big": 1})

    assert data["total_calls"] == 10
    assert data["saving_ratio"] > 0.2
    assert data["mixed_cost"] < data["all_main_cost"]
    assert abs(sum(row["subtotal"] for row in data["detail"]) - data["mixed_cost"]) < 1e-6
    assert 0 < data["per_call_saving"] < 1


# ---------- 杠杆二：上下文压缩 ----------


def _dirty_json(chars: int = 4000) -> str:
    """一段「JSON + 整页 HTML」的脏工具返回。"""
    return json.dumps(
        {
            "order_id": "SO20261115001",
            "status": "已发货",
            "amount": "45.00",
            "eta": "2026-11-18",
            "desc": "<div>物流节点</div>" * (chars // 20),
        },
        ensure_ascii=False,
    )


def test_trim_tool_output_keeps_key_fields() -> None:
    """JSON 工具返回只留关键字段，节省率要高。"""
    result = day17.trim_tool_output(_dirty_json(4000), max_chars=400)

    assert result["after_tokens"] < result["before_tokens"]
    assert result["saved_ratio"] > 0.8
    assert "已发货" in result["text"]
    assert "45.00" in result["text"]
    assert any("status=" in item for item in result["kept_fields"])


def test_trim_tool_output_falls_back_to_truncation() -> None:
    """非 JSON 也要能剪：保留头部 + 打上截断标记。"""
    raw = "这是一整页 HTML 文本。" * 400
    result = day17.trim_tool_output(raw, max_chars=200)

    assert len(result["text"]) < len(raw)
    assert "已截断" in result["text"]
    assert result["kept_fields"] == []
    assert result["saved_tokens"] > 0


def test_trim_short_output_is_noop() -> None:
    """短返回不剪（剪了反而丢信息）。"""
    raw = '{"status": "已发货"}'
    result = day17.trim_tool_output(raw, max_chars=400)
    assert result["after_tokens"] <= result["before_tokens"]


def test_split_rounds_and_pairing() -> None:
    """按轮切分 + 配对检查：健康历史无问题，孤儿 tool 消息必须被抓出来。"""
    history = day17.synthetic_history(rounds=3, tool_chars=100)
    assert day17.check_tool_pairing(history) == []

    # 3 轮追问 + 开头那句原始诉求 = 4 轮（开头那条 user 也是一轮，不能被吞）
    rounds = day17.split_rounds([m for m in history if m["role"] != "system"])
    assert len(rounds) == 4
    assert all(group[0]["role"] == "user" for group in rounds)

    # 反面教材：按条数从尾部切，一定切出孤儿 tool 消息
    naive = day17.naive_slice(history, keep=3)
    assert day17.check_tool_pairing(naive) != []

    dangling = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "x"}]}]
    assert day17.check_tool_pairing(dangling) != []


def test_compress_messages_under_budget_is_noop() -> None:
    """没到触发线就一条不动（别为了「优化」而优化）。"""
    history = day17.synthetic_history(rounds=2, tool_chars=100)
    result = day17.compress_messages(history, budget_tokens=100000)

    assert result["compressed"] is False
    assert result["before_tokens"] == result["after_tokens"]
    assert result["saved_ratio"] == 0.0
    assert "无需压缩" in result["reason"]


def test_compress_messages_keeps_anchor_and_pairing() -> None:
    """超预算时按轮压缩：省 token、保留 system 锚点、配对仍然健康。"""
    history = day17.synthetic_history(rounds=8, tool_chars=400)
    result = day17.compress_messages(history, budget_tokens=1200, keep_recent=2)

    assert result["compressed"] is True
    assert result["after_tokens"] < result["before_tokens"]
    assert result["saved_ratio"] > 0.3
    assert result["pairing_issues"] == []
    assert result["rounds_after"] < result["rounds_before"]

    contents = [m.get("content") or "" for m in result["messages"]]
    assert any("约束锚点" in text for text in contents)   # 安全边界永驻
    assert any("[前情摘要]" in text for text in contents)


def test_compress_with_external_summary() -> None:
    """外部传入摘要（= 小模型摘要）时直接采用，不再走抽取式。"""
    history = day17.synthetic_history(rounds=6, tool_chars=200)
    result = day17.compress_messages(
        history, budget_tokens=800, keep_recent=1, summary="用户问了 6 次订单物流，均已答复"
    )
    assert result["summary"] == "用户问了 6 次订单物流，均已答复"
    assert any("用户问了 6 次订单物流" in (m.get("content") or "") for m in result["messages"])


def test_extractive_summary_is_deterministic() -> None:
    """抽取式摘要：只取首句、去重、长度受限。"""
    messages = [
        {"role": "user", "content": "第一句在这里。第二句要丢掉。"},
        {"role": "assistant", "content": "第一句在这里。重复的要去掉。"},
        {"role": "user", "content": "另一条消息的首句。"},
    ]
    summary = day17.extractive_summary(messages, max_chars=40)
    assert "第二句" not in summary
    assert summary.count("第一句在这里") == 1
    assert len(summary) <= 41


def test_compress_demo_structure() -> None:
    """压缩接口的三个区块都要有，且反面教材必须暴露配对问题。"""
    result = asyncio.run(
        day17.compress_demo(
            None,
            Day17CompressRequest(history_rounds=10, budget=2000, keep_recent=2, use_model=False),
        )
    )

    assert result["prompt"]["lean_tokens"] < result["prompt"]["fat_tokens"]
    assert result["prompt"]["saving_ratio"] > 0.3
    assert result["tool_trim"]["saved_ratio"] > 0.5
    assert result["context"]["compressed"] is True
    assert result["context"]["pairing_ok"] is True
    assert result["context"]["naive_pairing_issues"]
    assert result["context"]["summary_source"] == "extractive"


# ---------- 杠杆三：缓存三层 ----------


def test_tool_cache_readonly_whitelist() -> None:
    """安全阀 1：写操作永不缓存；只读工具命中；TTL 过期即失效。"""
    cache = day17.ToolCache(ttl=300)
    args = {"order_id": "SO20261115001"}

    assert cache.lookup("get_order", args)["hit"] is False
    assert cache.save("get_order", args, "ok") is True
    assert cache.lookup("get_order", args)["hit"] is True

    # 参数顺序不同也应命中（键对参数排序）
    assert cache.lookup("get_order", {"order_id": "SO20261115001"})["hit"] is True

    assert cache.save("create_refund", {"order_id": "x", "amount": 45}, "ok") is False
    blocked = cache.lookup("create_refund", {"order_id": "x", "amount": 45})
    assert blocked["hit"] is False
    assert "写操作" in blocked["reason"]

    expired = day17.ToolCache(ttl=0)
    expired.save("get_order", args, "ok")
    assert expired.lookup("get_order", args)["hit"] is False

    stats = cache.stats()
    assert stats["hits"] >= 2
    assert stats["blocked"] >= 2
    assert 0 <= stats["hit_rate"] <= 1


def test_entities_extract_numbers_and_ids() -> None:
    """实体抽取：数字与单号都要抓到，否则语义缓存没有可比的安全阀。"""
    assert day17.entities("退款 45 元多久到账") == {"45"}
    assert day17.entities("退款 450.5 元多久到账") == {"450.5"}
    assert day17.entities("订单 SO20261115001 到哪了") >= {"SO20261115001"}


def test_semantic_cache_entity_valve() -> None:
    """本日最重要的一条：45 与 450 相似度 1.0，必须被实体校验拦下。"""
    cache = day17.SemanticCache(threshold=0.95)
    cache.save("退款 45 元什么时候到账", "45 元 3 个工作日原路退回", user_id="u1")

    hit = cache.lookup("退款 45 元多久到账", "u1")
    assert hit["hit"] is True
    assert hit["score"] >= 0.95

    blocked = cache.lookup("退款 450 元多久到账", "u1")
    assert blocked["hit"] is False
    assert blocked["score"] >= 0.95          # 相似度够高——所以要靠实体拦住
    assert "实体不同" in blocked["reason"]
    assert cache.stats()["blocked_entity"] == 1

    cross_user = cache.lookup("退款 45 元多久到账", "u2")
    assert cross_user["hit"] is False
    assert "跨用户" in cross_user["reason"]
    assert cache.stats()["blocked_user"] == 1


def test_semantic_cache_threshold_blocks_low_similarity() -> None:
    """阈值真的生效：完全不同的问题不命中。"""
    cache = day17.SemanticCache(threshold=0.99)
    cache.save("退款 45 元什么时候到账", "答案", user_id="u1")
    result = cache.lookup("发票怎么开", "u1")
    assert result["hit"] is False
    assert result["score"] < 0.99


def test_prefix_stability_demo() -> None:
    """带秒时间戳打穿前缀缓存；稳定前缀 + 变动后缀才保得住命中率。"""
    data = day17.prefix_stability(5)

    assert data["bad"]["unique"] == 5
    assert data["bad"]["hit_rate"] == 0.0
    assert data["good"]["unique"] == 1
    assert data["good"]["hit_rate"] == 0.8

    request = day17.build_request("稳定前缀", "用户问题（今天是 2026-11-15）")
    assert request["prefix_hash"] == day17.prefix_fingerprint("稳定前缀")
    assert request["messages"][0]["role"] == "system"


def test_cache_demo_matrix() -> None:
    """三层缓存的现场：L1 先 miss 再 hit、写操作被拦；L2 三条探针各司其职。"""
    result = day17.cache_demo(Day17CacheRequest())

    l1_hits = [row["hit"] for row in result["l1"]["scenarios"]]
    assert l1_hits[0] is False        # 第一次查订单
    assert l1_hits[1] is True         # 第二次命中
    assert l1_hits[2] is True         # 参数顺序不同也命中
    assert l1_hits[3] is False        # 写操作
    assert l1_hits[4] is False        # 写操作再查仍然不缓存

    matrix = result["l2"]["matrix"]
    assert len(matrix) == 3
    assert matrix[0]["hit"] is True
    assert matrix[1]["hit"] is False and "实体不同" in matrix[1]["reason"]
    assert matrix[2]["hit"] is False and "跨用户" in matrix[2]["reason"]
    assert result["l3"]["prefix_stability"]["bad"]["hit_rate"] == 0.0


# ---------- 杠杆四：并行 ----------


def test_parallel_duration_pure_model() -> None:
    """并行耗时 = 按上限分批取 max；加速比随并发上限提升但不超过理论值。"""
    durations = [1000, 1000, 1000, 1000]

    assert day17.serial_duration(durations) == 4000
    assert day17.parallel_duration(durations, limit=4) == 1000
    assert day17.parallel_duration(durations, limit=2) == 2000
    assert day17.parallel_duration(durations, limit=1) == 4000
    assert day17.parallel_duration([], limit=4) == 0

    assert day17.speedup(durations, limit=4) == 4.0
    assert day17.speedup(durations, limit=2) == 2.0
    assert day17.speedup([], limit=4) == 0.0

    # 长短不齐时，并行收益更明显（小任务填进大任务的空隙）
    uneven = [1200, 300, 300, 300]
    assert day17.parallel_duration(uneven, limit=4) == 1200
    assert day17.serial_duration(uneven) == 2100


def test_parallel_demo_reports_failures_without_crashing() -> None:
    """单点失败不炸整批：一个工具报错，其余照常返回。"""
    payload = Day17ParallelRequest(
        tools=[
            {"name": "a", "ms": 30},
            {"name": "b", "ms": 40},
            {"name": "c", "ms": 50},
        ],
        limit=3,
        timeout_ms=2000,
        fail_index=1,
    )
    result = asyncio.run(day17.parallel_demo(payload))

    assert result["serial_ms"] == 120
    assert len(result["results"]) == 3
    failed = [row for row in result["results"] if not row["ok"]]
    assert len(failed) == 1
    assert failed[0]["name"] == "b"
    assert result["speedup"] >= 1.0
    assert len(result["timeline"]) == 3


def test_parallel_demo_timeout_is_caught() -> None:
    """超时也只是一条失败记录，不是异常穿透。"""
    payload = Day17ParallelRequest(tools=[{"name": "slow", "ms": 800}], limit=1, timeout_ms=200)
    result = asyncio.run(day17.parallel_demo(payload))

    assert result["results"][0]["ok"] is False
    assert "超时" in result["results"][0]["error"]


# ---------- 预算熔断 ----------


def test_budget_three_gates() -> None:
    """三闸：钱 / 步数 / 墙钟，任一超限都要报出可读原因。"""
    budget = day17.Budget(max_cost=0.01, max_steps=2, max_wall=100)
    assert budget.check() is None

    budget.cost = 0.02
    assert "cost" in (budget.check() or "")

    budget.cost = 0.0
    budget.steps = 2
    assert "steps" in (budget.check() or "")

    budget.steps = 0
    budget.t0 -= 200
    assert "wall" in (budget.check() or "")
    assert budget.snapshot()["max_cost"] == 0.01


def test_simulate_budget_agent_fuses_and_degrades() -> None:
    """脚本化模型永不收敛 → 必然在步数闸熔断 → 摘工具降级交付（不报错）。"""
    budget = day17.Budget(max_cost=1.0, max_steps=3, max_wall=100)
    result = day17.simulate_budget_agent(question="我的订单到哪了", budget=budget)

    assert result["degraded"] is True
    assert "steps 3" in result["stop_reason"]
    # 3 步决策 + 1 步降级
    assert len(result["steps"]) == 4
    assert result["steps"][-1]["node"] == "degrade"
    assert result["steps"][-1]["stopped"] is True
    assert result["answer"]
    assert result["ledger"]["steps"] == 4
    assert result["budget"]["steps"] == 4


def test_simulate_budget_agent_cost_gate() -> None:
    """钱闸也能单独触发：把 max_cost 压低，第一步之后就该熔断。"""
    budget = day17.Budget(max_cost=1e-6, max_steps=9, max_wall=100)
    result = day17.simulate_budget_agent(question="查订单", budget=budget)

    assert "cost" in result["stop_reason"]
    assert result["degraded"] is True


def test_budget_demo_simulate_needs_no_model(tmp_path, monkeypatch) -> None:
    """simulate 模式不碰数据库与模型：不传 db 也能跑。"""
    result = asyncio.run(
        day17.budget_demo(None, Day17BudgetRequest(mode="simulate", max_steps=2, max_cost=1.0))
    )

    assert result["mode"] == "simulate"
    assert result["degraded"] is True
    assert result["cache"]["hit_rate"] == 0.0


# ---------- A/B 投影 ----------


def test_project_levers_is_monotonic() -> None:
    """四个杠杆逐档叠加：成本与延迟都必须单调下降，正确率列必须留空。"""
    report = day17.project_levers(Day17AbRequest(steps=5, tool_tokens=800), "deepseek-chat")
    rows = report["rows"]

    assert len(rows) == 5
    costs = [row["cost"] for row in rows]
    latencies = [row["latency"] for row in rows]

    assert costs == sorted(costs, reverse=True)
    assert latencies == sorted(latencies, reverse=True)
    assert all(row["accuracy"] is None for row in rows)
    assert rows[0]["delta_cost_ratio"] == 0.0

    # 压缩是收益最大的一刀；并行只改延迟不改成本
    assert rows[1]["delta_cost_ratio"] > 0.4
    assert rows[4]["cost"] == rows[3]["cost"]
    assert rows[4]["latency"] < rows[3]["latency"]


def test_project_levers_conclusion_mentions_day16() -> None:
    """结论必须明确「正确率得由 Day 16 实测」，不能把课程数字抄成自己的结论。"""
    report = day17.project_levers(Day17AbRequest(), "deepseek-chat")
    assert "Day 16" in report["conclusion"]
    assert report["gate"]["accuracy_drop_max"] == 0.02
    assert "day16" in report["gate"]["day16_hint"]


def test_ab_demo_writes_report(tmp_path, monkeypatch) -> None:
    """A/B 报告落盘要到临时目录。"""
    target = tmp_path / "ab_report.json"
    monkeypatch.setattr(day17, "AB_FILE", target)

    result = asyncio.run(day17.ab_demo(None, Day17AbRequest(model="deepseek-chat")))

    assert result["files"]["ab_report"] == str(target)
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8"))["rows"]


# ---------- 演示工具与路由接口 ----------


def test_call_tool_returns_structured_errors() -> None:
    """「错误即结果」：查不到订单返回 JSON 错误，而不是抛异常。"""
    ok = json.loads(day17.call_tool("get_order", {"order_id": "SO20261115001"}))
    assert ok["ok"] is True
    assert ok["amount"] == "45.00"

    missing = json.loads(day17.call_tool("get_order", {"order_id": "SO00000000000"}))
    assert missing["ok"] is False
    assert missing["error"] == "order_not_found"

    unknown = json.loads(day17.call_tool("nope", {}))
    assert unknown["error"] == "unknown_tool"


def test_router_demo_without_llm() -> None:
    """不勾 use_llm 时只跑规则：不碰模型、不花钱。"""
    result = asyncio.run(day17.router_demo(None, Day17RouterRequest(question="营业时间几点")))

    assert result["rules"]["tier"] == "small"
    assert result["llm"] is None
    assert len(result["tiers"]) == 3
    assert result["mix"]["saving_ratio"] > 0
    assert result["escalation"][0]["tier"] == "small"


def test_tool_schemas_are_sellable() -> None:
    """工具说明书：五个工具、都有 description 与 required 参数（Day 03 三要素）。"""
    names = {schema["function"]["name"] for schema in day17.TOOL_SCHEMAS}
    assert names == {"get_order", "get_shipping", "get_coupon", "get_invoice", "create_refund"}

    for schema in day17.TOOL_SCHEMAS:
        function = schema["function"]
        assert function["description"]
        assert function["parameters"]["required"]
        # 白名单与工具集必须自洽：可缓存的工具都得真实存在
        assert {name for name in day17.READONLY_TOOLS} <= names


@pytest.mark.parametrize("ratio", [0.0, 0.28, 0.5])
def test_cache_hit_rate_affects_step_count(ratio: float) -> None:
    """缓存命中率越高，实际执行步数越少（A/B 表里「少跑几步」的来源）。"""
    report = day17.project_levers(
        Day17AbRequest(steps=5, tool_tokens=400, cache_hit_rate=ratio), "deepseek-chat"
    )
    cache_row = report["rows"][3]
    assert cache_row["steps"] <= 5
