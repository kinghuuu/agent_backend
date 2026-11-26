"""Day 15 单测：可观测性（自建 span / 树还原 / 指标聚合 / 回调收集 / 脱敏 / 跨进程合并）。

运行：``python -m pytest tests/test_day15_observability.py -p no:cacheprovider -q``

四个原则：
1. 纯逻辑优先：span 嵌套、树还原、分位数、指标聚合、脱敏、时间轴合并全都不碰模型；
2. 真实模型的接口（callback / own-trace / full-stack）只验证结构与契约，不在单测里烧 token；
3. 并发隔离用真 asyncio 跑，因为「contextvars vs threading.local」的差异只有并发才能证明；
4. 落盘用例统一指向 ``tmp_path``，绝不污染 var/day15 下的真实 trace。
"""

import asyncio
import importlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.modules.study.schemas.study import Day15FeedbackRequest
from app.modules.study.services import day15_observability as day15


# ---------- 纯逻辑：概念表 ----------


def test_mapping_covers_course_sections() -> None:
    """概念表要覆盖课程 1.x + 第 4 节 + 第 5 节：五个坑、两条路线、十行速查表。"""
    data = asyncio.run(day15.mapping())

    assert len(data["failure_modes"]) == 5
    assert len(data["pillars"]) == 3
    assert len(data["routes"]) == 3
    assert len(data["pitfalls"]) == 5
    assert len(data["cheatsheet"]) == 10
    assert len(data["trace_fields"]) == 8
    assert data["failure_modes"][0]["agent"].startswith("「软失败」")
    assert "langsmith" in data["platform"]


def test_trace_fields_are_the_eight_key_model() -> None:
    """八字段数据模型是今天的地基：顺序与字段名都要稳定。"""
    fields = [row["field"] for row in day15.TRACE_FIELDS]
    assert fields == [
        "trace_id",
        "span_id",
        "parent_id",
        "name",
        "start_ts",
        "duration_ms",
        "attributes",
        "status",
    ]


# ---------- 自建 span：嵌套、落盘、错误、树 ----------


def test_span_nesting_and_jsonl(tmp_path: Path) -> None:
    """嵌套 span 的 parent_id 逐层指向父级，写进 JSONL 的每行都是八字段。"""
    path = tmp_path / "traces.jsonl"
    tracer = day15.SpanTracer(path=path)

    with tracer.span("agent-run", kind="agent", question="报销 45 元") as root:
        with tracer.span("chat", kind="llm", model="gpt-4o-mini") as llm:
            llm["attributes"]["input_tokens"] = 12
            llm["attributes"]["output_tokens"] = 3
        with tracer.span("tool:search", kind="tool", args={"keyword": "餐饮"}) as tool:
            tool["attributes"]["output"] = "上限 100 元"

    records = day15.load_records(path)
    expected_fields = {row["field"] for row in day15.TRACE_FIELDS}
    assert len(records) == 3
    assert all(set(record) >= expected_fields for record in records)

    by_name = {record["name"]: record for record in records}
    assert by_name["agent-run"]["parent_id"] is None
    assert by_name["chat"]["parent_id"] == by_name["agent-run"]["span_id"]
    assert by_name["tool:search"]["parent_id"] == by_name["agent-run"]["span_id"]

    tree = tracer.tree()
    assert tree[0]["name"] == "agent-run"
    assert len(tree[0]["children"]) == 2
    chat = next(child for child in tree[0]["children"] if child["name"] == "chat")
    assert chat["attributes"]["input_tokens"] == 12
    assert chat["duration_ms"] >= 0


def test_span_error_status_and_manual_override(tmp_path: Path) -> None:
    """异常自动标 error；工具报错被吞掉时，调用方也能手动把 status 标成 error。"""
    path = tmp_path / "traces.jsonl"
    tracer = day15.SpanTracer(path=path)

    with pytest.raises(RuntimeError):
        with tracer.span("boom", kind="tool"):
            raise RuntimeError("工具炸了")

    with tracer.span("soft-fail", kind="tool") as record:
        record["status"] = "error"
        record["attributes"]["error"] = "业务失败但不抛异常"

    rows = {record["name"]: record for record in day15.load_records(path)}
    assert rows["boom"]["status"] == "error"
    assert "RuntimeError" in rows["boom"]["error"]
    assert rows["soft-fail"]["status"] == "error"
    assert rows["soft-fail"]["attributes"]["error"] == "业务失败但不抛异常"


def test_tree_orders_children_by_start_time(tmp_path: Path) -> None:
    """树的子节点按开始时间排序，回放时才和真实执行顺序一致。"""
    tracer = day15.SpanTracer(persist=False)
    with tracer.span("root", kind="agent"):
        for index in range(3):
            with tracer.span(f"step-{index}", kind="step"):
                time.sleep(0.001)

    tree = tracer.tree()
    assert [child["name"] for child in tree[0]["children"]] == ["step-0", "step-1", "step-2"]


def test_flatten_tree_adds_depth() -> None:
    """前端用压平后的 depth 画缩进，比递归组件省事。"""
    tracer = day15.SpanTracer(persist=False)
    with tracer.span("root", kind="agent"):
        with tracer.span("chat", kind="llm"):
            pass

    rows = day15.flatten_tree(tracer.tree())
    assert [(row["name"], row["depth"]) for row in rows] == [("root", 0), ("chat", 1)]


# ---------- 脱敏（坑 2） ----------


def test_mask_pii_patterns() -> None:
    """手机号 / 身份证 / 邮箱都要打码，普通文本原样保留。"""
    text = "联系 13812345678 / 110101199001011234 / zhang@example.com，金额 45 元"
    masked = day15.mask_pii(text)

    assert "13812345678" not in masked
    assert "138****5678" in masked
    assert "110101199001011234" not in masked
    assert "zhang@example.com" not in masked
    assert "45 元" in masked


def test_span_scrubs_sensitive_keys_before_write(tmp_path: Path) -> None:
    """api_key 整段打码，嵌套结构里的手机号也要处理——脱敏发生在写盘前。"""
    path = tmp_path / "traces.jsonl"
    tracer = day15.SpanTracer(path=path)

    with tracer.span(
        "agent-run",
        kind="agent",
        api_key="sk-super-secret",
        profile={"phone": "13812345678"},
        messages=[{"role": "user", "content": "我的手机 13900001111"}],
    ):
        pass

    record = day15.load_records(path)[0]
    assert record["attributes"]["api_key"] == "***"
    assert record["attributes"]["profile"]["phone"] == "138****5678"
    assert "13900001111" not in json.dumps(record, ensure_ascii=False)


# ---------- 指标聚合（三大件的第三件） ----------


def _record(
    trace_id: str,
    span_id: str,
    *,
    parent_id: str | None = None,
    kind: str = "step",
    status: str = "ok",
    duration_ms: float = 10.0,
    started: float = 0.0,
    **attributes: object,
) -> dict[str, object]:
    """手工构造一条八字段记录（指标聚合的输入）。"""
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_id": parent_id,
        "name": span_id,
        "kind": kind,
        "start_ts": started,
        "duration_ms": duration_ms,
        "attributes": dict(attributes),
        "status": status,
    }


def test_percentile_uses_nearest_rank() -> None:
    """p95 用最近秩：只有两条数据时取最大值，不会被平均值抹平。"""
    assert day15.percentile([], 0.95) == 0.0
    assert day15.percentile([1000, 3000], 0.5) == 1000
    assert day15.percentile([1000, 3000], 0.95) == 3000
    assert day15.percentile([1, 2, 3, 4, 5], 0.5) == 3


def test_summarize_metrics_from_records() -> None:
    """端到端 p50/p95、token 与成本、工具错误率、终止原因分布全部从 trace 算出来。"""
    records = [
        _record("t1", "r1", kind="agent", duration_ms=1000, termination="normal", steps=2),
        _record("t1", "l1", parent_id="r1", kind="llm", input_tokens=100, output_tokens=20),
        _record("t1", "tool1", parent_id="r1", kind="tool", status="error"),
        _record("t2", "r2", kind="agent", duration_ms=3000, termination="truncated"),
        _record("t2", "l2", parent_id="r2", kind="llm", input_tokens=200, output_tokens=30),
        _record("t2", "tool2", parent_id="r2", kind="tool"),
    ]

    metrics = day15.summarize(records)

    assert metrics["traces"] == 2
    assert metrics["llm_calls"] == 2
    assert metrics["tool_calls"] == 2
    assert metrics["tool_errors"] == 1
    assert metrics["tool_error_rate"] == 0.5
    assert metrics["latency_p50_ms"] == 1000
    assert metrics["latency_p95_ms"] == 3000
    assert metrics["input_tokens"] == 300
    assert metrics["output_tokens"] == 50
    assert metrics["total_tokens"] == 350
    assert metrics["cost_cny"] == pytest.approx(0.075)
    assert metrics["termination"] == {"normal": 1, "truncated": 1}


def test_summarize_empty_is_safe() -> None:
    """没有任何 trace 时不能除零，指标看板照样能打开。"""
    metrics = day15.summarize([])
    assert metrics["traces"] == 0
    assert metrics["error_rate"] == 0.0
    assert metrics["tool_error_rate"] == 0.0
    assert metrics["cost_cny"] == 0.0


# ---------- 终止原因判定 ----------


def _message(kind: str, tool_calls: list | None = None) -> SimpleNamespace:
    """构造只带 type / tool_calls 的假消息（不依赖 langchain）。"""
    return SimpleNamespace(type=kind, tool_calls=tool_calls or [])


def test_classify_termination() -> None:
    """normal / truncated / pending / empty 四态：软失败排查的第一道分诊。"""
    assert day15.classify_termination([]) == "empty"
    assert day15.classify_termination([_message("human"), _message("ai")]) == "normal"
    truncated = [_message("ai", [{"name": "x"}]), _message("tool"), _message("ai", [{"name": "y"}])]
    assert day15.classify_termination(truncated, max_steps=2) == "truncated"
    assert day15.classify_termination(truncated, max_steps=5) == "pending"


# ---------- 并发不串线（坑 4） ----------


def test_concurrency_isolated_by_contextvars() -> None:
    """N 个协程各自一条 trace：根唯一、父子关系指向自己的根、worker 不混。"""
    result = asyncio.run(day15.demo_concurrency(runs=3, steps=2))

    assert result["ok"] is True
    assert len(result["traces"]) == 3
    for row in result["traces"]:
        assert row["roots"] == 1
        assert row["steps"] == 2
        assert row["same_worker"] is True
        assert row["parent_ok"] is True


# ---------- 回调收集器（课程 02） ----------


def test_callback_collector_builds_tree() -> None:
    """回调的 parent_run_id 决定父子关系；token 与工具输出在关节点时补齐。"""
    collector = day15.CallbackCollector()
    collector.start("root", "LangGraph", "chain")
    collector.start("llm-1", "gpt-4o-mini", "llm", "root", n_messages=2)
    collector.end("llm-1", output="先查制度", input_tokens=10, output_tokens=4, total_tokens=14)
    collector.start("tool-1", "search_expense_policy", "tool", "llm-1", input="交通")
    collector.end("tool-1", output="高铁二等座全额报销")
    collector.end("root", output={"messages": []})

    tree = collector.tree()

    assert len(tree) == 1
    assert tree[0]["name"] == "LangGraph"
    llm = tree[0]["children"][0]
    assert llm["kind"] == "llm"
    assert llm["attributes"]["input_tokens"] == 10
    # 工具挂在 LLM 下面：这正是「模型决定调用 → 工具执行」的因果关系
    assert llm["children"][0]["name"] == "search_expense_policy"
    assert "高铁" in llm["children"][0]["attributes"]["output"]
    assert collector.records[0]["trace_id"] == collector.records[0]["span_id"]


def test_callback_collector_orphan_parent_falls_back_to_root() -> None:
    """父节点没被记录（例如被过滤的通用 chain）时挂到根上，而不是整条记录丢失。"""
    collector = day15.CallbackCollector()
    collector.start("root", "LangGraph", "chain")
    collector.start("llm-1", "gpt-4o-mini", "llm", "unknown-parent")

    tree = collector.tree()
    assert tree[0]["span_id"] == "root"
    assert tree[0]["children"][0]["span_id"] == "llm-1"


def test_extract_usage_from_llm_result() -> None:
    """token 用量两种形态都要认：llm_output.token_usage 与 message.usage_metadata。"""
    with_output = SimpleNamespace(
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        generations=[],
    )
    assert day15._extract_usage(with_output) == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    with_metadata = SimpleNamespace(
        llm_output=None,
        generations=[[SimpleNamespace(message=SimpleNamespace(usage_metadata={
            "input_tokens": 7, "output_tokens": 2, "total_tokens": 9
        }))]],
    )
    assert day15._extract_usage(with_metadata) == {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}

    assert day15._extract_usage(SimpleNamespace(llm_output=None, generations=[])) == {}


def test_callback_timeline_offsets_from_first_event() -> None:
    """时间轴以第一条事件为 0 点——前端画甘特条只关心相对位置。"""
    started = time.perf_counter()
    rows = [
        {
            "name": "chat",
            "kind": "llm",
            "status": "ok",
            "start_ts": started + 0.2,
            "duration_ms": 120.0,
            "parent_id": None,
            "attributes": {"total_tokens": 42, "model": "gpt-4o-mini"},
        },
        {
            "name": "tool:x",
            "kind": "tool",
            "status": "error",
            "start_ts": started + 0.1,
            "duration_ms": 3.0,
            "parent_id": "chat",
            "attributes": {"input": "{}"},
        },
    ]

    timeline = day15.callback_timeline(rows, started)

    assert [row["name"] for row in timeline] == ["tool:x", "chat"]
    assert timeline[0]["offset_ms"] == 0
    assert timeline[1]["offset_ms"] == 100
    assert "tokens=42" in timeline[1]["detail"]


# ---------- 平台接入就绪状态 ----------


def test_platform_status_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """没配 key 时 configured=false 并列出缺项；配齐后为 true（只读，不改环境）。"""
    # 关掉 .env 桥接，避免本机 .env 里的真实 key 干扰断言
    monkeypatch.setattr(day15, "_DOTENV_LOADED", True)
    for key in (
        "LANGSMITH_TRACING",
        "LANGSMITH_API_KEY",
        "LANGSMITH_PROJECT",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
    ):
        monkeypatch.delenv(key, raising=False)

    status = day15.platform_status()
    assert status["langsmith"]["configured"] is False
    assert "LANGSMITH_API_KEY" in status["langsmith"]["missing"]

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_pt_test")
    monkeypatch.setenv("LANGSMITH_PROJECT", "agent-course-day15")

    status = day15.platform_status()
    assert status["langsmith"]["configured"] is True
    assert status["langsmith"]["project"] == "agent-course-day15"


def test_langfuse_missing_gives_install_hint() -> None:
    """Langfuse 是可选依赖：没装时给 pip 命令，而不是 ImportError 堆栈。"""
    try:
        import langfuse  # noqa: F401
    except ImportError:
        with pytest.raises(ValueError, match="pip install langfuse"):
            day15._require_langfuse()
    else:  # pragma: no cover - 装了 langfuse 的环境走这里
        day15._require_langfuse()


# ---------- 跨进程合并（课程 05） ----------


def test_merge_timeline_pairs_client_and_server() -> None:
    """同一 trace 的两侧事件按时间合并：客户端说「调了什么」，服务端说「跑了多久」。"""
    client_rows = [
        {"type": "tool_call", "name": "submit_expense", "ts": 100.0, "args": {"amount": 78}},
        {"type": "answer", "name": "answer", "ts": 102.0, "answer": "已提交", "duration_ms": 2000},
    ]
    server_rows = [
        {"tool": "submit_expense", "ts": 100.2, "dur_ms": 12.5, "status": "ok", "args": {"amount": 78}},
    ]

    merged = day15.merge_timeline(client_rows, server_rows)

    assert [row["side"] for row in merged] == ["client", "server", "client"]
    assert merged[0]["offset_ms"] == 0
    assert merged[1]["offset_ms"] == 200
    assert merged[1]["duration_ms"] == 12.5
    assert "已提交" in merged[2]["detail"]


# ---------- 真实文件：落盘 / 回放 / 反馈 ----------


def test_list_traces_and_feedback_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """写盘 → 回放成树 → 聚合指标 → 反馈落库，整条闭环能在临时目录里跑通。"""
    trace_file = tmp_path / "traces.jsonl"
    feedback_file = tmp_path / "feedback.jsonl"
    monkeypatch.setattr(day15, "TRACE_FILE", trace_file)
    monkeypatch.setattr(day15, "FEEDBACK_FILE", feedback_file)

    tracer = day15.SpanTracer(path=trace_file)
    with tracer.span("agent-run", kind="agent", question="报销 45 元", termination="normal", steps=2):
        with tracer.span("chat", kind="llm", input_tokens=30, output_tokens=8):
            pass

    feedback = day15.record_feedback(
        Day15FeedbackRequest(trace_id=day15.load_records(trace_file)[0]["trace_id"], value=1, comment="准")
    )
    assert feedback["summary"]["likes"] == 1
    assert feedback["summary"]["like_rate"] == 1.0

    data = day15.list_traces(limit=5)
    assert len(data["traces"]) == 1
    assert data["traces"][0]["question"] == "报销 45 元"
    assert data["traces"][0]["metrics"]["input_tokens"] == 30
    assert data["metrics"]["traces"] == 1
    assert data["feedback"]["likes"] == 1


def test_load_records_skips_broken_lines(tmp_path: Path) -> None:
    """日志坏行不能让查询接口 500：跳过坏行，保留好行。"""
    path = tmp_path / "traces.jsonl"
    path.write_text('{"trace_id": "t1"}\nnot-json\n\n{"trace_id": "t2"}\n', encoding="utf-8")

    rows = day15.load_records(path)
    assert [row["trace_id"] for row in rows] == ["t1", "t2"]
    assert day15.load_records(tmp_path / "missing.jsonl") == []


# ---------- MCP Server 端：工具 schema 里必须有 trace_id ----------


def test_expense_server_exposes_trace_id() -> None:
    """跨进程追踪的前提：每个工具都把 trace_id 声明成必填参数。"""
    pytest.importorskip("mcp")
    module = importlib.import_module("app.modules.study.services.mcp_servers.expense_server")
    manager = getattr(module.mcp, "_tool_manager", None)
    tools = list(getattr(manager, "_tools", {}).values()) if manager else []
    by_name = {getattr(tool, "name", ""): tool for tool in tools}

    assert set(by_name) == {"search_expense_policy", "submit_expense"}
    for tool in by_name.values():
        schema = tool.parameters
        assert "trace_id" in schema["properties"]
        assert "trace_id" in schema["required"]


def test_expense_server_writes_span_with_trace_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Server 端日志以 trace_id 为钥匙：一行 grep 就能把两侧拼起来。"""
    pytest.importorskip("mcp")
    module = importlib.import_module("app.modules.study.services.mcp_servers.expense_server")
    log_path = tmp_path / "server_spans.jsonl"
    monkeypatch.setattr(module, "SPAN_LOG", log_path)

    module.log_span("abc123", "submit_expense", {"amount": 78}, 12.5, "ok")

    rows = day15.load_records(log_path)
    assert rows[0]["trace_id"] == "abc123"
    assert rows[0]["tool"] == "submit_expense"
    assert rows[0]["dur_ms"] == 12.5
    assert rows[0]["source"] == "server"
    assert rows[0]["ts"] > 0


def test_expense_server_env_var_selects_log_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """宿主通过 DAY15_SERVER_SPANS 下发路径——子进程与宿主写同一份目录。"""
    pytest.importorskip("mcp")
    target = Path(os.getcwd()) / "var" / "day15" / "unit-test-server.jsonl"
    monkeypatch.setenv("DAY15_SERVER_SPANS", str(target))
    module = importlib.import_module("app.modules.study.services.mcp_servers.expense_server")
    module = importlib.reload(module)

    assert module.SPAN_LOG == target
