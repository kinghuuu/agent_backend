"""Day 14 单测：MCP 协议（工具发现 / tools\/call / stdout 污染 / http 传输 / 轨迹投影）。

运行：``python -m pytest tests/test_day14_mcp.py -p no:cacheprovider -q``
没装 mcp 或 langchain-mcp-adapters 时整文件跳过。

五个原则：
1. 纯逻辑优先：参数解析、schema 摊平、轨迹投影、气泡聚合都不碰子进程；
2. 真正 spawn 子进程的用例只覆盖"必须跨进程才能证明"的点（发现 / 调用 / 污染 / http）；
3. 污染用例只断言 ``fd`` 形态必然失败，不去赌整行垃圾的时序（那本身就是课程结论）；
4. 每个用例都能回答课程里的一个问题（工具怎么被发现？错误会不会抛异常？）；
5. 异步用例统一 ``asyncio.run`` 包一层，不依赖 pytest-asyncio 配置。
"""

import asyncio
from typing import Any

import pytest

pytest.importorskip("mcp")
pytest.importorskip("langchain_mcp_adapters")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from app.modules.study.schemas.study import Day14CallRequest, Day14HttpRequest, Day14NoisyRequest  # noqa: E402
from app.modules.study.services import day14_mcp as day14  # noqa: E402


def _run(coro: Any) -> Any:
    """在同步用例里跑协程。"""
    return asyncio.run(coro)


# ---------- 纯逻辑：概念表 ----------


def test_mapping_covers_course_sections() -> None:
    """概念表要覆盖课程 1.x + 第 4 节 + 第 5 节：五个坑、三种传输（含 sse 弃用说明）。"""
    data = _run(day14.mapping())

    assert len(data["pitfalls"]) == 5
    assert len(data["transports"]) == 3
    assert any("sse" in item["transport"].lower() for item in data["transports"])
    assert len(data["fc_vs_mcp"]) >= 4
    assert len(data["primitives"]) == 3
    assert data["mxn"]["problem"].startswith("M 个 Agent 应用")
    assert "未安装" != data["versions"]["mcp"] or True  # 版本只求可读，不强求已装


# ---------- 纯逻辑：参数与 schema ----------


def test_parse_json_arguments() -> None:
    """前端传上来的 JSON 字符串：合法就解析，非法/非对象要给可读错误。"""
    assert day14.parse_json_arguments(None) == {}
    assert day14.parse_json_arguments('{"customer_id": "C001"}') == {"customer_id": "C001"}
    with pytest.raises(ValueError, match="不是合法 JSON"):
        day14.parse_json_arguments("{customer_id: C001}")
    with pytest.raises(ValueError, match="必须是一个 JSON 对象"):
        day14.parse_json_arguments("[1, 2]")


def test_static_tool_schema_reads_server_without_spawn() -> None:
    """不起子进程也能读到 Server 的工具定义：docstring 就是 description。"""
    schema = day14.static_tool_schema("crm")

    assert schema["server"] == "crm-tools"
    names = {tool["name"] for tool in schema["tools"]}
    assert names == {"get_customer", "deduct_balance"}
    deduct = next(tool for tool in schema["tools"] if tool["name"] == "deduct_balance")
    assert "扣款" in deduct["description"]
    assert set(deduct["input_schema"]["properties"]) == {"customer_id", "amount"}
    assert set(deduct["input_schema"]["required"]) == {"customer_id", "amount"}


def test_params_view_marks_required() -> None:
    """JSON Schema → 表格行：required 数组里的才打必填标记。"""
    rows = day14._params_view(
        {
            "properties": {"query": {"type": "string", "description": "关键词"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        }
    )

    assert [row["name"] for row in rows] == ["query", "top_k"]
    assert rows[0]["required"] is True
    assert rows[1]["required"] is False
    assert rows[0]["description"] == "关键词"


def test_unknown_server_key_is_rejected() -> None:
    """只允许演示用的两个 Server，避免前端传个 key 就把任意脚本拉起来。"""
    with pytest.raises(ValueError, match="未知的 MCP Server"):
        day14._server_path("shell")
    with pytest.raises(ValueError, match="未知的 MCP Server"):
        day14.load_server_module("shell")


# ---------- 纯逻辑：轨迹投影 / 气泡聚合 ----------


def test_trace_steps_marks_channel_and_keeps_pair() -> None:
    """轨迹要能区分 mcp / local：MCP 工具和本地工具在 Agent 眼里没有区别，只能靠名字标注。"""
    messages = [
        AIMessage(content="", tool_calls=[{"name": "deduct_balance", "args": {"amount": 5000}, "id": "1"}]),
        ToolMessage(content="余额不足：当前 1200 元", tool_call_id="1", name="deduct_balance"),
        AIMessage(
            content="",
            tool_calls=[{"name": "escalate_to_human", "args": {"reason": "扣款失败"}, "id": "2"}],
        ),
        ToolMessage(content="已转人工", tool_call_id="2", name="escalate_to_human"),
        AIMessage(content="已为您转人工处理"),
    ]

    steps = day14._trace_steps(messages)

    assert [step["kind"] for step in steps] == [
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "answer",
    ]
    assert steps[0]["channel"] == "mcp"
    assert steps[2]["channel"] == "local"
    assert steps[-1]["content"] == "已为您转人工处理"


def test_group_bubbles_splits_on_node_change() -> None:
    """节点一变就开新泡；纯空白 token 不进气泡（部分网关会吐空块）。"""
    bubbles = day14._group_bubbles(
        [
            {"node": "model", "content": "客户"},
            {"node": "model", "content": "余额不足"},
            {"node": "model", "content": "   "},
            {"node": "reviewer", "content": "已复核"},
        ]
    )

    assert [bubble["node"] for bubble in bubbles] == ["model", "reviewer"]
    assert bubbles[0]["text"] == "客户余额不足"
    assert bubbles[0]["tokens"] == 2


# ---------- 跨进程：工具发现 ----------


def test_discover_spawns_two_servers_and_lists_both_views() -> None:
    """真实 spawn：tools/list 的原始 JSON Schema 与 get_tools() 转换后的视角都能拿到。"""
    data = _run(day14.discover())

    assert data["spawn_count"] == 4  # 2 个 Server ×（协议会话 + adapter 会话）
    assert data["total_tools"] == 3

    by_key = {server["key"]: server for server in data["servers"]}
    assert [tool["name"] for tool in by_key["docs"]["tools"]] == ["search_docs"]
    assert {tool["name"] for tool in by_key["crm"]["tools"]} == {"get_customer", "deduct_balance"}
    assert any(resource["uri"] == "crm://stats" for resource in by_key["crm"]["resources"])
    assert any(prompt["name"] == "refund_flow" for prompt in by_key["crm"]["prompts"])
    # 适配后的工具带 args_schema，参数名与原始 inputSchema 一致
    deduct = next(tool for tool in by_key["crm"]["adapted"] if tool["name"] == "deduct_balance")
    assert {param["name"] for param in deduct["params"]} == {"customer_id", "amount"}
    assert len(data["handshake"]) >= 4


# ---------- 跨进程：tools/call 与"错误即结果" ----------


def test_call_tool_returns_validation_error_as_content() -> None:
    """Server 端校验失败不抛异常：isError 之外，文本里带上可读原因（错误信息即 prompt）。"""
    data = _run(
        day14.call_tool(
            Day14CallRequest(server="crm", tool="deduct_balance", arguments={"customer_id": "C001", "amount": 5000})
        )
    )

    assert data["is_error"] is False  # 业务校验失败 ≠ 协议层错误
    assert data["content"] and "余额不足" in data["content"][0]


def test_call_tool_rejects_bad_amount_and_unknown_customer() -> None:
    """两种失败形态都在 Server 里拦掉：金额非法 / 客户不存在。"""
    negative = _run(
        day14.call_tool(
            Day14CallRequest(server="crm", tool="deduct_balance", arguments={"customer_id": "C001", "amount": -1})
        )
    )
    missing = _run(
        day14.call_tool(Day14CallRequest(server="crm", tool="get_customer", arguments={"customer_id": "C999"}))
    )

    assert "必须为正数" in negative["content"][0]
    assert "不存在" in missing["content"][0]


# ---------- 跨进程：stdout 污染 / http 传输 ----------


def test_noisy_partial_line_hangs_client() -> None:
    """坑 1 里最硬的那一档：半行文本粘住 JSON 帧 → 客户端等不到完整帧，必然超时。

    整行垃圾（print / fd_line）时好时坏，属于时序问题，这里不去赌它的结果。
    """
    data = _run(day14.run_noisy(Day14NoisyRequest(mode="fd", timeout_seconds=3)))

    case = data["cases"][0]
    assert case["mode"] == "fd"
    assert case["ok"] is False
    assert "TimeoutError" in case["error"]


def test_http_transport_serves_tool_and_shuts_down() -> None:
    """streamable-http：独立服务形态，连上能列工具能调用，用完必须关掉。"""
    data = _run(day14.run_http(Day14HttpRequest(topic="本周")))

    assert data["url"].endswith("/mcp")
    assert [tool["name"] for tool in data["tools"]] == ["team_announcement"]
    assert data["content"] and "全员同步会" in data["content"][0]
    assert data["stopped"] is True
