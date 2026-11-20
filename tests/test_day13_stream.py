"""Day 13 单测：流式输出（四模式订阅 / token 分流 / custom 进度 / interrupt + 流式 / SSE 帧）。

运行：``python -m pytest tests/test_day13_stream.py -p no:cacheprovider -q``
没装 langgraph 时整文件跳过。

五个原则：
1. 纯逻辑优先：SSE 帧、路由解析、resume 归一、气泡聚合都不碰网络；
2. interrupt 实验本身**不调模型**，所以端到端可以直接断言（暂停位置、退款次数）；
3. token 分流用脚本化假模型验证图结构（主管不无限乒乓、专家摘要带 name）；
4. 每个用例都能回答课程里的一个坑（只订 messages 会漏暂停吗？重放会扣两次款吗？）；
5. 异步用例统一 ``asyncio.run`` 包一层，不依赖 pytest-asyncio 配置。
"""

import asyncio
from typing import Any

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from app.modules.study.schemas.study import Day13InterruptRequest  # noqa: E402
from app.modules.study.services import day13_stream as day13  # noqa: E402


def _run(coro: Any) -> Any:
    """在同步用例里跑协程（避免依赖 pytest-asyncio 的 mode 配置）。"""
    return asyncio.run(coro)


def _request(**kwargs: Any) -> Any:
    """构造课程 04 的请求对象（默认金额 199、放行、演示续跑）。"""
    return Day13InterruptRequest(**kwargs)


# ---------- 纯逻辑：SSE 帧与响应头 ----------


def test_sse_frame_has_double_newline_and_keeps_chinese() -> None:
    """SSE 帧必须以 ``data: `` 开头、``\\n\\n`` 结尾，且中文不被转义成 \\uXXXX。"""
    frame = day13.sse({"type": "token", "content": "检索中"})

    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert "检索中" in frame
    assert "\\u" not in frame


def test_sse_headers_disable_proxy_buffering() -> None:
    """X-Accel-Buffering: no 是给 Nginx 看的，少了它流式就变"攒批发"。"""
    headers = day13.sse_headers()

    assert headers["X-Accel-Buffering"] == "no"
    assert headers["Cache-Control"] == "no-cache"


# ---------- 纯逻辑：路由解析 / resume 归一 ----------


def test_parse_route_tolerates_fence_and_prose() -> None:
    """thinking 模型常带 ```json 围栏或解释文字，解析要能容忍，失败时返回空串。"""
    assert day13._parse_route('```json\n{"next": "researcher"}\n```') == "researcher"
    assert day13._parse_route('好的，{"next"："writer"}') == "writer"
    assert day13._parse_route("我决定交给 researcher") == "researcher"
    assert day13._parse_route("完全不相关") == ""


def test_normalize_decision_three_forms() -> None:
    """resume 三种形态与 Day 11/12 一致：true / false / 改参数字典。"""
    assert day13._normalize_decision(True) == (True, None, "")
    assert day13._normalize_decision(False)[0] is False
    approved, amount, note = day13._normalize_decision(
        {"approved": True, "amount": 150, "note": "老板批了"}
    )
    assert approved is True
    assert amount == 150.0
    assert note == "老板批了"


# ---------- 纯逻辑：token 过滤与气泡聚合 ----------


def test_token_filter_hides_supervisor_json() -> None:
    """主管的 json 碎片不能进气泡；专家 token 与空白块的判定。"""
    assert day13._should_show_token("supervisor", '{"next": "researcher"}') is False
    assert day13._should_show_token("researcher", "向量检索擅长语义匹配") is True
    assert day13._should_show_token("writer", "   ") is False


def test_group_bubbles_splits_by_node() -> None:
    """节点一变就开新气泡：researcher 的 token 不会混进 writer 的气泡。"""
    rows = [
        {"node": "researcher", "content": "要点一", "offset_ms": 10},
        {"node": "researcher", "content": "；要点二", "offset_ms": 20},
        {"node": "writer", "content": "结论：互补。", "offset_ms": 30},
    ]
    bubbles = day13._group_bubbles(rows)

    assert [item["node"] for item in bubbles] == ["researcher", "writer"]
    assert bubbles[0]["text"] == "要点一；要点二"
    assert bubbles[0]["tokens"] == 2
    assert bubbles[0]["first_offset_ms"] == 10
    assert bubbles[1]["first_offset_ms"] == 30

    grouped = day13._collect_tokens(rows)
    assert grouped["first_token_ms"] == 10
    assert {item["node"] for item in grouped["node_timing"]} == {"researcher", "writer"}


# ---------- 图结构：主管不无限乒乓 / 专家摘要 ----------


class ScriptedLLM:
    """只实现 ``ainvoke`` 的假模型：按 system prompt 关键字返回脚本化内容。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def ainvoke(self, messages: Any) -> AIMessage:
        system = ""
        for message in messages:
            if getattr(message, "type", "") == "system":
                system = day13._text(message)
        if "主管" in system:
            self.calls.append("supervisor")
            return AIMessage(content='{"next": "researcher"}')  # 故意一直派 researcher
        if "研究员" in system:
            self.calls.append("researcher")
            return AIMessage(content="要点：状态图 + 存档")
        self.calls.append("writer")
        return AIMessage(content="结论：互补使用。")

    def bind_tools(self, tools: list[Any]) -> "ScriptedLLM":
        return self


def test_token_graph_stops_pingpong_and_names_experts() -> None:
    """主管脚本化地一直派 researcher 也不会无限循环：硬守卫把它推到 writer → FINISH。"""
    graph = day13._build_token_graph(ScriptedLLM())

    state = _run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="RAG 是什么？")],
                "findings": [],
                "dispatches": 0,
            }
        )
    )

    assert state["next"] == "FINISH"
    assert state["dispatches"] == 3  # 三次主管决策：researcher → writer → FINISH
    named = {getattr(message, "name", "") for message in state["messages"]}
    assert {"researcher", "writer"} <= named
    assert any(item.startswith("[researcher]") for item in state["findings"])
    assert any(item.startswith("[writer]") for item in state["findings"])


# ---------- 课程 04：interrupt + 流式（不调模型，完全确定性） ----------


def test_messages_only_never_sees_interrupt() -> None:
    """今天的主坑：只订 messages → 事件 0 条、图已暂停、interrupt_seen=False。"""
    result = _run(day13.run_interrupt_stream(_request(amount=199, with_resume=False)))

    messages_only = result["messages_only"]
    assert messages_only["event_count"] == 0
    assert messages_only["paused"] is True
    assert messages_only["next"] == ["refund"]
    assert messages_only["interrupt_seen"] is False

    with_updates = result["with_updates"]
    assert with_updates["interrupt_seen"] is True
    assert with_updates["paused"] is True
    assert with_updates["interrupt"]["amount"] == 199
    # 暂停信号所在的那条事件必须是 updates 模式，且节点名固定为 __interrupt__
    kinds = [(row["mode"], row["node"]) for row in with_updates["events"]]
    assert ("updates", "__interrupt__") in kinds
    assert result["resume"] is None


def test_resume_approve_records_refund_once() -> None:
    """双订阅拿到审批后再 resume：图跑完、退款流水只记一条（重放不重复扣款）。"""
    result = _run(day13.run_interrupt_stream(_request(amount=88, decision=True)))

    assert result["with_updates"]["interrupt"]["amount"] == 88
    assert result["resume"]["paused"] is False
    assert len(result["refunds"]) == 1
    assert result["refunds"][0]["amount"] == 88
    # 续跑阶段能收到 refund 节点的 update 事件（副作用发生在 resume 之后）
    assert any(row["kind"] == "update" for row in result["resume"]["events"])


def test_resume_reject_keeps_money() -> None:
    """驳回：不产生任何退款流水（写操作被闸门拦住）。"""
    result = _run(day13.run_interrupt_stream(_request(amount=199, decision=False)))

    assert result["refunds"] == []
    assert result["resume"]["paused"] is False


def test_resume_with_patched_amount() -> None:
    """人工改金额后放行：流水记的是改过的金额，事件里标注人工改过。"""
    result = _run(
        day13.run_interrupt_stream(_request(amount=199, decision={"approved": True, "amount": 150}))
    )

    assert len(result["refunds"]) == 1
    assert result["refunds"][0]["amount"] == 150
    # patched=True 说明流水里的 150 来自人工改写，而不是模型/入参
    assert result["refunds"][0]["patched"] is True


# ---------- 概念表完整性 ----------


def test_mapping_covers_all_four_modes_and_five_pitfalls() -> None:
    """概念表必须能对齐课程：四种常用模式 + debug、四条 interrupt 规则、五个坑、四类事件协议。"""
    data = _run(day13.mapping())

    modes = {row["mode"] for row in data["stream_modes"]}
    assert {"values", "updates", "messages", "custom"} <= modes
    assert len(data["interrupt_rules"]) >= 4
    assert len(data["pitfalls"]) == 5
    protocol = {row["type"] for row in data["event_protocol"]}
    assert {"token", "interrupt", "node_start", "done"} <= protocol
    assert len(data["sse_rules"]) >= 3
    assert data["presets"]["modes"] and data["presets"]["sse"]
