"""Day 10 单测：LangGraph 状态图（reducer / 条件边 / ReAct 环 / 并行 / 子图），不联网。

运行：``python -m pytest tests/test_day10_graph.py -p no:cacheprovider -q``
没装 langgraph 时整文件跳过，不影响其它测试。

两个原则：
1. 一律用 ``ScriptedLLM`` 顶替真实模型——图的控制流与模型质量无关，脚本化的假模型
   才能断言"第几步走了哪条边"；
2. 异步用例统一用 ``asyncio.run`` 包一层，不依赖 pytest-asyncio 的配置。
"""

import asyncio
from typing import Any

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402

from app.modules.study.schemas.study import Day10ParallelRequest  # noqa: E402
from app.modules.study.services import day10_langgraph as day10  # noqa: E402


def _run(coro: Any) -> Any:
    """在同步用例里跑协程（避免依赖 pytest-asyncio 的 mode 配置）。"""
    return asyncio.run(coro)


class ScriptedLLM:
    """按脚本依次吐消息的假模型：不联网、可断言调用次数。

    ``bind_tools`` 直接返回自己——工具 schema 不参与控制流，测试只关心"下一步去哪"。

    ⚠️ ``default_factory`` 必须是**工厂**而不是现成对象：消息对象自带自动生成的 id，
    反复返回同一个实例会让 ``add_messages`` 认为是"同 id 更新"而不是"追加"，
    于是最后一条永远是之前那条 ToolMessage，环被提前判定为收敛。
    """

    def __init__(
        self, script: list[AIMessage], default_factory: Any | None = None
    ) -> None:
        self.script = list(script)
        self.default_factory = default_factory or (lambda: AIMessage(content="（脚本已用完）"))
        self.calls = 0

    def bind_tools(self, tools: list[Any]) -> "ScriptedLLM":
        return self

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls += 1
        return self.script.pop(0) if self.script else self.default_factory()


def _tool_call(call_id: str = "call_1", order_id: str = "SO20261101001") -> AIMessage:
    """一条"决定调工具"的 AI 消息。"""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "get_order_status", "args": {"order_id": order_id}, "id": call_id}
        ],
    )


# ---------- 01 状态与 reducer ----------


def test_state_reducer_demo_semantics() -> None:
    """四个实验的结论必须与课程一致：覆盖 / 追加 / 同 id 更新 / 重复追加。"""
    result = _run(day10.state_reducer_demo())
    by_key = {item["key"]: item for item in result["experiments"]}

    assert by_key["no_reducer"]["result"] == {"log": ["two"]}  # 后写的覆盖前写的
    assert by_key["operator_add"]["result"] == {"log": ["one", "two"]}  # 追加
    # node_two 确实"看到"了 ['one']：读语义对、写语义错，这才是难点
    assert by_key["no_reducer"]["patches"][1]["received"]["log"] == ["one"]

    messages = by_key["add_messages"]["result"]["messages"]
    assert len(messages) == 2  # 同 id 是"更新"不是"追加"
    assert messages[1]["content"] == "整机保修 24 个月，电池 12 个月。"

    # 坑：节点返回全量 + reducer 追加 → 长度爆炸
    assert len(by_key["total_copy"]["result"]["log"]) > 3


# ---------- 03 条件边 ----------


def test_route_intent_is_pure_and_confidence_gated() -> None:
    """路由函数只读 state：低置信度先转人工，其余按 intent 分流。"""
    assert day10.route_intent("policy", 0.59, 0.6) == "human"
    assert day10.route_intent("order", 0.9, 0.6) == "order"
    assert day10.route_intent("policy", 1.0, 0.6) == "policy"
    assert day10.route_intent("chat", 0.8, 0.6) == "chat"
    # 未知 intent 兜底到 chat，绝不返回映射表里没有的 key
    assert day10.route_intent("unknown", 0.99, 0.6) == "chat"


# ---------- 04 ReAct 环 ----------


def test_react_graph_loops_with_add_messages() -> None:
    """一次工具往返：agent → tools → agent，messages 靠 add_messages 累积不丢。"""
    llm = ScriptedLLM([_tool_call(), AIMessage(content="订单已发货，预计 11-09 送达")])
    graph = day10._build_react_graph(llm, day10._build_react_tools(), step_limit=6, as_subgraph=False)

    state = _run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="SO20261101001 到哪了？")], "steps": 0, "degraded": False}
        )
    )

    types = [message.type for message in state["messages"]]
    assert types == ["human", "ai", "tool", "ai"]  # 历史没被覆盖 = add_messages 生效
    assert state["steps"] == 2  # 两次 agent
    assert state["messages"][-1].content == "订单已发货，预计 11-09 送达"
    assert state["degraded"] is False


def test_react_graph_step_limit_degrades_instead_of_raising() -> None:
    """练习 2：工具一直失败时，step_limit 到顶就走 degrade 节点，而不是烧到 recursion_limit。"""
    llm = ScriptedLLM(
        [_tool_call("c1"), _tool_call("c2"), AIMessage(content="降级：基于已知信息，请稍后重试")],
        default_factory=lambda: AIMessage(content="降级回答"),
    )
    graph = day10._build_react_graph(llm, day10._build_react_tools(broken=True), step_limit=2, as_subgraph=False)

    state = _run(
        graph.ainvoke(
            {
                "messages": [HumanMessage(content="SO20261101001 到哪了？")],
                "steps": 0,
                "degraded": False,
            }
        )
    )

    assert state["degraded"] is True
    assert state["steps"] == 2
    # 降级节点产出的最后一条是正常 AI 回答（没有 tool_calls），可以直接回给用户
    assert getattr(state["messages"][-1], "tool_calls", None) in (None, [])
    assert "降级" in state["messages"][-1].content


def test_react_graph_broken_tool_returns_error_message() -> None:
    """工具持续失败时返回错误文本（self-heal 契约），而不是抛异常打断图。"""
    tools = day10._build_react_tools(broken=True)
    tool = {item.name: item for item in tools}["get_order_status"]
    assert "错误" in str(_run(tool.ainvoke({"type": "tool_call", "name": "get_order_status", "args": {"order_id": "SO1"}, "id": "c1"})))


def test_recursion_limit_is_only_a_fuse() -> None:
    """没有语义刹车时，recursion_limit 会抛 GraphRecursionError——所以 run_react 必须捕获它。"""
    llm = ScriptedLLM([], default_factory=_tool_call)
    graph = day10._build_react_graph(llm, day10._build_react_tools(), step_limit=99, as_subgraph=False)

    with pytest.raises(GraphRecursionError):
        _run(
            graph.ainvoke(
                {"messages": [HumanMessage(content="死循环")], "steps": 0, "degraded": False},
                {"recursion_limit": 4},
            )
        )


def test_sanitize_for_degrade_drops_unpaired_tool_calls() -> None:
    """降级前必须清掉悬空的 tool_calls：配对断裂会让下一次调用直接 400。"""
    messages = [
        HumanMessage(content="查订单"),
        _tool_call(),
        ToolMessage(content="订单不存在", tool_call_id="call_1"),
        HumanMessage(content="那退款呢"),
    ]
    cleaned = day10._sanitize_for_degrade(messages)

    assert not [item for item in cleaned if getattr(item, "tool_calls", None)]
    assert not [item for item in cleaned if isinstance(item, ToolMessage)]
    # 工具结果改写成摘要，信息不丢
    assert any("订单不存在" in day10._text(item) for item in cleaned)


def test_react_subgraph_reports_inner_namespace() -> None:
    """练习 3：子图当节点，subgraphs=True 的流里能看到子图内部事件（带命名空间）。"""
    llm = ScriptedLLM([_tool_call(), AIMessage(content="已发货")])
    graph = day10._build_react_graph(llm, day10._build_react_tools(), step_limit=6, as_subgraph=True)

    async def collect() -> list[tuple[tuple[str, ...], set[str]]]:
        events: list[tuple[tuple[str, ...], set[str]]] = []
        async for namespace, updates in graph.astream(
            {"messages": [HumanMessage(content="SO20261101001 到哪了？")], "steps": 0, "degraded": False},
            stream_mode="updates",
            subgraphs=True,
        ):
            events.append((namespace, set(updates.keys())))
        return events

    events = _run(collect())
    assert any("react_subgraph" in namespace[0] for namespace, _ in events if namespace)
    # 外层节点名是 react_subgraph，子图内部节点是 agent / tools
    outer_nodes = {name for namespace, nodes in events if not namespace for name in nodes}
    inner_nodes = {name for namespace, nodes in events if namespace for name in nodes}
    assert outer_nodes == {"react_subgraph"}
    assert inner_nodes == {"agent", "tools"}


# ---------- 05 并行 / 子图 ----------


def test_parallel_fanout_is_concurrent_and_uses_reducer() -> None:
    """扇出耗时取最大而不是求和；docs 由 operator.add 合并成三条。"""
    result = _run(day10.run_parallel(Day10ParallelRequest(query="退款；订单物流")))
    static = result["static"]

    assert len(static["docs"]) == 3
    assert static["elapsed_s"] < static["sum_s"] * 0.9  # 0.5 < 1.2 * 0.9
    assert "merge" in static["graph"]["nodes"]
    # 汇聚节点的入边有三条：三条边汇到同一个点才算扇入
    incoming = [edge for edge in static["graph"]["edges"] if edge["target"] == "merge"]
    assert len(incoming) == 3


def test_parallel_send_branches_follow_query() -> None:
    """Send 的分支数运行时决定：分号隔出两段就两条分支。"""
    result = _run(day10.run_parallel(Day10ParallelRequest(query="退款；发票政策")))
    dynamic = result["dynamic"]

    assert dynamic["branch_count"] == 2
    assert len(dynamic["docs"]) == 2
    assert all(item.startswith("[sub]") for item in dynamic["docs"])


def test_parallel_subgraph_events_are_namespaced() -> None:
    """子图内部事件带命名空间，且父图 log 不重复追加（父图用覆盖语义）。"""
    result = _run(day10.run_parallel(Day10ParallelRequest(query="退款")))
    subgraph = result["subgraph"]

    scopes = {event["scope"] for event in subgraph["events"]}
    assert "外层" in scopes
    assert "research" in scopes
    assert subgraph["summary"].count("triage 分流") == 1  # 父图写 operator.add 就会变两次


# ---------- 概念表 ----------


def test_mapping_tables_cover_course_checklist() -> None:
    """静态概念表齐全：三件套 / reducer / while 对照 / stream_mode / 六个坑 / Send 三条规矩。"""
    data = _run(day10.mapping())

    assert [item["piece"] for item in data["triple_pieces"]] == ["State", "Node", "Edge"]
    assert len(data["reducer_table"]) == 4
    assert len(data["while_vs_graph"]) == 7
    assert {item["mode"] for item in data["stream_modes"]} == {"updates", "values", "messages", "debug"}
    assert len(data["pitfalls"]) == 6
    assert len(data["send_rules"]) == 3
    assert data["langgraph_version"] != "未安装"


def test_jsonable_handles_messages_and_unknown_objects() -> None:
    """_jsonable 把消息对象转成可序列化 dict，未知对象走 str() 兜底不抛异常。"""
    from langchain_core.messages import HumanMessage as _Human

    assert day10._jsonable({"a": [1, "x"]}) == {"a": [1, "x"]}
    assert day10._jsonable(_Human(content="你好", id="m1")) == {
        "type": "human",
        "id": "m1",
        "content": "你好",
    }
    assert isinstance(day10._jsonable(object()), str)
    assert day10._jsonable(_tool_call())["tool_calls"][0]["name"] == "get_order_status"
