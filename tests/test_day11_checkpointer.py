"""Day 11 单测：Checkpointer 与人机协同（存档 / 续跑 / 中断 / 重放 / 时间旅行），不联网。

运行：``python -m pytest tests/test_day11_checkpointer.py -p no:cacheprovider -q``
没装 langgraph 时整文件跳过；SQLite 持久化用例再单独判断 langgraph-checkpoint-sqlite。

三个原则：
1. 用脚本化假模型顶替真实 LLM——checkpointer 的行为与模型质量无关；
2. 异步用例统一 ``asyncio.run`` 包一层，不依赖 pytest-asyncio 配置；
3. 每个用例都能单独回答课程里的一个问题（续跑？隔离？暂停？重放几次？）。
"""

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.modules.study.services import day11_checkpointer as day11  # noqa: E402


def _run(coro: Any) -> Any:
    """在同步用例里跑协程（避免依赖 pytest-asyncio 的 mode 配置）。"""
    return asyncio.run(coro)


class ScriptedLLM:
    """按脚本依次吐消息的假模型；`bind_tools` 返回自己（schema 不参与控制流）。

    与 Day 10 同款教训：``default_factory`` 必须是工厂——消息对象自带 id，
    反复返回同一个实例会被 ``add_messages`` 当成"同 id 更新"。
    """

    def __init__(self, script: list[AIMessage], default_factory: Any | None = None) -> None:
        self.script = list(script)
        self.default_factory = default_factory or (lambda: AIMessage(content="（脚本已用完）"))
        self.calls = 0

    def bind_tools(self, tools: list[Any]) -> "ScriptedLLM":
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.calls += 1
        return self.script.pop(0) if self.script else self.default_factory()


class EchoLLM:
    """把最后一条消息原文回显的假模型：用来断言"重跑时 state 里的参数确实换了"。"""

    def bind_tools(self, tools: list[Any]) -> "EchoLLM":
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        text = day11._text(messages[-1]) if isinstance(messages, list) else str(messages)
        return AIMessage(content=text)


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    """一条"决定调工具"的 AI 消息。"""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


# ---------- 01 存档 / 续跑 / 隔离 ----------


def test_chat_graph_resumes_same_thread_and_resets_isolated_thread() -> None:
    """同 thread 第二轮：turns=2、四条消息；换 thread：turns=1、两条消息。"""
    llm = ScriptedLLM(
        [
            AIMessage(content="记住了"),
            AIMessage(content="你叫小胡，最喜欢的数字是 7"),
            AIMessage(content="抱歉，你还没有告诉过我"),
        ]
    )
    app = day11._build_chat_graph(llm, MemorySaver())
    cfg_a = day11._config("user-a")
    cfg_b = day11._config("user-b")

    _run(app.ainvoke({"messages": [HumanMessage(content="我叫小胡，最喜欢的数字是 7")], "turns": 0}, cfg_a))
    # 关键：第二轮不传 turns、不传历史，只带一条新消息
    state = _run(app.ainvoke({"messages": [HumanMessage(content="我叫什么？")]}, cfg_a))

    assert state["turns"] == 2  # 计数来自存档帧，不是 Python 变量
    assert [item.type for item in state["messages"]] == ["human", "ai", "human", "ai"]
    assert "小胡" in day11._text(state["messages"][-1])

    snap = _run(app.aget_state(cfg_a))
    assert tuple(snap.next) == ()  # 跑完 = next 为空
    assert snap.values["turns"] == 2

    state_b = _run(app.ainvoke({"messages": [HumanMessage(content="我叫什么？")], "turns": 0}, cfg_b))
    assert state_b["turns"] == 1
    assert len(state_b["messages"]) == 2  # 跨 thread 隔离：看不到 A 的历史


def test_snapshot_view_exposes_four_elements() -> None:
    """StateSnapshot 四要素：values / next / tasks / created_at 都能给前端。"""
    llm = ScriptedLLM([AIMessage(content="好的")])
    app = day11._build_chat_graph(llm, MemorySaver())
    cfg = day11._config("snap")
    _run(app.ainvoke({"messages": [HumanMessage(content="你好")], "turns": 0}, cfg))

    view = day11._snapshot_view(_run(app.aget_state(cfg)))
    assert view["values"]["turns"] == 1
    assert view["next"] == []
    assert view["tasks"] == []
    assert view["created_at"]
    assert view["checkpoint_id"]
    assert view["messages_count"] == 2


def test_frames_are_a_chain_newest_first() -> None:
    """历史帧按时间倒序串成链，每帧都带 checkpoint_id。"""
    llm = ScriptedLLM([AIMessage(content="一"), AIMessage(content="二")])
    app = day11._build_chat_graph(llm, MemorySaver())
    cfg = day11._config("frames")
    _run(app.ainvoke({"messages": [HumanMessage(content="第一句")], "turns": 0}, cfg))
    _run(app.ainvoke({"messages": [HumanMessage(content="第二句")]}, cfg))

    frames = _run(day11._frames(app, cfg, limit=20))
    assert len(frames) >= 3  # 初始帧 + 两轮结果帧
    assert frames[0]["messages_count"] == 4  # 新的在前
    assert all(frame["checkpoint_id"] for frame in frames)
    assert frames[0]["index"] == 0


# ---------- 02 确认门 ----------


def _approval_app(script: list[AIMessage]) -> tuple[Any, dict[str, Any]]:
    """搭一张确认门图并返回（图, 待审批状态）；首轮脚本应走到 refund_order。"""
    llm = ScriptedLLM(script)
    app = day11._build_approval_graph(llm, day11._build_approval_tools(), MemorySaver())
    cfg = day11._config("ticket-1")
    state = _run(
        app.ainvoke(
            {
                "messages": [HumanMessage(content="订单 A1001 不想要了，请退款 299 元")],
                "approved": False,
                "patched_args": {},
                "path": [],
            },
            cfg,
        )
    )
    return app, cfg


def _refund_script(final: str = "已为您退款") -> list[AIMessage]:
    """先查订单（安全），再退款（敏感：触发 interrupt），恢复后再回一句。"""
    return [
        _tool_call("get_order", {"order_id": "A1001"}, "c1"),
        _tool_call("refund_order", {"order_id": "A1001", "amount": 299}, "c2"),
        AIMessage(content=final),
    ]


def test_approval_pauses_at_superstep_boundary() -> None:
    """命中敏感工具 → 图冻结：next=('approval',)，tasks 里带 interrupt 载荷。"""
    app, cfg = _approval_app(_refund_script())
    snap = _run(app.aget_state(cfg))
    view = day11._snapshot_view(snap)

    assert view["next"] == ["approval"]
    assert view["has_interrupt"] is True
    payload = view["tasks"][0]["interrupts"][0]
    assert payload["tool"] == "refund_order"
    assert payload["args"]["amount"] == 299
    # 暂停时退款还没执行：历史里只有 get_order 的结果
    tools = day11._tool_results(snap.values["messages"])
    assert [row["tool"] for row in tools] == ["get_order"]


def test_resume_true_executes_refund_once() -> None:
    """resume=True：approval 重放但只有 interrupt，退款在 tools 节点执行一次。"""
    app, cfg = _approval_app(_refund_script())
    state = _run(app.ainvoke(Command(resume=True), cfg))

    tools = day11._tool_results(state["messages"])
    refunds = [row for row in tools if row["tool"] == "refund_order"]
    assert len(refunds) == 1  # 没有重复扣款
    assert "299" in refunds[0]["content"]
    assert refunds[0]["blocked"] is False
    assert day11._last_answer(state["messages"]) == "已为您退款"
    assert tuple(_run(app.aget_state(cfg)).next) == ()


def test_resume_false_blocks_sensitive_tool_but_keeps_pairing() -> None:
    """resume=False：退款不执行，但仍回一条 ToolMessage（tool_calls 配对不能断）。"""
    app, cfg = _approval_app(_refund_script("很抱歉，退款未获批准"))
    state = _run(app.ainvoke(Command(resume=False), cfg))

    tools = day11._tool_results(state["messages"])
    refunds = [row for row in tools if row["tool"] == "refund_order"]
    assert len(refunds) == 1
    assert refunds[0]["blocked"] is True
    assert "已拦截" in refunds[0]["content"]
    # 每个 tool_call 都有配对的 ToolMessage：2 次调用 → 2 条结果
    tool_call_ids = {
        call["id"] for message in state["messages"] for call in (getattr(message, "tool_calls", None) or [])
    }
    tool_message_ids = {
        message.tool_call_id for message in state["messages"] if isinstance(message, ToolMessage)
    }
    assert tool_call_ids == tool_message_ids


def test_resume_with_patched_args_overrides_model_params() -> None:
    """练习 2：人工传 {'amount': 199}，最终退款金额是 199 而不是 299。"""
    app, cfg = _approval_app(_refund_script())
    state = _run(app.ainvoke(Command(resume={"amount": 199}), cfg))

    refunds = [row for row in day11._tool_results(state["messages"]) if row["tool"] == "refund_order"]
    assert len(refunds) == 1
    assert "199" in refunds[0]["content"]
    assert "299" not in refunds[0]["content"]
    assert "参数已按人工审批修改" in refunds[0]["content"]
    assert state["approved"] is True
    assert state["patched_args"] == {"amount": 199}


# ---------- 坑 1：节点重放与副作用 ----------


def test_replay_demo_shows_side_effect_runs_twice() -> None:
    """反面写法：暂停时扣一次、恢复重放再扣一次；正面写法只扣一次。"""
    result = _run(day11.replay_demo())
    by_key = {item["key"]: item for item in result["experiments"]}

    assert by_key["bad"]["before_resume"] == 1
    assert by_key["bad"]["after_resume"] == 2  # 重放导致重复副作用
    assert by_key["good"]["before_resume"] == 0
    assert by_key["good"]["after_resume"] == 1  # 副作用在 resume 之后
    assert len(result["rules"]) == 3


# ---------- 04 时间旅行 ----------


def test_time_travel_forks_from_history_without_losing_original() -> None:
    """从 answer 之前那一帧改写 topic 后重跑：答案变了，原历史帧仍在。"""
    app = day11._build_time_travel_graph(EchoLLM(), MemorySaver())
    cfg = day11._config("research-1")

    async def scenario() -> tuple[str, str, bool]:
        first = await app.ainvoke({"messages": [HumanMessage(content="RAG")], "topic": ""}, cfg)
        original = day11._text(first["messages"][-1])
        original_last_id = day11._checkpoint_id_of(await app.aget_state(cfg))

        target = None
        async for snap in app.aget_state_history(cfg):
            if tuple(snap.next or ()) == ("answer",):
                target = snap
                break
        assert target is not None

        # 沿用历史帧的完整 config（含 checkpoint_ns），只拼 thread_id + checkpoint_id 会缺字段
        fork = {"configurable": dict((target.config or {}).get("configurable") or {})}
        # 用 update_state 返回的新 config 重跑（它指向刚写出的分叉帧）
        fork_cfg = await app.aupdate_state(fork, {"topic": "MCP 协议"}, as_node="set_topic")
        new_state = await app.ainvoke(None, fork_cfg)

        frames = await day11._frames(app, cfg, limit=30)
        preserved = any(row["checkpoint_id"] == original_last_id for row in frames)
        return original, day11._text(new_state["messages"][-1]), preserved

    original, new, preserved = _run(scenario())
    assert "RAG" in original
    assert "MCP 协议" in new
    assert original != new
    assert preserved is True  # 分叉不是覆盖


# ---------- 03 SQLite 落盘：换连接 = 换进程 ----------


def test_sqlite_saver_survives_new_connection(tmp_path: Path) -> None:
    """两个独立的 SQLite 连接（≈ 两次进程）仍能读到同一个 thread 的历史。"""
    pytest.importorskip("langgraph.checkpoint.sqlite.aio")
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    db_file = tmp_path / "day11_test.db"
    cfg = day11._config("sqlite-user")

    async def scenario() -> dict[str, Any]:
        async with AsyncSqliteSaver.from_conn_string(str(db_file)) as saver:
            app = day11._build_chat_graph(ScriptedLLM([AIMessage(content="记住了 Python")]), saver)
            await app.ainvoke(
                {"messages": [HumanMessage(content="记住：我最喜欢的语言是 Python")], "turns": 0},
                cfg,
            )
        # 换一条全新连接：内存里的东西全没了，只剩文件
        async with AsyncSqliteSaver.from_conn_string(str(db_file)) as saver:
            app = day11._build_chat_graph(ScriptedLLM([AIMessage(content="是 Python")]), saver)
            return await app.ainvoke({"messages": [HumanMessage(content="我最喜欢什么语言？")]}, cfg)

    state = _run(scenario())
    assert state["turns"] == 2
    assert len(state["messages"]) == 4
    assert db_file.exists()
    # 文件里确实有存档表（不是空跑）
    with sqlite3.connect(db_file) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "checkpoints" in tables


# ---------- 概念表 ----------


def test_mapping_tables_cover_course_checklist() -> None:
    """静态概念表齐全：快照四要素 / 两套暂停机制 / 三档存档器 / 六个坑 / 时间旅行三步。"""
    data = _run(day11.mapping())

    assert [item["field"] for item in data["snapshot_fields"]] == ["values", "next", "tasks", "created_at"]
    assert len(data["pause_mechanisms"]) == 5
    assert len(data["savers"]) == 3
    assert len(data["pitfalls"]) == 6
    assert len(data["time_travel_steps"]) == 3
    assert data["langgraph_version"] != "未安装"
    assert data["checkpoint_sqlite_version"] != "未安装"
