"""Day 12 单测：多智能体 Supervisor（主管派活 / 专家摘要回传 / 预算 / 去重 / 审批 / 并行），不联网。

运行：``python -m pytest tests/test_day12_supervisor.py -p no:cacheprovider -q``
没装 langgraph 时整文件跳过。

四个原则：
1. 用脚本化假模型顶替真实 LLM——主管/专家的**控制流**与模型质量无关；
2. 异步用例统一 ``asyncio.run`` 包一层，不依赖 pytest-asyncio 配置；
3. 主管的决策函数可注入，直接验证"派单 / 预算 / 去重"三条逻辑；
4. 每个用例都能单独回答课程里的一个问题（主管会自己回答吗？专家带 name 吗？预算真的兜底吗？）。
"""

import asyncio
from typing import Any

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402
from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.modules.study.services import day12_supervisor as day12  # noqa: E402


def _run(coro: Any) -> Any:
    """在同步用例里跑协程（避免依赖 pytest-asyncio 的 mode 配置）。"""
    return asyncio.run(coro)


class ScriptedLLM:
    """按脚本依次吐消息的假模型，同时支持 ``bind_tools`` 与 ``with_structured_output``。

    - ``answers``：专家子图/单 Agent 循环按顺序取用的正文；
    - ``routes``：主管结构化输出按顺序取用的 (next, task) 二元组；
    - ``method``：记录主管走过的下发方式（用于断言降级链没被短路）。
    """

    def __init__(
        self,
        *,
        answers: list[str] | None = None,
        routes: list[tuple[str, str]] | None = None,
    ) -> None:
        self.answers = list(answers or [])
        self.routes = list(routes or [])
        self.methods: list[str] = []

    def bind_tools(self, tools: list[Any]) -> "ScriptedLLM":
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        return AIMessage(content=self.answers.pop(0) if self.answers else "（脚本已用完）")

    def with_structured_output(self, schema: type, method: str | None = None) -> Any:
        llm = self
        chosen = method or "default"

        class _Runner:
            async def ainvoke(self, messages: Any) -> Any:
                if not llm.routes:
                    raise RuntimeError("没有更多路由脚本")
                llm.methods.append(chosen)
                nxt, task = llm.routes.pop(0)
                return schema(next=nxt, task=task, reason="脚本")

        return _Runner()


# ---------- 主管的输出空间 ----------


def test_route_schema_has_no_content_field() -> None:
    """主管的 schema 只有 next/task/reason——没有 content，它想自己回答也没地方放（坑 4）。"""
    assert set(day12.Route.model_fields) == {"next", "task", "reason"}
    assert "content" not in day12.Route.model_fields


# ---------- 02 最小 Supervisor ----------


def _build_min_graph(
    *,
    routes: list[tuple[str, str]],
    answers: dict[str, list[str]],
    max_handoff: int = 6,
    dedupe: bool = False,
) -> tuple[Any, dict[str, int]]:
    """用注入的决策函数与假专家搭一张最小团队图（不经过 LLM，只测控制流）。"""
    exec_calls: dict[str, int] = {"researcher": 0, "writer": 0}

    async def decide(state: dict[str, Any]) -> dict[str, Any]:
        if not routes:
            return {"next": "FINISH", "task": "", "reason": "脚本结束", "method": "script"}
        nxt, task = routes.pop(0)
        return {"next": nxt, "task": task, "reason": "脚本", "method": "script"}

    async def researcher_run(messages: list[Any], assignment: str) -> str:
        exec_calls["researcher"] += 1
        queue = answers.get("researcher") or []
        return queue.pop(0) if queue else "调研结论"

    async def writer_run(messages: list[Any], assignment: str) -> str:
        exec_calls["writer"] += 1
        queue = answers.get("writer") or []
        return queue.pop(0) if queue else "对客答复"

    nodes = {
        "researcher": day12._expert_node("researcher", researcher_run),
        "writer": day12._expert_node("writer", writer_run, back_to="supervisor"),
    }
    app = day12._build_team_graph(
        decide=decide,
        nodes=nodes,
        roster=day12.ROSTER_MIN,
        max_handoff=max_handoff,
        dedupe=dedupe,
        finish_to_writer=False,
    )
    return app, exec_calls


def test_team_dispatches_and_experts_return_named_messages() -> None:
    """researcher → writer → FINISH：派单带 assignment，回传消息带 name，账本逐条累积。"""
    app, exec_calls = _build_min_graph(
        routes=[("researcher", "查订单与政策"), ("writer", "写答复"), ("FINISH", "")],
        answers={"researcher": ["订单 A1001 签收 3 天，政策允许 7 天内退。"], "writer": ["您好，可以退。"]},
    )
    state = _run(app.ainvoke(day12._initial_team_state("订单 A1001 能退吗？")))

    assert [row["role"] for row in state["dispatches"]] == ["researcher", "writer"]
    assert state["dispatches"][0]["task"] == "查订单与政策"
    # 专家回传的 AIMessage 必须带 name，否则主管分不清谁说了什么
    named = {message.name for message in state["messages"] if isinstance(message, AIMessage)}
    assert {"researcher", "writer"} <= named
    assert any(item.startswith("[researcher]") for item in state["findings"])
    assert any(item.startswith("[writer]") for item in state["findings"])
    assert state["steps"] == 2  # 只有主管派单计数
    assert exec_calls == {"researcher": 1, "writer": 1}
    assert day12._last_answer(state["messages"]) == "您好，可以退。"


def test_supervisor_never_answers_by_itself() -> None:
    """主管只产生事件、不产生 AI 消息：messages 里的每条 AI 都来自专家（带 name）。"""
    app, _ = _build_min_graph(
        routes=[("researcher", "查事实"), ("FINISH", "")],
        answers={"researcher": ["事实结论"], "writer": ["答复"]},
    )
    state = _run(app.ainvoke(day12._initial_team_state("随便问问")))

    ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
    assert [m.name for m in ai_messages] == ["researcher"]
    assert all(item["node"] != "supervisor" or item["kind"] != "expert" for item in state["events"])


def test_budget_guard_forces_writer_and_flags_budget_hit() -> None:
    """交接预算耗尽 → 不再派单，强制转 writer 降级交付（语义终止失灵的保险）。"""
    app, exec_calls = _build_min_graph(
        routes=[("researcher", "第一轮"), ("researcher", "第二轮"), ("researcher", "第三轮")],
        answers={"researcher": ["结论"], "writer": ["阶段性答复"]},
        max_handoff=2,
    )
    state = _run(app.ainvoke(day12._initial_team_state("能退吗")))

    assert state["steps"] == 2  # 预算到 2 就停
    assert any(item["kind"] == "budget" for item in state["events"])
    assert exec_calls["writer"] == 1  # 强制走 writer，而不是直接 END
    assert day12._last_answer(state["messages"]) == "阶段性答复"


def test_dedupe_guard_skips_repeated_dispatch() -> None:
    """练习 1：连续两次派给同一人且任务高度相似 → 跳过本次派单转 writer（乒乓球被截断）。"""
    task = "查订单 A1001 的签收天数与退款政策"
    app, exec_calls = _build_min_graph(
        routes=[("researcher", task), ("researcher", task), ("FINISH", "")],
        answers={"researcher": ["结论"], "writer": ["阶段答复"]},
        dedupe=True,
    )
    state = _run(app.ainvoke(day12._initial_team_state("能退吗")))

    assert exec_calls["researcher"] == 1  # 第二次相似派单被跳过
    assert any(item["kind"] == "dedupe" for item in state["events"])
    assert state["steps"] == 1  # 被跳过的派单不占预算
    assert day12._last_answer(state["messages"]) == "阶段答复"


def test_dedupe_guard_off_keeps_dispatching() -> None:
    """关掉守卫时同样的脚本会真的派两轮——证明上面的断言是守卫带来的差异。"""
    task = "查订单 A1001 的签收天数与退款政策"
    app, exec_calls = _build_min_graph(
        routes=[("researcher", task), ("researcher", task), ("FINISH", "")],
        answers={"researcher": ["一", "二"], "writer": ["答复"]},
        dedupe=False,
    )
    state = _run(app.ainvoke(day12._initial_team_state("能退吗")))
    assert exec_calls["researcher"] == 2
    assert not any(item["kind"] == "dedupe" for item in state["events"])


def test_expert_returns_command_and_only_summary() -> None:
    """专家节点：返回 Command(goto=supervisor)，且只把最后一段结论放进共享 messages。"""
    async def runner(messages: list[Any], assignment: str) -> str:
        return "只回传结论"

    node = day12._expert_node("researcher", runner)
    patch = _run(node({"messages": [], "assignment": "查事实", "branch": ""}))
    assert isinstance(patch, Command)
    assert patch.goto == "supervisor"
    assert patch.update["findings"] == ["[researcher] 只回传结论"]
    assert len(patch.update["messages"]) == 1
    assert patch.update["messages"][0].name == "researcher"


# ---------- 03 完整版：审批 + 预算 + 存档 ----------


def test_full_graph_pauses_at_interrupt_then_resumes_once() -> None:
    """完整版：coder 前挂 interrupt 冻结；resume 后 writer 成稿，退款流水只落一次。"""
    llm = ScriptedLLM(
        routes=[
            ("researcher", "查订单 A1001 与退款政策"),
            ("coder", "为订单 A1001 退款 299，并计算 299*0.85"),
            ("FINISH", ""),
        ],
        answers=["订单 A1001 签收 3 天，政策允许 7 天内退。", "您好，已为您办理退款 299 元。"],
    )
    saver = MemorySaver()
    counter: dict[str, int] = {}
    app = day12._build_full_app(llm, max_handoff=6, counter=counter, checkpointer=saver)
    cfg = {"configurable": {"thread_id": "team-test-1"}}
    day12.REFUND_LOG.clear()

    before = int(counter.get("expert_calls", 0))
    _run(app.ainvoke(day12._initial_team_state("订单 A1001 签收3天，能不能退？退 299，算 299*0.85", thread_id="team-test-1"), cfg))
    snap = _run(app.aget_state(cfg))

    assert tuple(snap.next) == ("coder",)  # 卡在审批节点
    assert day12.REFUND_LOG == []  # 暂停时一分钱没退
    payload = snap.tasks[0].interrupts[0].value
    assert payload["action"] == "refund"
    assert payload["args"]["order_id"] == "A1001"

    state = _run(app.ainvoke(Command(resume={"approved": True, "amount": 254}), cfg))
    assert tuple(_run(app.aget_state(cfg)).next) == ()
    # 退款只落一次（审批节点被重放，但它里面只有 interrupt）
    assert len(day12.REFUND_LOG) == 1
    assert day12.REFUND_LOG[0]["amount"] == 254  # 人工改过的金额生效
    assert [row["role"] for row in state["dispatches"]] == ["researcher", "coder"]
    assert day12._last_answer(state["messages"]) == "您好，已为您办理退款 299 元。"
    assert int(counter.get("expert_calls", 0)) > before


def test_full_graph_rejected_refund_never_hits_ledger() -> None:
    """驳回：coder 明确说未执行，退款流水为空，图仍跑到 writer 成稿。"""
    llm = ScriptedLLM(
        routes=[("coder", "为订单 A1001 退款 299"), ("FINISH", "")],
        answers=["您好，退款未获批准。"],
    )
    saver = MemorySaver()
    app = day12._build_full_app(llm, max_handoff=6, counter={}, checkpointer=saver)
    cfg = {"configurable": {"thread_id": "team-test-2"}}
    day12.REFUND_LOG.clear()

    _run(app.ainvoke(day12._initial_team_state("帮我退款 299", thread_id="team-test-2"), cfg))
    state = _run(app.ainvoke(Command(resume=False), cfg))

    assert day12.REFUND_LOG == []
    assert day12._last_answer(state["messages"]) == "您好，退款未获批准。"


def test_coder_node_rejects_unsafe_expression() -> None:
    """安全求值：白名单外的字符一律拒绝，四则运算正常。"""
    assert day12._safe_calc("299*0.85") == "254.15"
    assert "拒绝执行" in day12._safe_calc("__import__('os').system('ls')")
    assert day12._extract_expression("退款 299，顺便算 299*0.85") == "299*0.85"
    assert day12._extract_expression("只是查一下订单") == ""


def test_claim_detection_ignores_conditional_sentences() -> None:
    """假收敛判定逐句进行：『已为您退款』算声称完成，『回复确认后提交』不算。"""
    assert day12._claims_done("已为您退款 299 元，请留意到账。") is True
    assert day12._claims_done("回复确认后提交退款申请。") is False
    assert day12._claims_done("需要您确认后才能提交退款。") is False

    issues = day12._assess_issues(
        toolset="narrow",
        counts={"refund_order": 0},
        outputs=[],
        final_answer="已为您退款 299 元。",
    )
    assert any("假收敛" in item for item in issues)
    # 窄工具集不该被扣上"越权"的帽子
    assert not any("越权" in item for item in issues)


# ---------- 04 并行专家 ----------


def test_parallel_branches_merge_and_writer_called_once() -> None:
    """Send 扇出两条分支：findings 两条都在，writer 只调一次。"""
    llm = ScriptedLLM(answers=["政策：7 天内可退。", "订单 A1001 签收 3 天。", "您好，可以退款。"])
    counter: dict[str, int] = {}
    app = day12._build_parallel_app(llm, "A1001", counter)

    state = _run(app.ainvoke({"messages": [HumanMessage(content="能退吗")], "findings": [], "events": []}))

    findings = list(state["findings"])
    # 两条分支的结论都在（writer 自己也会写一条 findings，所以这里只断言前两条）
    assert any(item.startswith("[政策]") for item in findings)
    assert any(item.startswith("[订单]") for item in findings)
    assert len([item for item in findings if item.startswith("[政策]") or item.startswith("[订单]")]) == 2
    expert_events = [item for item in state["events"] if item["kind"] == "expert"]
    assert len([item for item in expert_events if item["node"] == "writer"]) == 1
    assert day12._last_answer(state["messages"]) == "您好，可以退款。"
    # 两条分支各有一句带 branch 署名的消息
    names = {m.name for m in state["messages"] if isinstance(m, AIMessage)}
    assert {"researcher:政策", "researcher:订单", "writer"} <= names


# ---------- 概念表 ----------


def test_mapping_tables_cover_course_checklist() -> None:
    """静态概念表齐全：三堵墙 / 三角色 / 三种交接 / 三层状态 / 三层终止 / 五个坑 / 六条选型。"""
    data = _run(day12.mapping())

    assert len(data["walls"]) == 3
    assert len(data["roles"]) == 3
    assert len(data["handoff_modes"]) == 3
    assert len(data["state_layers"]) == 3
    assert len(data["terminations"]) == 3
    assert len(data["pitfalls"]) == 5
    assert len(data["decision_table"]) == 6
    assert len(data["checklist"]) == 9
    assert data["langgraph_version"] != "未安装"
