"""短期记忆单测（Day 06 练习 1）：属性测试守住 tool_call 配对协议。

运行：``python -m pytest tests/test_memory.py -p no:cacheprovider -q``
"""

import json
import random

from app.modules.study.services import memory
from app.modules.study.services.memory import (
    Budget,
    Round,
    SessionState,
    digest_of,
    pin_constraints,
    summary_kept_entities,
    water_level,
)


def _make_round(index: int, with_tools: bool = True) -> Round:
    """构造一轮完整对话：assistant(tool_calls) + tool + assistant，配对合法。"""
    messages: list[dict] = []
    if with_tools:
        call_id = f"call_{index}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "search_orders", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    {
                        "ok": True,
                        "total": index,
                        "orders": [{"order_id": f"O202600{index:02d}", "amount": 99.0}],
                        # 拉长内容，确保触发"旧工具结果 >120 字符降级为 digest"的分支
                        "raw": "x" * 160,
                    },
                    ensure_ascii=False,
                ),
            }
        )
    messages.append({"role": "assistant", "content": f"这是第 {index} 轮的回答"})
    return Round(f"第 {index} 轮问题", messages)


def test_no_orphan_tool_messages_200_random_states() -> None:
    """练习 1：200 组随机 state 渲染后，不存在孤儿 tool 消息。

    裁剪单位必须是"轮"：只要切在 assistant(tool_calls) 与 tool 之间就会 400。
    """
    random.seed(20261116)
    for _ in range(200):
        state = SessionState("S", "你是客服助手")
        for index in range(random.randint(1, 12)):
            state.recent.append(_make_round(index, random.random() < 0.7))

        budget = Budget(
            recent=random.randint(80, 800),
            tool_results=random.randint(40, 500),
        )
        rendered = memory.render(state, budget, tools_tokens=50)

        assert rendered[0]["role"] == "system", "system 必须永远在首位"
        seen_call_ids: set[str] = set()
        for message in rendered:
            if message["role"] == "assistant":
                for call in message.get("tool_calls") or []:
                    seen_call_ids.add(call["id"])
            elif message["role"] == "tool":
                assert message["tool_call_id"] in seen_call_ids, (
                    f"发现孤儿 tool 消息：{message['tool_call_id']}"
                )


def test_constraints_survive_aggressive_trimming() -> None:
    """硬约束（③ 区）即使预算极紧也必须在 system 头部原样出现。"""
    state = SessionState("S", "你是客服助手", constraints=["只能走对公转账，别推荐在线支付"])
    for index in range(10):
        state.recent.append(_make_round(index))

    budget = Budget(system=200, recent=10, tool_results=10, summary=10)
    rendered = memory.render(state, budget, tools_tokens=0)
    assert "只能走对公转账，别推荐在线支付" in rendered[0]["content"]


def test_pin_constraints() -> None:
    """约束抽取：命中硬约束关键词的句子才进锚点区。"""
    hits = pin_constraints("我叫小明。我们公司只能走对公转账。今天天气不错。禁止推荐在线支付！")
    assert hits == ["我们公司只能走对公转账", "禁止推荐在线支付"]


def test_old_tool_results_downgraded_but_last_kept() -> None:
    """⑥ 区：旧工具结果降级为 digest，最近一轮保留原文。"""
    state = SessionState("S", "你是客服助手")
    state.recent.append(_make_round(0))
    state.recent.append(_make_round(1))
    memory.render(state, Budget(), tools_tokens=0)

    old = state.recent[0].messages[1]
    latest = state.recent[1].messages[1]
    assert len(old["content"]) <= 120 and "orders" not in old["content"]
    assert "orders" in latest["content"]  # 最近一轮原文不动
    assert old["tool_call_id"] == "call_0"  # 只换 content，协议结构不动


def test_summary_entity_qa() -> None:
    """摘要漂移质检：漏掉的订单号必须被检测出来（随后补进锚点区）。"""
    dropped = [{"role": "tool", "content": '{"ok": true, "order_id": "O20260012", "amount": 299}'}]
    assert summary_kept_entities("用户询问了订单状态", dropped) == ["O20260012"]
    assert summary_kept_entities("订单 O20260012 已发货", dropped) == []


def test_apply_summary_qa_adds_anchors() -> None:
    """质检发现漏实体时，把它硬追加到 anchors（事实不依赖摘要措辞）。"""
    state = SessionState("S", "你是客服助手")
    state.summary = "用户问了订单情况"
    memory.apply_summary_qa(state, [{"role": "tool", "content": '{"order_id": "O20260012"}'}])
    assert any("O20260012" in anchor for anchor in state.anchors)


def test_digest_of_ok_and_error() -> None:
    """digest：成功只留关键字段，失败留 hint（下次模型能改对）。"""
    assert len(digest_of(json.dumps({"ok": True, "data": {"total": 3, "orders": [{"x": 1}]}}))) < 80
    assert "请重新" in digest_of(
        json.dumps({"ok": False, "error_type": "invalid_args", "hint": "请重新输入"})
    )


def test_water_levels() -> None:
    """三档水位线：60% 观察 / 80% 摘要 / 95% 强制砍轮。"""
    assert water_level(0.5) == "normal"
    assert water_level(0.6) == "watch"
    assert water_level(0.8) == "summarize"
    assert water_level(0.96) == "force_trim"


def test_plan_eviction_keeps_newest() -> None:
    """淘汰计划：预算紧张时保留最新轮次，挤出最早轮次。"""
    state = SessionState("S", "你是客服助手")
    for index in range(6):
        state.recent.append(_make_round(index))
    plan = memory.plan_eviction(state, Budget(recent=200, tool_results=100))
    assert plan.kept_indices and max(plan.kept_indices) == 5
    assert all(index not in plan.kept_indices for index in range(0, min(plan.kept_indices)))
    assert plan.messages, "应该有被挤出的消息供摘要使用"


def test_ensure_final_message_keeps_reasoning_content() -> None:
    """回归：最终回答必须复用模型原始返回，不能重建丢 reasoning_content。

    thinking 模式模型（Qwen3 等）下一轮要求把上一轮 assistant 的
    reasoning_content 原样传回，丢了就 400 —— Day 06 多轮会话的真实故障。
    """
    from app.modules.study.services import agent_loop

    trace: list[dict] = []
    raw = {
        "role": "assistant",
        "content": "订单已发货",
        "reasoning_content": "先查订单再回答",
    }
    agent_loop.ensure_final_message(trace, "订单已发货", raw)
    assert trace[-1]["reasoning_content"] == "先查订单再回答"


def test_ensure_final_message_does_not_duplicate() -> None:
    """轨迹末尾已是最终 assistant 时，不重复追加（避免历史里出现两条同款回答）。"""
    from app.modules.study.services import agent_loop

    trace = [{"role": "assistant", "content": "订单已发货", "reasoning_content": "想一下"}]
    agent_loop.ensure_final_message(trace, "订单已发货")
    assert len(trace) == 1
    assert trace[0]["reasoning_content"] == "想一下"


def test_ensure_final_message_fallback_when_missing() -> None:
    """异常路径（轨迹里没有最终 assistant）才兜底补一条裸消息。"""
    from app.modules.study.services import agent_loop

    trace = [{"role": "tool", "tool_call_id": "c1", "content": "{}"}]
    agent_loop.ensure_final_message(trace, "已完成")
    assert trace[-1] == {"role": "assistant", "content": "已完成"}


def test_commit_turn_and_render_keep_reasoning_content() -> None:
    """回归：写入 state 再渲染，assistant 的 reasoning_content 必须原样保留。

    这是 Day 06 多轮会话 400 的完整链路复现——只测 ensure_final 不够，
    必须保证 commit_turn → render 全程不剥字段。
    """
    state = SessionState("S", "你是客服助手")
    trace = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "先查订单",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search_orders", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true, "total": 1}'},
        {"role": "assistant", "content": "已发货", "reasoning_content": "整理给用户"},
    ]
    memory.commit_turn(state, "订单到哪了", trace, "已发货")
    rendered = memory.render(state, Budget())

    assistants = [m for m in rendered if m.get("role") == "assistant"]
    assert assistants, "渲染结果里必须有 assistant 消息"
    assert all(m.get("reasoning_content") for m in assistants), (
        "所有 assistant 消息都必须保留 reasoning_content，否则下一轮回放会 400"
    )
