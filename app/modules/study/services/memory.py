"""短期记忆（Day 06）：把 messages 从"存储"降级为"视图"。

核心概念：
- ``Round``：不可分割的裁剪单元（一条 user + 它后面的全部 assistant/tool 消息）；
- ``SessionState``：真相源，可序列化、可单测（约束锚点 / 事实槽位 / 摘要 / 近轮原文）；
- ``render``：每次调用模型前临时投影出 messages——它决定"这一轮给模型看什么"。

设计原则：所有"要不要裁"的判断都是纯函数、零 I/O；摘要生成由外部注入。
七分区配额见 ``Budget``；①③⑦ 永不可裁，超配额按 ⑥→⑤→④ 依次压缩。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from app.modules.study.services.termination import extract_entities


# ============================================================
# token 估算（无依赖版；生产请按模型名换 tiktoken）
# ============================================================


def estimate_tokens(text: str) -> int:
    """粗估：中文约 1 token/字，其他约 0.28 token/字符，外加少量结构开销。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return int(cjk * 1.0 + (len(text) - cjk) * 0.28) + 4


def count_messages(messages: list[dict[str, Any]], tools_tokens: int = 0) -> int:
    """含工具 Schema 的真实上下文占用——Schema 每轮都要重发，必须算进去。"""
    total = tools_tokens
    for message in messages:
        total += estimate_tokens(json.dumps(message, ensure_ascii=False, default=str))
    return total


# ============================================================
# 预算与数据结构
# ============================================================


@dataclass
class Budget:
    """课程 1.2 上下文预算表的代码版（按 8k 有效窗口设计）。"""

    system: int = 800
    tools: int = 1200
    constraints: int = 300
    summary: int = 400
    recent: int = 3000
    tool_results: int = 2000
    output_reserve: int = 1500  # ⑦ 必须先扣：忘了输出预留，答案会被截断

    @property
    def total(self) -> int:
        return (
            self.system + self.tools + self.constraints + self.summary
            + self.recent + self.tool_results + self.output_reserve
        )


@dataclass
class Round:
    """一轮对话 = 一条 user + 其后所有 assistant/tool 消息，裁剪时不可分割。"""

    user: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0

    def all_messages(self) -> list[dict[str, Any]]:
        """本轮的完整消息序列（user 在最前）。"""
        return [{"role": "user", "content": self.user}] + self.messages


@dataclass
class SessionState:
    """会话真相源：可序列化、可测试、可分支（复制一份试两种策略）。"""

    session_id: str
    system: str
    constraints: list[str] = field(default_factory=list)  # ③ 永不裁剪
    facts: dict[str, Any] = field(default_factory=dict)    # 结构化槽位（事实不靠摘要）
    anchors: list[str] = field(default_factory=list)       # 摘要质检补回的硬事实
    summary: str = ""                                       # ④ 滚动摘要
    recent: list[Round] = field(default_factory=list)       # ⑤ 近轮原文
    turn: int = 0
    total_tokens: int = 0
    dropped: list[dict[str, Any]] = field(default_factory=list)
    water_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """快照（供接口展示；学习项目不落盘）。"""
        return {
            "session_id": self.session_id,
            "system": self.system,
            "constraints": self.constraints,
            "facts": self.facts,
            "anchors": self.anchors,
            "summary": self.summary,
            "turn": self.turn,
            "total_tokens": self.total_tokens,
            "recent_rounds": len(self.recent),
            "dropped": self.dropped,
            "water_events": self.water_events,
        }


# ============================================================
# 约束抽取（③ 号分区：每轮原样重放，不进摘要、不进窗口）
# ============================================================

_CONSTRAINT_CUES = ("只能", "必须", "不要", "别", "禁止", "不允许", "务必", "记住")


def pin_constraints(text: str, limit: int = 6) -> list[str]:
    """从用户输入里抽出需要永久保留的短句（真实项目建议用 LLM 抽取）。"""
    out: list[str] = []
    for sentence in re.split(r"[。！？\n；]", text):
        sentence = sentence.strip()
        if sentence and any(cue in sentence for cue in _CONSTRAINT_CUES):
            out.append(sentence[:80])  # 约束要短，超长截断
    return out[:limit]


# ============================================================
# 水位线（练习 2）：60% 记日志 / 80% 摘要 / 95% 强制砍轮
# ============================================================

WATER_LEVELS = ((0.60, "watch"), (0.80, "summarize"), (0.95, "force_trim"))


def water_level(ratio: float) -> str:
    """按上下文占用比例返回当前水位档位。"""
    level = "normal"
    for threshold, name in WATER_LEVELS:
        if ratio >= threshold:
            level = name
    return level


def record_water(state: SessionState, ratio: float, action: str) -> None:
    """结构化记录一次水位事件（练习 2 的日志点，可替换成真正的 logger）。"""
    state.water_events.append(
        {
            "turn": state.turn,
            "ratio": round(ratio, 3),
            "level": water_level(ratio),
            "action": action,
        }
    )


# ============================================================
# 纯逻辑：淘汰计划 / 渲染 / 摘要质检 / 回写
# ============================================================


@dataclass
class EvictionPlan:
    """淘汰计划：被挤出的消息（喂摘要用）+ 保留的轮次下标（更新 state 用）。"""

    messages: list[dict[str, Any]] = field(default_factory=list)
    kept_indices: set[int] = field(default_factory=set)


def plan_eviction(
    state: SessionState, budget: Budget, tools_tokens: int = 0
) -> EvictionPlan:
    """算出哪些轮次要被挤出（纯函数，不修改 state）。

    从最新往回装，装不下就停——因此被挤出的永远是更早的轮次。
    """
    room = budget.recent + budget.tool_results
    kept: set[int] = set()
    for index in range(len(state.recent) - 1, -1, -1):
        cost = count_messages(state.recent[index].all_messages())
        if cost > room and kept:
            break
        kept.add(index)
        room -= cost

    evicted: list[dict[str, Any]] = []
    for index, round_ in enumerate(state.recent):
        if index not in kept:
            evicted.extend(round_.all_messages())
    return EvictionPlan(messages=evicted, kept_indices=kept)


def _head(state: SessionState) -> str:
    """不可裁剪的头部：system + ③ 约束锚点 + 事实槽位 + 摘要质检补回的硬事实。"""
    head = state.system
    if state.constraints:
        head += "\n\n【用户硬约束，必须遵守】\n" + "\n".join(f"- {c}" for c in state.constraints)
    if state.facts:
        head += "\n\n【已确认事实】\n" + "\n".join(f"- {k}: {v}" for k, v in state.facts.items())
    if state.anchors:
        head += "\n\n【早期对话关键事实（摘要质检补回）】\n" + "\n".join(
            f"- {a}" for a in state.anchors
        )
    return head


def render(
    state: SessionState,
    budget: Budget,
    tools_tokens: int = 0,
    summarize: Callable[[str, list[dict[str, Any]]], str] | None = None,
) -> list[dict[str, Any]]:
    """把 state 投影成本轮要发给模型的 messages（所有降级都发生在这里）。

    ``summarize`` 可选且同步：传入时用它做增量摘要；多轮会话服务会先异步
    生成好摘要再调本函数（决策逻辑保持纯函数，I/O 不进来）。
    """
    state.turn += 1

    plan = plan_eviction(state, budget, tools_tokens)
    evicted = plan.messages
    if evicted:
        if summarize is not None:
            state.summary = summarize(state.summary, evicted)
        state.dropped.append(
            {"turn": state.turn, "evicted_msgs": len(evicted), "summary_len": len(state.summary)}
        )
        if summarize is not None:
            apply_summary_qa(state, evicted)

    # ⑤ 只保留装得下的轮次（被挤出的已进入摘要 / 锚点）
    state.recent = [
        round_ for index, round_ in enumerate(state.recent)
        if index in plan.kept_indices
    ]

    # ⑥ 旧工具结果降级为 digest：只换 content，不动 role/tool_call_id，协议依然合法
    for round_ in state.recent[:-1]:
        for message in round_.messages:
            if message["role"] == "tool" and len(str(message.get("content", ""))) > 120:
                message["content"] = digest_of(str(message["content"]))

    def build_messages() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": _head(state)}]
        if state.summary:
            out.append({"role": "system", "content": f"【早期对话摘要】\n{state.summary}"})
        for round_ in state.recent:
            out.extend(round_.all_messages())
        return out

    messages = build_messages()

    # ⑦ 预算体检：超限就继续砍最老的轮（约束 / 摘要 / 锚点永不动）
    while count_messages(messages, tools_tokens) + budget.output_reserve > budget.total:
        if len(state.recent) <= 1:
            break  # 只剩最后一轮，认了
        state.recent.pop(0)
        messages = build_messages()

    # 断言守住"宪法"：system 必须在首位，且约束原样在位
    assert messages[0]["role"] == "system"
    for constraint in state.constraints:
        assert constraint in messages[0]["content"]
    return messages


def summary_kept_entities(summary: str, dropped: list[dict[str, Any]]) -> list[str]:
    """摘要把哪些硬实体弄丢了（订单号/金额/日期等）——丢了就补锚点，别信摘要。"""
    entities = extract_entities(dropped)
    return [entity for entity in entities if entity not in summary]


def apply_summary_qa(state: SessionState, dropped: list[dict[str, Any]]) -> None:
    """摘要质检：漏掉的实体以"fact: …"形式硬塞进锚点区（叙事归摘要，事实归锚点）。"""
    for entity in summary_kept_entities(state.summary, dropped):
        anchor = f"fact: {entity} 出现在早期对话"
        if anchor not in state.anchors:
            state.anchors.append(anchor)


def digest_of(raw: str) -> str:
    """把一坨 JSON 压成一句结论（真实项目可以交给小模型）。"""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw[:120]
    if not isinstance(data, dict):
        return raw[:120]
    if data.get("ok") is False:
        return f"调用失败：{data.get('hint', data.get('error_type', '未知'))}"
    kept = {k: v for k, v in data.items() if k in ("total", "returned", "hint", "status", "task_id", "amount")}
    return json.dumps(kept, ensure_ascii=False)[:120] if kept else raw[:120]


def commit_turn(
    state: SessionState, user_input: str, trace_messages: list[dict[str, Any]], answer: str
) -> None:
    """把一轮执行写回 state（trace 来自 Day 05 的 assistant/tool 消息）。

    末尾若已是最终 assistant（无 tool_calls）就直接复用，**不重建**——重建会丢
    thinking 模型的 ``reasoning_content``，下一轮回放会被上游 400。
    """
    messages = list(trace_messages)
    last_is_final = (
        messages
        and messages[-1].get("role") == "assistant"
        and not messages[-1].get("tool_calls")
    )
    if not last_is_final:
        messages.append({"role": "assistant", "content": answer})
    state.recent.append(Round(user_input, messages, count_messages(messages)))
