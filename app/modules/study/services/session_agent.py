"""多轮会话（Day 06）：用一个 SessionState 包住 Day 05 的单轮 Agent 循环。

调用链：
    chat() → 抽约束 → 计划淘汰 → 异步增量摘要 → render() 投影 messages
          → agent_loop.run(initial_messages=...) → commit_turn() 回写 state

会话状态存在进程内存（模块级注册表），重启即清——这正是 Day 07 长期记忆要解决的边界。
messages 不再是"存档"，只是每轮临时渲染出来的视图；state 才是真相源。
"""

import json
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import SessionChatRequest
from app.modules.study.services import agent_loop, memory, tools
from app.modules.study.services.llm import build_client, resolve_model
from app.modules.study.services.memory import SessionState

# 增量摘要提示词（Day 06 1.5）：只喂"旧摘要 + 被裁轮次"，不许全量重算
SUMMARY_PROMPT = """你在维护客服会话的滚动摘要，供后续轮次使用。

硬性要求：
1. 只保留会影响后续决策的信息：用户诉求、已确认的事实、已排除的可能、用户的硬约束。
2. 订单号、金额、日期、人名等具体值必须逐字保留，不许改写、不许四舍五入。
3. 不记录寒暄、不记录助手措辞、不复制工具返回的原始 JSON。
4. 总长度不超过 200 字。超了优先删过程、保留结论。
5. 只输出摘要正文，不要任何解释。

已有摘要：
{prev_summary}

需要并入的新对话：
{new_turns}
"""


class SessionAgent:
    """持有一份 SessionState 的多轮 Agent（会话内隔离，不共享 messages）。"""

    def __init__(self, state: SessionState) -> None:
        self.state = state
        self.budget = memory.Budget()
        # 工具 Schema 每轮重发，必须计入预算
        self.tools_tokens = memory.estimate_tokens(
            json.dumps(tools.TOOL_SCHEMAS, ensure_ascii=False)
        )

    async def chat(self, db: AsyncSession, payload: SessionChatRequest) -> dict[str, Any]:
        """处理一轮对话：投影上下文 → 跑 Agent → 回写记忆 → 返回会话态。"""
        cfg = await resolve_model(db, payload.model_config_id)
        client = build_client(cfg)

        # ① 用户的硬约束抽出来单独立法（③ 区，永不裁剪，每轮原样重放）
        for constraint in memory.pin_constraints(payload.message):
            if constraint not in self.state.constraints:
                self.state.constraints.append(constraint)

        # ② 淘汰计划 + 异步增量摘要（只喂旧摘要 + 被裁掉的轮次）
        plan = memory.plan_eviction(self.state, self.budget, self.tools_tokens)
        if plan.messages:
            self.state.summary = await self._summarize(
                client, cfg.model_name, self.state.summary, plan.messages
            )
            self.state.dropped.append(
                {
                    "turn": self.state.turn + 1,
                    "evicted_msgs": len(plan.messages),
                    "summary_len": len(self.state.summary),
                }
            )
            # 摘要质检：漏掉的实体补进锚点区（叙事归摘要，事实归锚点）
            memory.apply_summary_qa(self.state, plan.messages)

        # ③ 渲染本轮上下文（state → messages 的视图投影）
        messages = memory.render(self.state, self.budget, self.tools_tokens)
        ratio = memory.count_messages(messages, self.tools_tokens) / self.budget.total
        memory.record_water(self.state, ratio, "render")

        # ④ 复用 Day 05 的完整循环（四层终止 + 强制收敛）
        run_result = await agent_loop.run(
            db,
            agent_loop.AgentRunRequest(
                question=payload.message,
                budget_profile=payload.budget_profile,
                max_steps=payload.max_steps,
                model_config_id=payload.model_config_id,
                user_confirmed=payload.user_confirmed,
            ),
            initial_messages=messages,
        )

        # ⑤ 回写 state：轨迹消息进近轮原文，关键实体进 facts 槽位
        trace_messages = run_result.get("session_messages", [])
        memory.commit_turn(self.state, payload.message, trace_messages, run_result["final_answer"])
        self.state.total_tokens += run_result.get("tokens", 0)
        self._update_facts(trace_messages)

        return {
            "session_id": self.state.session_id,
            "turn": self.state.turn,
            "context_tokens": memory.count_messages(messages, self.tools_tokens),
            "water_level": memory.water_level(ratio),
            "constraints": list(self.state.constraints),
            "facts": dict(self.state.facts),
            "anchors": list(self.state.anchors),
            "summary": self.state.summary,
            "recent_rounds": len(self.state.recent),
            "dropped": list(self.state.dropped),
            "water_events": list(self.state.water_events),
            "run": run_result,
        }

    async def _summarize(
        self, client: Any, model: str, prev: str, dropped: list[dict[str, Any]]
    ) -> str:
        """增量摘要：只喂旧摘要 + 被裁轮次，避免 O(n²) 与措辞漂移。"""
        text = "\n".join(
            f"{m['role']}: {m.get('content') or ''}" for m in dropped
        )
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {
                    "role": "user",
                    "content": f"已有摘要：\n{prev or '（无）'}\n\n需要并入的新对话：\n{text}",
                },
            ],
            temperature=0,
            tools=None,
        )
        return (resp.choices[0].message.content or "").strip()[:400]

    def _update_facts(self, trace_messages: list[dict[str, Any]]) -> None:
        """从工具结果里捞关键实体进 facts——事实不靠摘要，靠结构化槽位。"""
        for message in trace_messages:
            if message.get("role") != "tool":
                continue
            try:
                body = json.loads(message.get("content") or "{}")
            except json.JSONDecodeError:
                continue
            data = body.get("data") if isinstance(body, dict) else None
            if not isinstance(data, dict):
                continue
            for order in (data.get("orders") or [])[:1]:
                if order.get("order_id"):
                    self.state.facts["last_order_id"] = order["order_id"]
                    self.state.facts["last_order_amount"] = order.get("amount")
                    self.state.facts["last_order_status"] = order.get("status")
            for refund in (data.get("refunds") or [])[:1]:
                if refund.get("refund_id"):
                    self.state.facts["last_refund_id"] = refund["refund_id"]
            if data.get("task_id"):
                self.state.facts["task_id"] = data["task_id"]

    def snapshot(self) -> dict[str, Any]:
        """会话状态快照（接口展示 / 调试用，不落盘）。"""
        return self.state.to_dict()


class SessionManager:
    """内存会话注册表：session_id → SessionAgent（上限 50，插入序淘汰）。"""

    MAX_SESSIONS = 50

    def __init__(self) -> None:
        self._sessions: dict[str, SessionAgent] = {}

    def get_or_create(self, session_id: str | None, system: str) -> tuple[str, SessionAgent]:
        """取回或新建会话；不传 id 时自动生成（返回值里可拿到 id 继续追问）。"""
        if session_id and session_id in self._sessions:
            return session_id, self._sessions[session_id]
        sid = session_id or uuid4().hex
        self._sessions[sid] = SessionAgent(SessionState(sid, system))
        while len(self._sessions) > self.MAX_SESSIONS:
            del self._sessions[next(iter(self._sessions))]
        return sid, self._sessions[sid]

    def get(self, session_id: str) -> SessionAgent | None:
        """按 id 查会话（不存在返回 None）。"""
        return self._sessions.get(session_id)

    def reset(self, session_id: str) -> bool:
        """新建一份同 id 的会话（换 ID 即清空记忆的后端等价物）。"""
        if session_id not in self._sessions:
            return False
        system = self._sessions[session_id].state.system
        self._sessions[session_id] = SessionAgent(SessionState(session_id, system))
        return True


# 模块级单例：与 react_loop 的会话表同思路，进程重启即清空
session_manager = SessionManager()
