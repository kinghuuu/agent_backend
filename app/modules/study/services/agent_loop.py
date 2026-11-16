"""Function Calling 版 Agent 主循环（Day 01 骨架 + Day 03/04/05 增强）。

与 scratch/day01/agent.py 的差别只有两点：
1. 不用 print，而是把每一轮的"决策 → 工具调用 → 观察结果"记录成结构化轨迹；
2. 模型配置来自数据库（模型管理），而不是 .env。

Day 03：统一走 ``tools.dispatch``，轨迹带耗时 / error_type / self_heal 标记。
Day 04：结构化错误契约 + 自愈上限 + ``user_confirmed`` 确认放行。
Day 05：四层终止（语义 → 结构 → 资源 → 结果校验）+ 强制收敛降级交付。

终止四层在本文件里的落点：
- 第 1 层语义终止：``if not msg.tool_calls`` → 直接交付（入口 ④ 之前先做校验）；
- 第 2 层结构终止：``LoopGuard.check`` —— 动作指纹抓硬自旋，结果指纹抓变体自旋；
- 第 3 层资源熔断：``Budget``（步数 / token / 时延，按任务档位）+ 自愈上限；
- 第 4 层结果校验：``looks_grounded`` —— 答案没引用工具事实就回炉一次。

熔断不是"报错"，而是 ``_force_final``：``tools=None`` 逼模型用手上信息交卷，
产出明确标注"未能确认"的降级回答——60% 的答案远好于 0%。
"""

import asyncio
import json
import time
from dataclasses import replace
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import AgentRunRequest
from app.modules.study.services import termination, tools
from app.modules.study.services.llm import build_client, resolve_model
from app.modules.study.services.termination import LoopGuard, StopReason

# Day 02 四段式 System Prompt；Day 04 补纠错与确认规则；Day 05 补"该停"的规则。
# 注意：这里只放"行为偏好"，真正的终止/确认约束都在代码里——prompt 是建议，代码是法律
SYSTEM_PROMPT = """## 角色
你是一个可以使用工具的助手，能查天气/时间/新闻、做计算，也能查订单、导出数据和处理退款。

## 规则
1. 需要外部事实（天气、时间、新闻、订单）或精确计算时，必须调用工具，绝不允许凭记忆编造订单号、金额、物流状态。
2. 参数必须严格符合工具说明；格式不确定时先向用户确认，不要猜。
3. 工具返回 ok=false 时，读 hint 和 errors 修正参数再调用；同一个错误不要重复超过一次。
4. 工具返回 total=0 且 hint 说明"确实没有/不会变多"时，这就是最终事实：如实告知用户，不要换查询条件反复重试。
5. 同一工具、同一参数不要重复调用；轮询类工具（如 get_task_status）可以重复，直到状态变为 done。
6. 信息足够后直接给出答案，不要为了"更完整"再多查一轮。
7. 退款等资金操作必须先复述订单号与金额请用户确认；页面勾选"已获用户确认"后才允许实际提交。

## 输出格式
先给结论再给关键细节，并说明依据；信息不足时明确写"未能确认：…"，不要编造。

## 示例
用户：北京现在多少度？
助手：（调用 get_weather 后）北京现在 18℃，晴，西北风 3 级，空气质量良。
"""

# 强制收敛提示词：熔断后只用已有信息交卷（真正的硬约束是 tools=None）
FORCE_FINAL_PROMPT = (
    "已达执行上限（{reason}）。现在不允许再调用任何工具，"
    "请只用上面已有的信息回答用户。"
    "信息不足的部分明确写\"未能确认：…\"，不要编造。"
)

# 第 4 层打回提示词：点名它该引用的实体（订单号/日期等）
GROUNDING_RETRY_PROMPT = (
    "你的回答没有引用上面任何工具返回的事实（例如：{entities}）。"
    "请基于已有工具结果重写答案；仍无法确认的部分明确写\"未能确认：…\"，不要编造。"
)

# 练习 2：任务复杂度分类的轻量提示词（一行搞定，一次调用几 token）
CLASSIFY_PROMPT = (
    "判断用户任务的复杂度，只输出一个词："
    "只需一次工具调用或直接回答的输出 simple；需要多工具、多步推理的输出 complex。"
)

# 自愈上限：同一工具同一类错误连续失败超过该次数就熔断（Day 04 起）
MAX_SAME_ERROR = 2

# 轮询退避上限：课程是 8s，学习页是同步 HTTP 请求，收紧到 4s 避免页面长时间转圈
MAX_BACKOFF_SECONDS = 4.0


async def _classify_task(client: Any, model: str, question: str) -> str:
    """按问题判断任务档位（练习 2）：simple / complex，失败时保守取 simple。"""
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=4,
        )
        text = (resp.choices[0].message.content or "").lower()
        return "complex" if "complex" in text else "simple"
    except Exception:  # noqa: BLE001  分类失败不该拖垮主流程
        return "simple"


def _usage_tokens(resp: Any) -> int:
    """累计 token 用量；兼容不返回 usage 的服务（缺失按 0 处理）。"""
    usage = getattr(resp, "usage", None)
    return int(getattr(usage, "total_tokens", 0) or 0)


async def run(db: AsyncSession, payload: AgentRunRequest) -> dict[str, Any]:
    """运行一次 Agent 会话：四层终止 + 强制收敛，返回完整执行轨迹。"""
    cfg = await resolve_model(db, payload.model_config_id)
    client = build_client(cfg)

    # ---------- 第 3 层参数：任务档位（练习 2） ----------
    profile = payload.budget_profile
    if profile == "auto":
        profile = await _classify_task(client, cfg.model_name, payload.question)
    budget = termination.TASK_BUDGETS[profile]
    if payload.max_steps:
        # 显式步数是硬覆盖，方便课堂实验（如设 1 观察"循环被砍掉"）
        budget = replace(budget, max_steps=payload.max_steps)

    # 第 2 层：结构终止守卫；轮询白名单从工具元数据里取，避免两处维护
    guard = LoopGuard(repeatable=set(tools.repeatable_tool_names()))

    t0 = time.monotonic()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": payload.question},
    ]
    steps: list[dict[str, Any]] = []
    results_all: list[dict[str, Any]] = []   # 第 4 层校验用的全部工具结果
    tokens = 0
    error_tools: set[str] = set()            # Day 03 self-heal 观测
    error_streak: dict[tuple[str, str], int] = {}
    grounding_retried = False                # 第 4 层只打回一次

    def build_result(
        reason: StopReason,
        answer: str,
        *,
        degraded: bool = False,
        escalated: bool = False,
        grounded: bool = True,
        grounding_note: str | None = None,
    ) -> dict[str, Any]:
        """统一的返回结构：停止原因 + 轨迹 + 成本 + 质检结论。"""
        return {
            "steps": steps,
            "final_answer": answer,
            "stop_reason": reason.value,
            "stopped_by_limit": reason is StopReason.MAX_STEPS,
            "degraded": degraded,
            "escalated": escalated,
            "grounded": grounded,
            "grounding_note": grounding_note,
            "guard_diagnosis": guard.diagnose(),
            "budget_profile": profile,
            "budget_desc": budget.describe(),
            "tokens": tokens,
            "elapsed_ms": round((time.monotonic() - t0) * 1000),
            "model": cfg.model_name,
        }

    for step in range(1, budget.max_steps + 1):
        # ---------- 让模型决策 ----------
        resp = await client.chat.completions.create(
            model=cfg.model_name,
            messages=messages,
            tools=tools.TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0,
        )
        tokens += _usage_tokens(resp)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # ---------- 第 1 层：语义终止 + 第 4 层结果校验 ----------
        if not msg.tool_calls:
            content = msg.content or ""
            grounded = termination.looks_grounded(content, results_all)
            note: str | None = None
            if not grounded and not grounding_retried:
                # 假收敛：答案没引用任何工具事实 → 打回一次，用 tools=None 强迫基于事实重写
                grounding_retried = True
                entities = "、".join(termination.extract_entities(results_all)[:3])
                messages.append(
                    {"role": "user", "content": GROUNDING_RETRY_PROMPT.format(entities=entities)}
                )
                retry = await client.chat.completions.create(
                    model=cfg.model_name, messages=messages, tools=None, temperature=0
                )
                tokens += _usage_tokens(retry)
                retry_msg = retry.choices[0].message
                messages.append(retry_msg.model_dump(exclude_none=True))
                content = retry_msg.content or content
                grounded = termination.looks_grounded(content, results_all)
                note = "答案未引用工具结果，已打回重写一次"

            steps.append(
                {"step": step, "tool_calls": [], "content": content, "progress_fp": None}
            )
            return build_result(
                StopReason.COMPLETED if grounded else StopReason.UNGROUNDED,
                content,
                grounded=grounded,
                grounding_note=note,
            )

        # ---------- 执行工具（并行 tool_calls 一轮全跑完） ----------
        step_calls: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        results: list[dict[str, Any]] = []
        for call in msg.tool_calls:
            name = call.function.name
            raw_args = call.function.arguments or "{}"
            self_heal = name in error_tools

            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            safe_args = args if isinstance(args, dict) else {}

            # 轮询白名单：重复调用不算"重复动作"，改为指数退避后再查
            if name in guard.repeatable:
                await asyncio.sleep(min(guard.backoff_seconds(name), MAX_BACKOFF_SECONDS))

            started = time.perf_counter()
            result = tools.dispatch(name, raw_args, user_confirmed=payload.user_confirmed)
            duration_ms = round((time.perf_counter() - started) * 1000)

            is_error = False
            error_type: str | None = None
            try:
                body = json.loads(result)
            except json.JSONDecodeError:
                body = {"ok": False, "error_type": "bad_response"}
            if body.get("ok") is False:
                is_error = True
                error_type = body.get("error_type")

            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})
            step_calls.append(
                {
                    "name": name,
                    "arguments": safe_args,
                    "result": str(result),
                    "duration_ms": duration_ms,
                    "is_error": is_error,
                    "error_type": error_type,
                    "self_heal": self_heal,
                }
            )
            actions.append(
                {
                    "name": name,
                    "args": safe_args,
                    "fp": termination.action_fp(name, safe_args),
                    "ok": not is_error,
                }
            )
            results.append(body)
            results_all.append(body)

            if is_error:
                error_tools.add(name)
                streak_key = (name, error_type or "unknown")
                error_streak[streak_key] = error_streak.get(streak_key, 0) + 1
                if error_streak[streak_key] >= MAX_SAME_ERROR:
                    # 自愈无望：不再硬失败，改为强制收敛给一个降级答案
                    steps.append(
                        {"step": step, "tool_calls": step_calls, "content": None, "progress_fp": None}
                    )
                    return await _force_final(
                        client, cfg, messages, StopReason.REPEATED_ACTION, steps,
                        tokens, t0, build_result, escalated=True,
                    )
            else:
                error_tools.discard(name)

        # ---------- 第 2 层：结构终止 ----------
        step_fp = termination.progress_fp(results)
        steps.append(
            {"step": step, "tool_calls": step_calls, "content": None, "progress_fp": step_fp}
        )
        reason = guard.check(actions, results)
        if reason is not None:
            return await _force_final(
                client, cfg, messages, reason, steps, tokens, t0, build_result
            )

        # ---------- 第 3 层：资源熔断 ----------
        elapsed = time.monotonic() - t0
        if tokens > budget.max_tokens or elapsed > budget.max_seconds:
            return await _force_final(
                client, cfg, messages, StopReason.BUDGET, steps, tokens, t0, build_result
            )

    # 跑满步数还没给出最终回答
    return await _force_final(
        client, cfg, messages, StopReason.MAX_STEPS, steps, tokens, t0, build_result
    )


async def _force_final(
    client: Any,
    cfg: Any,
    messages: list[dict[str, Any]],
    reason: StopReason,
    steps: list[dict[str, Any]],
    tokens: int,
    t0: float,
    build_result: Any,
    *,
    escalated: bool = False,
) -> dict[str, Any]:
    """强制收敛（降级交付）：tools=None 逼模型用已有信息交卷。

    ``tools=None`` 是硬约束，比在 prompt 里写"不要再调用工具"强一个数量级——
    熔断后直接把半成品丢掉，用户看到的是半句话或 500。
    """
    final_messages = messages + [
        {"role": "user", "content": FORCE_FINAL_PROMPT.format(reason=reason.value)}
    ]
    resp = await client.chat.completions.create(
        model=cfg.model_name, messages=final_messages, tools=None, temperature=0
    )
    # build_result 闭包读的是熔断前的 token/耗时，这次收敛调用要补上
    total_tokens = tokens + _usage_tokens(resp)
    result = build_result(
        reason, resp.choices[0].message.content or "", degraded=True, escalated=escalated
    )
    result["tokens"] = total_tokens
    result["elapsed_ms"] = round((time.monotonic() - t0) * 1000)
    return result
