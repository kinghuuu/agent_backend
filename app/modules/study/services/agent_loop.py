"""Function Calling 版 Agent 主循环（Day 01 骨架 + Day 03/04 增强）。

与 scratch/day01/agent.py 的差别只有两点：
1. 不用 print，而是把每一轮的"决策 → 工具调用 → 观察结果"记录成结构化轨迹，
   交给前端可视化展示；
2. 模型配置来自数据库（模型管理），而不是 .env。

Day 03 增强：工具执行统一走 ``tools.dispatch``；轨迹附带耗时与错误标记；
上一轮报错的工具本轮被再次调用时标记 self_heal（供 Day 05 重试策略分析）。

Day 04 增强：
- 错误契约升级为 ``{ok, error_type, message, hint, errors}`` 结构化对象，
  轨迹里记录 error_type，前端可以区分"参数错了"还是"工具坏了"；
- 自愈上限（``MAX_SAME_ERROR``）：同一工具同一类错误连续 2 次未修复就熔断，
  直接给出升级说明——别让模型在死胡同里继续烧 token；
- ``user_confirmed``：前端勾选后放行退款等资金操作（与 dispatch 的确认门配合）。

循环里三个不能省的工程约束（踩过坑才懂）：
- 最大步数：防"调工具 → 不满意 → 再调"的死循环；
- 异常兜底：工具报错不抛给上层，转成文本回给模型让它自己想办法；
- assistant 回填：带 tool_calls 的消息必须原样进历史，否则下一轮 400。
"""

import json
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import AgentRunRequest
from app.modules.study.services import tools
from app.modules.study.services.llm import build_client, resolve_model

# Day 02 升级：System Prompt 按"角色/规则/输出格式/示例"四段式重写（课程 1.3 模板）。
# Day 04 补充：ok=false 的纠错策略与资金操作确认规则
SYSTEM_PROMPT = """## 角色
你是一个可以使用工具的助手，能查天气/时间/新闻、做计算，也能查订单和处理退款。

## 规则
1. 需要外部事实（天气、时间、新闻、订单）或精确计算时，必须调用工具，绝不允许凭记忆编造订单号、金额、物流状态。
2. 参数必须严格符合工具说明；格式不确定时先向用户确认，不要猜。
3. 工具返回 ok=false 时，读 hint 和 errors 修正参数再调用；同一个错误不要重复超过一次，第二次仍失败就向用户如实说明。
4. 退款等资金操作必须先复述订单号与金额请用户确认；页面勾选"已获用户确认"后才允许实际提交。
5. 不要向用户暴露工具名、JSON 参数等内部细节，用自然语言回答。

## 输出格式
信息足够时直接给出最终答案，先给结论再给关键细节，并说明你是依据什么得出的。

## 示例
用户：北京现在多少度？
助手：（调用 get_weather 工具后）北京现在 18℃，晴，西北风 3 级，空气质量良。
"""

# 熔断时给用户的升级说明（与课程 agent.py 的"升级给人"话术一致）
ESCALATION_MESSAGE = (
    "这个请求我处理不了：同一个工具连续校验失败，已中止自动重试。"
    "请核对订单号/用户 ID 等参数的准确格式后再试，我马上继续。"
)


async def run(db: AsyncSession, payload: AgentRunRequest) -> dict[str, Any]:
    """运行一次 Agent 会话，返回完整执行轨迹（结构见 AgentRunResult）。"""
    cfg = await resolve_model(db, payload.model_config_id)
    client = build_client(cfg)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": payload.question},
    ]
    steps: list[dict[str, Any]] = []
    # Day 03 练习 3：上一轮报过错的工具名集合——本轮再次被调用即视为自我纠正（self-heal）
    error_tools: set[str] = set()
    # Day 04：同一 (工具, 错误类型) 连续失败计数——达到 MAX_SAME_ERROR 就熔断升级
    error_streak: dict[tuple[str, str], int] = {}

    for step in range(1, payload.max_steps + 1):
        # ---------- 1. 让模型决策：只输出"调谁、传什么"，执行权始终在我们手里 ----------
        resp = await client.chat.completions.create(
            model=cfg.model_name,
            messages=messages,
            tools=tools.TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0,
        )
        msg = resp.choices[0].message

        # ---------- 2. 关键：把 assistant 消息回填（不做的话下一轮直接 400） ----------
        messages.append(msg.model_dump(exclude_none=True))

        # ---------- 3. 没有工具调用 = 模型认为可以回答了，循环结束 ----------
        if not msg.tool_calls:
            content = msg.content or ""
            steps.append({"step": step, "tool_calls": [], "content": content})
            return {
                "steps": steps,
                "final_answer": content,
                "stopped_by_limit": False,
                "escalated": False,
                "model": cfg.model_name,
            }

        # ---------- 4. 有工具调用：统一分发器执行，结果以 role=tool 回填 ----------
        # 一轮里可能同时发起多个调用（并行 tool_calls，互不依赖时模型会这么做），
        # 全部执行完并回填后才能进入下一轮
        step_calls: list[dict[str, Any]] = []
        for call in msg.tool_calls:
            name = call.function.name
            raw_args = call.function.arguments or "{}"

            # self-heal 观测：该工具上一轮报过错、本轮模型又来调它，说明它读到了
            # 错误信息并尝试自我纠正——打上标记，供 Day 05 重试策略分析
            self_heal = name in error_tools

            # 轨迹里展示的参数独立解析一份（日志是全进程共享的，并发下不能从那里取）
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            started = time.perf_counter()
            result = tools.dispatch(
                name, raw_args, user_confirmed=payload.user_confirmed
            )
            duration_ms = round((time.perf_counter() - started) * 1000)

            # Day 04 错误契约：ok=false 即失败，error_type 区分"我错了"还是"工具坏了"
            is_error = False
            error_type: str | None = None
            try:
                body = json.loads(result)
                if body.get("ok") is False:
                    is_error = True
                    error_type = body.get("error_type")
            except json.JSONDecodeError:
                pass

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": str(result)}
            )
            step_calls.append(
                {
                    "name": name,
                    "arguments": args if isinstance(args, dict) else {},
                    "result": str(result),
                    "duration_ms": duration_ms,
                    "is_error": is_error,
                    "error_type": error_type,
                    "self_heal": self_heal,
                }
            )

            if is_error:
                error_tools.add(name)
                # 自愈上限：同一错误反复出现说明模型卡住了，别再烧 token，熔断升级
                streak_key = (name, error_type or "unknown")
                error_streak[streak_key] = error_streak.get(streak_key, 0) + 1
                if error_streak[streak_key] >= tools.MAX_SAME_ERROR:
                    steps.append({"step": step, "tool_calls": step_calls, "content": None})
                    return {
                        "steps": steps,
                        "final_answer": ESCALATION_MESSAGE,
                        "stopped_by_limit": False,
                        "escalated": True,
                        "model": cfg.model_name,
                    }
            else:
                error_tools.discard(name)

        steps.append({"step": step, "tool_calls": step_calls, "content": None})

    # ---------- 5. 超步数保护：跑满也没等到最终回答 ----------
    return {
        "steps": steps,
        "final_answer": "",
        "stopped_by_limit": True,
        "escalated": False,
        "model": cfg.model_name,
    }
