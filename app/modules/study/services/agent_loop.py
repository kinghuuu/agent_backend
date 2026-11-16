"""Function Calling 版最小 Agent 循环（Day 01 主循环的工程化版本，Day 03 增强）。

与 scratch/day01/agent.py 的差别只有两点：
1. 不用 print，而是把每一轮的"决策 → 工具调用 → 观察结果"记录成结构化轨迹，
   交给前端可视化展示；
2. 模型配置来自数据库（模型管理），而不是 .env。

Day 03 增强：工具执行统一走 ``tools.dispatch``（错误当结果返回的 self-heal 契约）；
轨迹附带耗时与错误标记（练习 2）；上一轮报错的工具本轮被再次调用时标记
self_heal（练习 3，供 Day 05 重试策略分析）。

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
# 规则用编号列表、一条一个约束，才能被逐条测试（Day 16 评估体系的基础）
SYSTEM_PROMPT = """## 角色
你是一个可以使用工具的助手，帮用户查询事实信息并完成简单计算。

## 规则
1. 需要外部事实（天气、时间、新闻）或精确计算时，必须调用工具，绝不允许凭记忆编造。
2. 一次只调用一个工具；拿到结果后再决定下一步，不要一次性猜多个。
3. 如果工具返回错误，可以换个参数重试，最多 2 次；仍然失败就如实告诉用户。
4. 不要向用户暴露工具名、JSON 参数等内部细节，用自然语言回答。

## 输出格式
信息足够时直接给出最终答案，并用一两句话说明你是依据什么得出的。

## 示例
用户：北京现在多少度？
助手：（调用 get_weather 工具后）北京现在晴，气温 16℃，西北风 3 级，空气质量良。
"""


async def run(db: AsyncSession, payload: AgentRunRequest) -> dict[str, Any]:
    """运行一次 Agent 会话，返回完整执行轨迹（结构见 AgentRunResult）。"""
    cfg = await resolve_model(db, payload.model_config_id)
    client = build_client(cfg)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": payload.question},
    ]
    steps: list[dict[str, Any]] = []
    # 练习 3：上一轮报过错的工具名集合——本轮再次被调用即视为模型自我纠正（self-heal）
    error_tools: set[str] = set()

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
                "model": cfg.model_name,
            }

        # ---------- 4. 有工具调用：统一分发器执行，结果以 role=tool 回填 ----------
        # 一轮里可能同时发起多个调用（并行 tool_calls，互不依赖时模型会这么做），
        # 全部执行完并回填后才能进入下一轮
        step_calls: list[dict[str, Any]] = []
        for call in msg.tool_calls:
            name = call.function.name
            raw_args = call.function.arguments or "{}"

            # 练习 3（self-heal 观测）：该工具上一轮报过错、本轮模型又来调它，
            # 说明模型读到了错误信息并尝试自我纠正——打上标记，供 Day 05 重试策略分析
            self_heal = name in error_tools

            # 轨迹里展示的参数独立解析一份（日志是全进程共享的，并发下不能从那里取）
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            started = time.perf_counter()
            result = tools.dispatch(name, raw_args)
            duration_ms = round((time.perf_counter() - started) * 1000)

            # dispatch 的错误统一以 {"error": ...} JSON 返回；解析一下用于
            # 维护"上轮报错的工具"集合（工具自己返回的错误文本不算 dispatch 级错误）
            is_error = False
            try:
                is_error = "error" in json.loads(result)
            except json.JSONDecodeError:
                pass

            if is_error:
                error_tools.add(name)
            else:
                error_tools.discard(name)

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
                    "self_heal": self_heal,
                }
            )

        steps.append({"step": step, "tool_calls": step_calls, "content": None})

    # ---------- 5. 超步数保护：跑满也没等到最终回答 ----------
    return {
        "steps": steps,
        "final_answer": "",
        "stopped_by_limit": True,
        "model": cfg.model_name,
    }
