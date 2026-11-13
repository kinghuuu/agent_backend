"""Function Calling 版最小 Agent 循环（Day 01 主循环的工程化版本）。

与 scratch/day01/agent.py 的差别只有两点：
1. 不用 print，而是把每一轮的"决策 → 工具调用 → 观察结果"记录成结构化轨迹，
   交给前端可视化展示；
2. 模型配置来自数据库（模型管理），而不是 .env。

循环里三个不能省的工程约束（踩过坑才懂）：
- 最大步数：防"调工具 → 不满意 → 再调"的死循环；
- 异常兜底：工具报错不抛给上层，转成文本回给模型让它自己想办法；
- assistant 回填：带 tool_calls 的消息必须原样进历史，否则下一轮 400。
"""

import json
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

        # ---------- 4. 有工具调用：查表执行，结果以 role=tool 回填 ----------
        step_calls: list[dict[str, Any]] = []
        for call in msg.tool_calls:
            name = call.function.name
            raw_args = call.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            fn = tools.TOOLS.get(name)
            if fn is None:
                result = f"错误：不存在名为 {name} 的工具。可用工具：{list(tools.TOOLS)}"
            else:
                try:
                    result = fn(**args)
                except TypeError as exc:
                    result = f"错误：参数不匹配（{exc}）。请检查必填参数。"
                except Exception as exc:  # noqa: BLE001
                    result = f"错误：{type(exc).__name__}: {exc}"

            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": str(result)}
            )
            step_calls.append(
                {
                    "name": name,
                    "arguments": args if isinstance(args, dict) else {},
                    "result": str(result),
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
