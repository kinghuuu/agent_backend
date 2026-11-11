"""不用 Function Calling 的手写 ReAct 循环（Day 01 第九节的工程化版本）。

所谓 Function Calling，本质是让模型按约定格式输出文本、再由厂商保证结构合法；
这里把"魔法"拆掉：自己定义 Thought/Action/Action Input 文本协议，自己写正则解析。

两个价值连城的细节：
- ``stop=["Observation:"]``：不加的话模型会自己把 Observation 编出来（凭空捏造工具结果）；
- 解析失败不抛异常，而是回一句"格式错误，请重新输出"——出错 → 告知 → 重试，
  这是后期做 Agent 可靠性（Day 20）的基础套路。
"""

import json
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import ReactRunRequest
from app.modules.study.services import tools
from app.modules.study.services.llm import build_client, resolve_model

# 工具清单渲染成纯文本塞进 prompt——模型的"能力感知"完全来自这段文字
TOOL_DESC = "\n".join(
    f"- {t['function']['name']}: {t['function']['description']}" for t in tools.TOOL_SCHEMAS
)

REACT_PROMPT = f"""你可以使用以下工具：
{TOOL_DESC}

你必须严格按下面的格式输出，一次只输出一步，不要多写：

Thought: 你的思考过程
Action: 工具名
Action Input: 参数 JSON，例如 {{"city": "北京"}}

当你已经能回答用户时，改为输出：

Thought: 我已获得足够信息
Final Answer: 给用户的最终回答

禁止跳过 Thought，禁止一次输出多个 Action。
"""


def parse_step(text: str) -> dict[str, Any]:
    """把模型的一段文本解析成结构化动作（final / action / error 三种）。"""
    if "Final Answer:" in text:
        return {"type": "final", "content": text.split("Final Answer:", 1)[1].strip()}

    m_action = re.search(r"Action:\s*([A-Za-z_]\w*)", text)
    if not m_action:
        return {"type": "error", "content": "没有解析到 Action，请严格按格式输出"}
    name = m_action.group(1)
    m_input = re.search(r"Action Input:\s*(.+)", text)
    raw = (m_input.group(1).strip() if m_input else "{}").strip("`").strip()
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        args = {"input": raw}  # 兜底：包一层，让工具自己去报参数错
    return {"type": "action", "name": name, "args": args}


async def run(db: AsyncSession, payload: ReactRunRequest) -> dict[str, Any]:
    """运行一次手写 ReAct 会话，返回完整执行轨迹（结构见 ReactRunResult）。"""
    cfg = await resolve_model(db, payload.model_config_id)
    client = build_client(cfg)

    history = ""  # 纯文本的"记忆"，靠字符串拼接（对比 agent_loop 的 messages 列表）
    steps: list[dict[str, Any]] = []

    for step in range(1, payload.max_steps + 1):
        prompt = f"{REACT_PROMPT}\n\n已有记录：\n{history}\n\n用户问题：{payload.question}\n"
        resp = await client.chat.completions.create(
            model=cfg.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stop=["Observation:"],  # 关键：防止模型自己编造观察结果
        )
        text = resp.choices[0].message.content or ""

        parsed = parse_step(text)
        entry: dict[str, Any] = {
            "step": step,
            "raw_text": text,
            "action_name": parsed.get("name"),
            "action_args": parsed.get("args"),
            "observation": None,
            "is_error": parsed["type"] == "error",
        }

        if parsed["type"] == "final":
            steps.append(entry)
            return {
                "steps": steps,
                "final_answer": parsed["content"],
                "stopped_by_limit": False,
                "model": cfg.model_name,
            }

        if parsed["type"] == "error":
            entry["observation"] = "格式错误，请重新输出"
            steps.append(entry)
            history += f"\n{text}\nObservation: 格式错误，请重新输出\n"
            continue

        fn = tools.TOOLS.get(parsed["name"])
        try:
            observation = str(fn(**parsed["args"])) if fn else f"不存在工具 {parsed['name']}"
        except Exception as exc:  # noqa: BLE001
            observation = f"执行出错：{exc}"

        entry["observation"] = observation
        steps.append(entry)
        history += f"\n{text}\nObservation: {observation}\n"

    return {
        "steps": steps,
        "final_answer": "",
        "stopped_by_limit": True,
        "model": cfg.model_name,
    }
