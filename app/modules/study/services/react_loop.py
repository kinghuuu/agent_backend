"""不用 Function Calling 的手写 ReAct 循环（Day 01 第九节的工程化版本）。

所谓 Function Calling，本质是让模型按约定格式输出文本、再由厂商保证结构合法；
这里把"魔法"拆掉：自己定义 Thought/Action/Action Input 文本协议，自己写正则解析。

两个价值连城的细节：
- ``stop=["Observation:"]``：不加的话模型会自己把 Observation 编出来（凭空捏造工具结果）；
- 解析失败不抛异常，而是回一句"格式错误，请重新输出"——出错 → 告知 → 重试，
  这是后期做 Agent 可靠性（Day 20）的基础套路。

练习 3（多轮对话）：history 由函数局部变量提成 ``ReactAgent`` 的成员变量，
配合 ``ReactSessionManager``（session_id → 实例的内存注册表），
同一个 session_id 的多次请求共享一份记忆，实现"接着上一句继续问"。
学习项目不做落库：会话只存在进程内存里，重启即清空，上限淘汰防止内存膨胀。
"""

import json
import re
from typing import Any
from uuid import uuid4

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


class ReactAgent:
    """持有跨请求记忆（history）的 ReAct 会话实例。

    练习 3 的核心改动点：history 不再是 ``run`` 的局部变量，
    而是实例成员——实例活着，记忆就在；实例被注册表淘汰，记忆才消失。
    """

    def __init__(self) -> None:
        # 纯文本的"记忆"，靠字符串拼接（对比 agent_loop 的 messages 列表）
        # 内容 = 历次「用户问题 + 模型输出 + Observation」，随 prompt 整体发给模型
        self.history: str = ""
        # 已完成的提问轮数（给前端展示"这是第几轮追问"）
        self.turns: int = 0

    async def run(self, db: AsyncSession, payload: ReactRunRequest) -> dict[str, Any]:
        """在既有记忆之上运行一轮 ReAct 会话，返回本轮执行轨迹。"""
        cfg = await resolve_model(db, payload.model_config_id)
        client = build_client(cfg)

        self.turns += 1
        # 本轮新增的历史片段先攒在局部变量，跑完再一次性并入 self.history——
        # 这样中途抛异常也不会把"半截记录"污染进跨轮记忆
        new_history = f"\n用户问题：{payload.question}\n"
        steps: list[dict[str, Any]] = []

        for step in range(1, payload.max_steps + 1):
            # prompt 每轮都是全新拼接：说明 + 全部历史（含本轮已发生的步骤）+ 当前问题
            # "用户问题："缀在最后，起到标记"最新指令"的作用，模型因此能理解追问的指代
            prompt = (
                f"{REACT_PROMPT}\n\n已有记录：\n{self.history}{new_history}"
                f"\n\n用户问题：{payload.question}\n"
            )
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
                self.history += f"{new_history}{text}\n"
                return {
                    "steps": steps,
                    "final_answer": parsed["content"],
                    "stopped_by_limit": False,
                    "model": cfg.model_name,
                }

            if parsed["type"] == "error":
                entry["observation"] = "格式错误，请重新输出"
                steps.append(entry)
                new_history += f"{text}\nObservation: 格式错误，请重新输出\n"
                continue

            fn = tools.TOOLS.get(parsed["name"])
            try:
                observation = str(fn(**parsed["args"])) if fn else f"不存在工具 {parsed['name']}"
            except Exception as exc:  # noqa: BLE001
                observation = f"执行出错：{exc}"

            entry["observation"] = observation
            steps.append(entry)
            new_history += f"{text}\nObservation: {observation}\n"

        # 跑满也没等到 Final Answer：本轮已发生的步骤同样并入记忆（下轮可参考）
        self.history += new_history
        return {
            "steps": steps,
            "final_answer": "",
            "stopped_by_limit": True,
            "model": cfg.model_name,
        }


class ReactSessionManager:
    """内存会话注册表：session_id → ReactAgent 实例。

    学习项目没有落库，会话状态只能活在进程内存里（重启即清空）。
    用插入序淘汰最老的会话，防止长期运行内存无限膨胀。
    """

    # 最多同时保留的会话数：超出后淘汰最早创建的
    MAX_SESSIONS: int = 50

    def __init__(self) -> None:
        self._agents: dict[str, ReactAgent] = {}

    def get_or_create(self, session_id: str | None) -> tuple[str, ReactAgent]:
        """取回（或新建）一个会话实例，返回 (session_id, agent)。

        session_id 为 None 时自动生成新 ID（一次性会话也可以之后继续追问）。
        """
        if session_id and session_id in self._agents:
            return session_id, self._agents[session_id]

        sid = session_id or uuid4().hex
        self._agents[sid] = ReactAgent()
        # 超上限淘汰：dict 保持插入序，next(iter(...)) 就是最早创建的会话
        while len(self._agents) > self.MAX_SESSIONS:
            del self._agents[next(iter(self._agents))]
        return sid, self._agents[sid]

    def reset(self, session_id: str) -> None:
        """清空指定会话的记忆（保留实例占位，方便前端"清空历史"复用同一 ID）。"""
        if session_id in self._agents:
            self._agents[session_id] = ReactAgent()


# 模块级单例：整个后端进程共享一份会话表
session_manager = ReactSessionManager()


async def run(db: AsyncSession, payload: ReactRunRequest) -> dict[str, Any]:
    """运行一轮 ReAct 会话（views 调用的入口，签名与改造前一致）。

    按 session_id 从注册表取回持有历史记忆的 ReactAgent 实例执行；
    返回值额外带上 session_id / history / turns，供前端展示与续问。
    """
    session_id, agent = session_manager.get_or_create(payload.session_id)
    result = await agent.run(db, payload)
    result.update({"session_id": session_id, "history": agent.history, "turns": agent.turns})
    return result
