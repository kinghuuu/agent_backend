import json
import re

from openai import OpenAI

from agent import client, MODEL          # 复用上文的 client
from tools import TOOLS, TOOL_SCHEMAS

# 1. 把工具清单渲染成纯文本，塞进 system prompt
tool_desc = "\n".join(
    f"- {t['function']['name']}: {t['function']['description']}"
    for t in TOOL_SCHEMAS
)

REACT_PROMPT = f"""你可以使用以下工具：
{tool_desc}

你必须严格按下面的格式输出，一次只输出一步，不要多写：

Thought: 你的思考过程
Action: 工具名
Action Input: 参数 JSON，例如 {{"city": "北京"}}

当你已经能回答用户时，改为输出：

Thought: 我已获得足够信息
Final Answer: 给用户的最终回答

禁止跳过 Thought，禁止一次输出多个 Action。
"""


def parse_step(text: str) -> dict:
    """把模型的一段文本解析成结构化动作。"""
    if "Final Answer:" in text:
        return {"type": "final", "content": text.split("Final Answer:", 1)[1].strip()}

    m_action = re.search(r"Action:\s*([A-Za-z_]\w*)", text)
    m_input = re.search(r"Action Input:\s*(.+)", text)
    if not m_action:
        return {"type": "error", "content": "没有解析到 Action，请严格按格式输出"}

    name = m_action.group(1)
    raw = (m_input.group(1).strip() if m_input else "{}")
    raw = raw.strip("`").strip()          # 模型爱加反引号，清掉
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        args = {"input": raw}              # 兜底：包一层，让工具自己去报参数错
    return {"type": "action", "name": name, "args": args}


def run_react(user_input: str, max_steps: int = 6) -> str:
    history = ""   # 纯文本的"记忆"，靠字符串拼接
    for step in range(1, max_steps + 1):
        prompt = f"{REACT_PROMPT}\n\n已有记录：\n{history}\n\n用户问题：{user_input}\n"
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stop=["Observation:"],   # 关键！防止模型自己编造观察结果
        )
        text = resp.choices[0].message.content or ""
        print(f"--- step {step} ---\n{text}")

        parsed = parse_step(text)
        if parsed["type"] == "final":
            return parsed["content"]

        if parsed["type"] == "error":
            history += f"\n{text}\nObservation: 格式错误，请重新输出\n"
            continue

        fn = TOOLS.get(parsed["name"])
        try:
            observation = str(fn(**parsed["args"])) if fn else f"不存在工具 {parsed['name']}"
        except Exception as e:
            observation = f"执行出错：{e}"

        history += f"\n{text}\nObservation: {observation}\n"

    return "已达到最大步数，未能完成。"