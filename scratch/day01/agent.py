import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import TOOLS, TOOL_SCHEMAS

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),  # 兼容服务时可填，官方留空即可
)
MODEL = os.getenv("MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """你是一个可以使用工具的助手。请严格遵守以下规则：
1. 需要外部事实（天气、时间）或精确计算时，必须调用工具，绝不允许凭记忆编造。
2. 一次只调用一个工具；拿到结果后再决定下一步，不要一次性猜多个。
3. 信息足够时直接给出最终答案，并用一两句话说明你是依据什么得出的。
4. 如果工具返回错误，可以换个参数重试，最多 2 次；仍然失败就如实告诉用户。
5. 不要向用户暴露工具名、JSON 参数等内部细节，用自然语言回答。
"""


def run_agent(user_input: str, max_steps: int = 8, verbose: bool = True) -> str:
    """运行一次 Agent 会话，返回最终回答。"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    for step in range(1, max_steps + 1):
        # ---------- 1. 让模型决策 ----------
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",   # 由模型自己决定要不要调工具
            temperature=0,        # 决策场景要确定性，别让它发挥
        )
        msg = resp.choices[0].message

        # ---------- 2. 关键：把 assistant 消息回填 ----------
        # 带 tool_calls 的这条消息必须原样进历史，否则下一轮会 400
        messages.append(msg.model_dump(exclude_none=True))

        # ---------- 3. 没有工具调用 = 模型认为可以回答了 ----------
        if not msg.tool_calls:
            if verbose:
                print(f"[step {step}] 模型直接回答，循环结束")
            return msg.content or ""

        # ---------- 4. 有工具调用：逐个执行 ----------
        for call in msg.tool_calls:
            name = call.function.name
            raw_args = call.function.arguments or "{}"

            if verbose:
                print(f"[step {step}] -> {name}({raw_args})")

            # 4.1 解析参数，模型偶尔会给出非法 JSON
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}

            # 4.2 查表执行，工具不存在 / 抛异常都转成文本回给模型
            fn = TOOLS.get(name)
            if fn is None:
                result = f"错误：不存在名为 {name} 的工具。可用工具：{list(TOOLS)}"
            else:
                try:
                    result = fn(**args)
                except TypeError as e:
                    result = f"错误：参数不匹配（{e}）。请检查必填参数。"
                except Exception as e:
                    result = f"错误：{type(e).__name__}: {e}"

            # 4.3 工具结果以 role=tool 回填，tool_call_id 必须对应
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),   # 一定要转字符串
            })

            if verbose:
                print(f"[step {step}] <- {str(result)[:80]}")

    # ---------- 5. 超步数保护 ----------
    return f"已达到最大步数 {max_steps}，任务未能完成。请缩小问题范围后重试。"


if __name__ == "__main__":
    print(run_agent("北京现在天气怎么样？如果气温低于 20 度，帮我算一下 18*7+46 等于多少。"))