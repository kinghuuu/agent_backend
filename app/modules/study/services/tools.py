"""Day 01 的工具定义：真实能力（普通 Python 函数）+ 给模型看的 JSON Schema 说明书。

学习要点（对应课程 Day 01 第五节）：
1. 工具 = 函数 + 说明书。模型看不到代码，只能读到 ``description`` 和 ``parameters``，
   它选不选这个工具、参数填得对不对，完全取决于说明书写得清不清楚。
2. description 是写给模型的（"何时必须调用"），不是写给人的注释。
3. 参数 Schema 越窄越好：参数越多，模型填错的概率越高。
4. 工具函数永远返回字符串——模型能读的只有文本。
"""

import math
from datetime import datetime
from typing import Any, Callable

# ---------- 1. 真实能力：普通的 Python 函数 ----------


def get_weather(city: str) -> str:
    """查询指定城市的实时天气（当前用假数据跑通流程，Day 7 RAG 时可换成真实数据源）。"""
    fake_db = {
        "北京": "晴，气温 16℃，西北风 3 级，空气质量良",
        "上海": "多云，气温 23℃，东南风 2 级",
        "深圳": "阵雨，气温 27℃，湿度 85%",
    }
    return fake_db.get(city, f"暂时没有 {city} 的天气数据")


def calculator(expression: str) -> str:
    """计算数学表达式。

    只放行 math 模块里的安全函数并禁掉 ``__builtins__``，避免模型给出的
    表达式被注入任意代码——这是工具白名单思想的雏形（Day 18 安全专题展开）。
    """
    safe_ns: dict[str, Any] = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_ns.update({"abs": abs, "round": round, "sum": sum})
    try:
        return str(eval(expression, {"__builtins__": {}}, safe_ns))  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return f"计算失败：{type(exc).__name__}: {exc}"


def current_time() -> str:
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


# ---------- 2. 工具注册表：名字 -> 可执行函数（Executor 的"查表执行"就靠它） ----------

TOOLS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
    "calculator": calculator,
    "current_time": current_time,
}

# ---------- 3. 工具说明书：随请求发给模型的 JSON Schema ----------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查询指定城市的实时天气，返回天气现象、气温、风力。"
                "当用户询问某地天气、气温、是否下雨、要不要带伞时必须调用本工具，不要凭记忆回答。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名称，例如 北京、上海。不要带'市'字后缀。",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "计算数学表达式，支持 + - * / ** 及 sqrt、sin、log 等 math 函数。"
                "凡涉及数值计算，必须调用本工具，禁止自己心算。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "合法的 Python 数学表达式，例如 (18*7+46)/4 或 sqrt(2)",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "获取当前日期和时间。用户问'现在几点''今天几号'时必须调用。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]
