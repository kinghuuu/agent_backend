"""Day 01 的工具定义：真实能力（普通 Python 函数）+ 给模型看的 JSON Schema 说明书。

学习要点（对应课程 Day 01 第五节）：
1. 工具 = 函数 + 说明书。模型看不到代码，只能读到 ``description`` 和 ``parameters``，
   它选不选这个工具、参数填得对不对，完全取决于说明书写得清不清楚。
2. description 是写给模型的（"何时必须调用"），不是写给人的注释。
3. 参数 Schema 越窄越好：参数越多，模型填错的概率越高。
4. 工具函数永远返回字符串——模型能读的只有文本。
"""

import json
import math
import re
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable

# ---------- 1. 真实能力：普通的 Python 函数 ----------


def get_weather(city: str, unit: str = "celsius") -> str:
    """查询指定城市的实时天气（当前用假数据跑通流程，Day 7 RAG 时可换成真实数据源）。

    unit 支持摄氏/华氏换算（Day 03：用 enum 收窄参数取值）。这个参数也是
    "串行依赖调度"的演示素材——模型可以先查摄氏度，再带着结果换算华氏度。
    返回值统一用 JSON 字符串，方便模型结构化引用。
    """
    fake_db: dict[str, tuple[int, str]] = {
        "北京": (18, "晴，西北风 3 级，空气质量良"),
        "上海": (28, "小雨，东南风 2 级"),
        "深圳": (27, "阵雨，湿度 85%"),
        "杭州": (26, "多云"),
    }
    temp, cond = fake_db.get(city, (22, "暂无该城市的天气数据"))
    if unit == "fahrenheit":
        temp = round(temp * 9 / 5 + 32)
    return json.dumps(
        {"city": city, "temp": temp, "unit": unit, "condition": cond},
        ensure_ascii=False,
    )


# Day 03 练习 1：eval 前的字符白名单。数字与四则运算符之外，
# 只允许出现在安全函数表里的标识符（sqrt/pi/...），其余一律拒绝——
# "先校验再 eval"是工具入参防御的第一道闸（Day 18 安全专题展开）
_CALC_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_]+")
_CALC_ALLOWED_IDENTIFIERS = {
    "abs", "round", "sum",
    "sqrt", "sin", "cos", "tan", "log", "log10", "exp",
    "pi", "e", "ceil", "floor", "factorial",
}


def calculator(expression: str) -> str:
    """计算数学表达式。

    双重防御：先做字符白名单校验（练习 1），再放行 math 模块安全函数并禁掉
    ``__builtins__``。计算失败统一返回带 ``error`` 键的 JSON——错误也是
    上下文，模型看到后下一轮往往会自己修正表达式（self-heal 契约）。
    """
    for token in _CALC_IDENTIFIER_PATTERN.findall(expression):
        if token not in _CALC_ALLOWED_IDENTIFIERS:
            return json.dumps(
                {"error": f"不允许的标识符 {token}（白名单外），只支持数学函数与数字"},
                ensure_ascii=False,
            )
    safe_ns: dict[str, Any] = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_ns.update({"abs": abs, "round": round, "sum": sum})
    try:
        return str(eval(expression, {"__builtins__": {}}, safe_ns))  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": f"计算失败：{type(exc).__name__}: {exc}"}, ensure_ascii=False)


def current_time() -> str:
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


def search_news(keyword: str) -> str:
    """按关键词搜索新闻，返回 3 条假新闻标题（Day 01 练习 1：加一个工具）。

    重点是让模型学会"遇到新闻类问题先检索再回答"——description 里写明了
    何时必须调用、禁止凭训练记忆编造新闻，返回值用 JSON 字符串方便模型引用。
    """
    # 练习数据：不管什么关键词都返回这三条，跑通"检索 → 引用 → 总结"的流程即可
    return json.dumps(
        [
            {"title": f"{keyword}行业发布全新技术标准，产业链公司集体受益", "date": "2026-11-10"},
            {"title": f"专家解读：{keyword}成为今年增长最快的赛道之一", "date": "2026-11-09"},
            {"title": f"多家巨头宣布加码{keyword}领域，投资规模创新高", "date": "2026-11-08"},
        ],
        ensure_ascii=False,
    )


# ---------- 2. 工具注册表：名字 -> 可执行函数（Executor 的"查表执行"就靠它） ----------

TOOLS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
    "calculator": calculator,
    "current_time": current_time,
    "search_news": search_news,
}


# ---------- 2.5 统一分发器（Day 03 核心）：解析参数 → 调函数 → 统一字符串出参 ----------

# 练习 2：调用日志（内存环形队列，最多留 200 条；学习项目不落盘，重启即清）。
# 用来观察"一轮 Agent 对话到底真实执行了几次工具、耗时多少、哪些报错了"
TOOL_CALL_LOG: deque[dict[str, Any]] = deque(maxlen=200)


def dispatch(name: str, arguments_json: str) -> str:
    """统一分发器：解析参数 JSON → 查表调用 → 统一返回字符串。

    工具不存在 / 参数非法 / 执行出错都不抛异常，把错误信息当结果返回——
    错误信息也是上下文，模型看到 ``{"error": ...}`` 后下一轮往往会自己
    补参重试（self-heal，Day 05 重试策略的基础）。agent_loop 循环里只
    需要调这一个函数，之后新增工具这里一行都不用改。
    """
    fn = TOOLS.get(name)
    if fn is None:
        result = json.dumps(
            {"error": f"工具 {name} 不存在，可用工具：{list(TOOLS)}"},
            ensure_ascii=False,
        )
        TOOL_CALL_LOG.append(
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "name": name,
                "arguments": {},
                "duration_ms": 0,
                "is_error": True,
            }
        )
        return result

    started = time.perf_counter()
    is_error = False
    args: dict[str, Any] = {}
    try:
        args = json.loads(arguments_json or "{}")
        if not isinstance(args, dict):
            raise TypeError(f"参数必须是 JSON 对象，收到 {type(args).__name__}")
        result = fn(**args)
    except (json.JSONDecodeError, TypeError) as exc:
        result = json.dumps({"error": f"参数错误: {exc}"}, ensure_ascii=False)
        is_error = True
    except Exception as exc:  # noqa: BLE001
        result = json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
        is_error = True
    duration_ms = round((time.perf_counter() - started) * 1000)

    TOOL_CALL_LOG.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "name": name,
            "arguments": args,
            "duration_ms": duration_ms,
            "is_error": is_error,
        }
    )
    return result

# ---------- 3. 工具说明书：随请求发给模型的 JSON Schema ----------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查询指定城市的实时天气，返回天气现象、气温、风力。"
                "当用户询问某地天气、气温、是否下雨、要不要带伞时必须调用本工具，不要凭记忆回答。"
                "默认返回摄氏度；用户明确要求华氏度时把 unit 传 fahrenheit。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市中文名称，例如 北京、上海。不要带'市'字后缀。",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认 celsius（摄氏度）",
                    },
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
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "按关键词搜索新闻，返回 3 条最新新闻标题和日期。"
                "当用户想了解某个关键词、话题、行业或公司的新闻动态时，"
                "必须调用本工具进行检索，禁止凭训练记忆编造新闻。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "新闻检索关键词，例如 人工智能、新能源。提取用户问题中的核心词即可，不要整句传入。",
                    }
                },
                "required": ["keyword"],
            },
        },
    },
]
