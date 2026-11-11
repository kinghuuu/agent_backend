import json
import math
from datetime import datetime

# ---------- 1. 真实能力：普通的 Python 函数 ----------

def get_weather(city: str) -> str:
    """查询指定城市的实时天气。"""
    # 真实项目里这里是一次 HTTP 请求，今天先用假数据把流程跑通
    fake_db = {
        "北京": "晴，气温 16℃，西北风 3 级，空气质量良",
        "上海": "多云，气温 23℃，东南风 2 级",
        "深圳": "阵雨，气温 27℃，湿度 85%",
    }
    return fake_db.get(city, f"暂时没有 {city} 的天气数据")


def calculator(expression: str) -> str:
    """计算一个数学表达式，支持 + - * / ** 和 sqrt、sin 等 math 函数。"""
    # 只放行 math 里的安全函数，禁掉 __builtins__，避免被注入任意代码
    safe_ns = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_ns.update({"abs": abs, "round": round, "sum": sum})
    try:
        return str(eval(expression, {"__builtins__": {}}, safe_ns))
    except Exception as e:
        return f"计算失败：{type(e).__name__}: {e}"


def current_time() -> str:
    """获取当前的日期和时间。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


def search_news(keyword: str) -> str:
    """按关键词搜索新闻，返回 3 条假新闻标题（Day01 练习用）。"""
    # 练习数据：不管什么关键词都返回这三条，重点是让模型学会调用并引用结果
    return json.dumps(
        [
            {"title": f"{keyword}行业发布全新技术标准，产业链公司集体受益", "date": "2026-11-10"},
            {"title": f"专家解读：{keyword}成为今年增长最快的赛道之一", "date": "2026-11-09"},
            {"title": f"多家巨头宣布加码{keyword}领域，投资规模创新高", "date": "2026-11-08"},
        ],
        ensure_ascii=False,
    )


# ---------- 2. 工具注册表：名字 -> 可执行函数 ----------

TOOLS = {
    "get_weather": get_weather,
    "calculator": calculator,
    "current_time": current_time,
    "search_news": search_news,
}


# ---------- 3. 工具说明书：给模型看的 JSON Schema ----------

TOOL_SCHEMAS = [
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