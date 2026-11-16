"""学习模块工具层（Day 01–04）：参数模型 = 唯一真相源 + 三层校验流水线。

Day 04 升级要点（对照课程）：
- ``@tool`` 注册器：函数 + Pydantic 参数模型 → 工具说明书自动生成（不再手写
  JSON Schema），一份模型同时产出给模型的 Schema 和给代码的运行时校验器；
  附带 ``side_effect``（read/write/destructive）与 ``require_confirm`` 元数据；
- description 四段式：做什么 / 何时用 / 何时不用 / 返回什么（含空结果语义）；
- ``dispatch`` 三层校验流水线：JSON 解析 → Schema 校验 → 业务规则 → 确认门 → 执行；
- 错误即 Prompt：结构化 ``{ok, error_type, message, hint, errors:[{field, why, got}]}``，
  ``hint`` 用祈使句给"下一步动作"，模型据此自我纠正；
- 返回值剪裁：统一 ``{ok, data}``，列表带 total/returned，空结果显式表达语义；
- 练习 1：``SearchOrdersArgs`` 日期范围 + ``model_validator`` 跨度校验；
- 练习 2：``spec_budget()`` 统计说明书总字符数（每轮请求都重复付费的成本项）；
- 练习 3：``create_refund`` 幂等键，相同调用重复到达返回 ``duplicate: true``。
"""

import hashlib
import json
import math
import re
import time
from collections import deque
from datetime import date, datetime
from typing import Any, Callable, Literal

from pydantic import (
    BaseModel,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


# ============================================================
# 1. 注册器：函数 + 参数模型 → 工具说明书（schema 自动生成，不手写）
# ============================================================

REGISTRY: dict[str, dict[str, Any]] = {}


def _strip_titles(node: Any) -> Any:
    """Pydantic 生成的 schema 带一堆 title 字段，白占 token，递归删掉。"""
    if isinstance(node, dict):
        node.pop("title", None)
        for value in node.values():
            _strip_titles(value)
    elif isinstance(node, list):
        for value in node:
            _strip_titles(value)
    return node


def tool(
    args_model: type[BaseModel],
    side_effect: Literal["read", "write", "destructive"] = "read",
    require_confirm: bool = False,
    repeatable: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """把一个函数注册成模型可见的工具。

    - ``args_model``：参数合同（唯一真相源）——Schema 发给模型，校验留给 dispatch；
    - ``side_effect``：副作用分级，dispatch 据此统一施加重试/确认策略；
    - ``require_confirm``：True 时执行前必须经过用户确认（安全边界落在代码，
      不能落在提示词里——提示词模型有概率直接忽略）；
    - ``repeatable``：Day 05 轮询白名单——轮询类工具（查任务状态）天然要重复调用，
      终止逻辑对它不加"重复动作"计数，改为指数退避。
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        REGISTRY[fn.__name__] = {
            "fn": fn,
            "args_model": args_model,
            "side_effect": side_effect,
            "require_confirm": require_confirm,
            "repeatable": repeatable,
            "spec": {
                "type": "function",
                "function": {
                    "name": fn.__name__,
                    "description": (fn.__doc__ or "").strip(),
                    "parameters": _strip_titles(args_model.model_json_schema()),
                },
            },
        }
        return fn

    return deco


# ============================================================
# 2. 参数合同：参数收窄四板斧（枚举化 / 格式锚定 / 边界约束 / 扁平化）
# ============================================================


class WeatherArgs(BaseModel):
    """get_weather 的参数合同。"""

    city: str = Field(..., description="城市名（中文，不带'市'字），例如：杭州")
    unit: Literal["celsius", "fahrenheit"] = Field(
        "celsius", description="温度单位，默认摄氏度"
    )

    @field_validator("city")
    @classmethod
    def _clean_city(cls, value: str) -> str:
        # 校验即清洗：模型爱带"市"字，在这里剥掉，函数拿到的就是干净参数
        value = value.strip().replace("市", "")
        if not value:
            raise ValueError("city 不能为空")
        return value


class CalculatorArgs(BaseModel):
    """calculator 的参数合同（表达式白名单前移到校验层）。"""

    expression: str = Field(
        ..., min_length=1, description="数学表达式，例如 (18*7+46)/4 或 sqrt(2)"
    )

    @field_validator("expression")
    @classmethod
    def _check_whitelist(cls, value: str) -> str:
        # 白名单校验放在参数层：违规直接 invalid_args（可自我纠正），
        # 而不是执行阶段的 tool_crash（模型会误以为工具坏了）
        for token in _CALC_IDENTIFIER_PATTERN.findall(value):
            if token not in _CALC_ALLOWED_IDENTIFIERS:
                raise ValueError(
                    f"表达式包含白名单外的标识符 {token}，只支持数字、四则运算与数学函数"
                )
        return value


class CurrentTimeArgs(BaseModel):
    """current_time 无参数（空模型也要给，保持 Schema 来源统一）。"""


class SearchNewsArgs(BaseModel):
    """search_news 的参数合同。"""

    keyword: str = Field(
        ..., min_length=1, max_length=20, description="新闻检索关键词，例如 人工智能。提取核心词即可，不要整句传入"
    )


class SearchOrdersArgs(BaseModel):
    """search_orders 的参数合同（练习 1：日期范围 + 跨度校验）。"""

    user_id: str = Field(
        ...,
        pattern=r"^U\d{6}$",
        description="用户 ID，U 开头加 6 位数字，例如 U000123",
    )
    status: Literal["all", "paid", "shipped", "refunded"] = Field(
        "all", description="订单状态筛选；不确定该筛什么就传 all"
    )
    limit: int = Field(5, ge=1, le=20, description="最多返回几条，1~20，默认 5")
    date_from: str | None = Field(
        None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="起始日期 YYYY-MM-DD（按下单日过滤），不传则不限",
    )
    date_to: str | None = Field(
        None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="结束日期 YYYY-MM-DD，不传则不限"
    )

    @model_validator(mode="after")
    def _check_date_range(self) -> "SearchOrdersArgs":
        # 练习 1：from <= to 且跨度不超过 90 天——跨字段的规则 schema 表达不了，
        # 用 model_validator 兜住，同样走 invalid_args 纠错通道
        if self.date_from and self.date_to:
            if self.date_from > self.date_to:
                raise ValueError(
                    f"date_from {self.date_from} 晚于 date_to {self.date_to}，请交换两个日期"
                )
            span = (
                date.fromisoformat(self.date_to) - date.fromisoformat(self.date_from)
            ).days
            if span > 90:
                raise ValueError(f"日期跨度 {span} 天超过上限 90 天，请缩小查询范围")
        return self


class CreateRefundArgs(BaseModel):
    """create_refund 的参数合同（write 副作用 + 确认门）。"""

    order_id: str = Field(
        ...,
        pattern=r"^O\d{8}$",
        description="订单号，O 开头加 8 位数字，例如 O20260012",
    )
    amount: float = Field(..., gt=0, description="退款金额（元），不得超过订单实付金额")
    reason: str = Field(..., min_length=4, max_length=100, description="退款原因，至少 4 个字")


class CreateExportArgs(BaseModel):
    """create_export 的参数合同（异步任务：先拿到 task_id，再轮询状态）。"""

    month: str = Field(..., pattern=r"^\d{4}-\d{2}$", description="导出月份 YYYY-MM，例如 2026-10")
    export_type: Literal["orders", "refunds"] = Field(
        "orders", description="导出内容类型，默认 orders（订单）"
    )


class TaskStatusArgs(BaseModel):
    """get_task_status 的参数合同（Day 05 轮询白名单演示）。"""

    task_id: str = Field(
        ..., pattern=r"^T\d{4}$", description="导出任务 ID，T 开头加 4 位数字，例如 T8821"
    )


# ============================================================
# 3. 演示数据与工具实现（返回 dict，由 dispatch 统一包成 {ok, data}）
# ============================================================

# 演示订单库：只给一个用户 U000123，字段刻意精简（返回值剪裁原则）
_ORDERS_DB: dict[str, list[dict[str, Any]]] = {
    "U000123": [
        {"order_id": "O20260012", "status": "paid", "amount": 268.0, "items": ["机械键盘"], "date": "2026-11-05"},
        {"order_id": "O20260031", "status": "shipped", "amount": 89.0, "items": ["鼠标垫"], "date": "2026-11-08"},
        {"order_id": "O20260044", "status": "refunded", "amount": 39.0, "items": ["数据线"], "date": "2026-11-10"},
    ]
}

# 练习 3：退款幂等台账 key → 上次退款结果（学习项目存内存，重启即清）
_REFUND_LEDGER: dict[str, dict[str, Any]] = {}

# Day 05：异步导出任务（轮询场景演示）——task_id → 状态。
# 每次轮询 polls+1，第 3 次返回 done，模拟"轮询到状态变化才算进展"
_EXPORT_TASKS: dict[str, dict[str, Any]] = {}
_EXPORT_SEQ = 8820
_EXPORT_TOTAL_POLLS = 3


def _idempotency_key(parsed: CreateRefundArgs) -> str:
    """幂等键 = hash(订单号 + 金额 + 当日)：同一笔退款当日重复请求会命中台账。"""
    raw = f"{parsed.order_id}:{parsed.amount}:{date.today().isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---- 通用字符白名单（Day 03 练习 1 引入，现在由 CalculatorArgs 校验器调用） ----

_CALC_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_]+")
_CALC_ALLOWED_IDENTIFIERS = {
    "abs", "round", "sum",
    "sqrt", "sin", "cos", "tan", "log", "log10", "exp",
    "pi", "e", "ceil", "floor", "factorial",
}


@tool(WeatherArgs, side_effect="read")
def get_weather(city: str, unit: str = "celsius") -> dict[str, Any]:
    """查询指定城市的当前天气（温度、天气现象、风力）。

    何时用：用户问某地现在的天气、气温、要不要带伞、出行穿什么。
    何时不用：问未来几天的天气预报或历史天气——本工具只提供当前实况，请如实说明能力边界。
    返回：{city, unit, temp, condition}；condition 为"暂无该城市的天气数据"表示
         没有这个城市的数据，应如实告知用户，不要编造天气，也不要换写法反复重试。
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
    return {"city": city, "unit": unit, "temp": temp, "condition": cond}


@tool(CalculatorArgs, side_effect="read")
def calculator(expression: str) -> dict[str, Any]:
    """计算数学表达式，支持 + - * / ** 与 sqrt、sin、log 等 math 函数。

    何时用：凡涉及数值计算必须调用，禁止心算。
    何时不用：纯文本处理、日期推算（日期请用 current_time 配合推算）。
    返回：{expression, value}；表达式语法错误时 ok=false 并给出修正提示。
    """
    # 白名单校验已在 CalculatorArgs 的 field_validator 里完成，这里直接安全求值：
    # 只放行 math 模块安全函数并禁掉 __builtins__，避免注入任意代码
    safe_ns: dict[str, Any] = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    safe_ns.update({"abs": abs, "round": round, "sum": sum})
    try:
        value = eval(expression, {"__builtins__": {}}, safe_ns)  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        # ValueError 走 business_rule 通道（参数问题可自我纠正），其余视为工具缺陷
        raise ValueError(f"表达式无法计算（{type(exc).__name__}: {exc}），请检查语法") from exc
    return {"expression": expression, "value": value}


@tool(CurrentTimeArgs, side_effect="read")
def current_time() -> dict[str, Any]:
    """获取当前的日期和时间。

    何时用：用户问"现在几点""今天几号"，或回答需要当前日期做推算时。
    何时不用：与当前时间无关的问题不要调用。
    返回：{now}，格式 YYYY-MM-DD HH:MM:SS (星期)。
    """
    return {"now": datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")}


@tool(SearchNewsArgs, side_effect="read")
def search_news(keyword: str) -> dict[str, Any]:
    """按关键词搜索新闻，返回最新的新闻标题与日期。

    何时用：用户想了解某个关键词、话题、行业或公司的新闻动态。
    何时不用：用户问的是事实性常识而非"动态/最新消息"时，直接回答即可。
    返回：{keyword, total, returned, news:[{title, date}]}；total=0 表示没有
         检索到相关新闻，应如实告知，不要编造新闻标题。
    """
    # 练习数据：不管什么关键词都返回这三条，跑通"检索 → 引用 → 总结"的流程即可
    news = [
        {"title": f"{keyword}行业发布全新技术标准，产业链公司集体受益", "date": "2026-11-10"},
        {"title": f"专家解读：{keyword}成为今年增长最快的赛道之一", "date": "2026-11-09"},
        {"title": f"多家巨头宣布加码{keyword}领域，投资规模创新高", "date": "2026-11-08"},
    ]
    return {"keyword": keyword, "total": len(news), "returned": len(news), "news": news}


@tool(SearchOrdersArgs, side_effect="read")
def search_orders(
    user_id: str,
    status: str = "all",
    limit: int = 5,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """按用户查询订单列表（支持按状态与下单日期过滤）。

    何时用：用户问"我买了什么""最近几单什么状态""有没有发货"时。
    何时不用：已知具体订单号要查单笔详情（直接根据 search_orders 的结果回答）；
             用户只是问退款政策，直接回答，不用查订单。
    返回：{total, returned, orders:[...]}；total=0 表示该用户确实没有符合条件
         的订单，此时如实告知用户，不要更换 user_id 反复重试。
    """
    rows = _ORDERS_DB.get(user_id, [])
    if status != "all":
        rows = [r for r in rows if r["status"] == status]
    if date_from:
        rows = [r for r in rows if r["date"] >= date_from]
    if date_to:
        rows = [r for r in rows if r["date"] <= date_to]
    total = len(rows)
    rows = rows[:limit]
    # total + returned 让模型知道"还有更多"，空结果用语义化的 total=0 表达
    payload: dict[str, Any] = {"total": total, "returned": len(rows), "orders": rows}
    if total == 0:
        # Day 05 接缝：空结果是合法信息，不是错误。显式写清含义，
        # 模型才不会把"查法不对"和"确实没有"混为一谈、换写法反复重查
        payload["hint"] = "该用户确实没有符合条件的订单，换查询条件不会变多，请如实告知用户"
    return payload


@tool(CreateRefundArgs, side_effect="write", require_confirm=True)
def create_refund(order_id: str, amount: float, reason: str) -> dict[str, Any]:
    """为用户发起一笔退款（真实资金操作，不可撤销）。

    何时用：用户明确要求退款，且已给出订单号、金额与原因。
    何时不用：用户只是在咨询退款政策——直接用文字回答，不要调用本工具。
    返回：{refund_id, order_id, amount, status}；status=accepted 表示已受理，
         必须向用户复述订单号与金额。
    """
    refund: dict[str, Any] = {
        "refund_id": f"R{order_id[-8:]}",
        "order_id": order_id,
        "amount": amount,
        "status": "accepted",
        "reason": reason,
    }
    # 练习 3：登记幂等台账，并把演示订单标记为已退款（重复退款会被业务规则拦截）
    _REFUND_LEDGER[_idempotency_key(CreateRefundArgs(order_id=order_id, amount=amount, reason=reason))] = refund
    for order in _ORDERS_DB.get("U000123", []):
        if order["order_id"] == order_id:
            order["status"] = "refunded"
    return refund


@tool(CreateExportArgs, side_effect="write")
def create_export(month: str, export_type: str = "orders") -> dict[str, Any]:
    """创建一个订单/退款数据的异步导出任务，立即返回任务 ID。

    何时用：用户要求"导出报表""下载明细""把订单打包发我"时。
    何时不用：用户只是想看几条订单——直接用 search_orders 查出来回答即可。
    返回：{task_id, status:"running"}；拿到 task_id 后必须用 get_task_status
         轮询，status=done 时才有下载地址，不要自己编 URL。
    """
    global _EXPORT_SEQ
    _EXPORT_SEQ += 1
    task_id = f"T{_EXPORT_SEQ}"
    _EXPORT_TASKS[task_id] = {
        "task_id": task_id,
        "month": month,
        "type": export_type,
        "polls": 0,
        "status": "running",
    }
    return {"task_id": task_id, "status": "running", "month": month, "type": export_type}


@tool(TaskStatusArgs, side_effect="read", repeatable=True)
def get_task_status(task_id: str) -> dict[str, Any]:
    """查询导出任务的进度（轮询接口：任务没完成时需要重复调用）。

    何时用：拿到 create_export 返回的 task_id 后查询进度，直到 status=done。
    何时不用：没有 task_id 时不要调用；用户问别的事情时不要调用。
    返回：{task_id, status, progress}；status=running 表示还在处理，可以稍后再查；
         status=done 时额外返回 url，此时应把下载链接给用户。
    """
    task = _EXPORT_TASKS.get(task_id)
    if task is None:
        # 抛 ValueError → dispatch 走 business_rule 通道，模型会看到可纠正的提示
        raise ValueError(f"导出任务 {task_id} 不存在，请先用 create_export 创建任务")
    task["polls"] += 1
    # 前两次轮询 running，第三次 done——结果指纹随之变化，才算"有进展"
    if task["polls"] >= _EXPORT_TOTAL_POLLS:
        task["status"] = "done"
        return {
            "task_id": task_id,
            "status": "done",
            "progress": 100,
            "url": f"https://static.example.com/exports/{task_id}.xlsx",
        }
    return {
        "task_id": task_id,
        "status": "running",
        "progress": round(task["polls"] / _EXPORT_TOTAL_POLLS * 100),
    }


# ============================================================
# 4. 兼容别名：react_loop 与 /tools 接口仍按旧名字取清单和原始函数
# ============================================================

TOOLS: dict[str, Callable[..., Any]] = {
    name: entry["fn"] for name, entry in REGISTRY.items()
}
TOOL_SCHEMAS: list[dict[str, Any]] = [entry["spec"] for entry in REGISTRY.values()]


def tool_specs() -> list[dict[str, Any]]:
    """给 API 的 tools 参数（与 TOOL_SCHEMAS 等价，语义化名字）。"""
    return TOOL_SCHEMAS


def repeatable_tool_names() -> list[str]:
    """Day 05 轮询白名单：这些工具的重复调用不算"重复动作"，改为指数退避。"""
    return [name for name, entry in REGISTRY.items() if entry["repeatable"]]


# ============================================================
# 5. 错误工厂 + dispatch（三层校验流水线）
# ============================================================


def _err(error_type: str, message: str, hint: str, **extra: Any) -> str:
    """结构化错误工厂：错误不是日志，是喂给模型的下一轮 prompt。

    error_type 让模型区分"我错了还是工具坏了"；hint 用祈使句给下一步动作；
    extra 里通常带 errors=[{field, why, got}]（你传了什么也要告诉它）。
    """
    payload: dict[str, Any] = {
        "ok": False,
        "error_type": error_type,
        "message": message,
        "hint": hint,
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def _format_validation(exc: ValidationError) -> list[dict[str, Any]]:
    """把 Pydantic 报错压成模型能读的短结构：哪个字段、什么规则、你传了什么。"""
    return [
        {
            "field": ".".join(str(loc) for loc in item["loc"]) or "(root)",
            "why": item["msg"],
            "got": item.get("input"),
        }
        for item in exc.errors()
    ]


def _business_check(name: str, parsed: BaseModel) -> str | None:
    """业务规则校验（③ 层）：schema 只保证形状，业务合法性要靠代码。

    返回 None 表示通过；返回字符串则直接作为工具结果（可能是错误，
    也可能是幂等命中的 duplicate 成功——练习 3）。
    """
    if name != "create_refund":
        return None
    refund_args = parsed if isinstance(parsed, CreateRefundArgs) else None
    if refund_args is None:
        return None

    # 幂等键命中：同一笔退款当日重复请求 → 直接返回上次结果（发生在确认门之前，
    # 因为这笔退款已经真实受理过，不需要再确认一遍）
    key = _idempotency_key(refund_args)
    if key in _REFUND_LEDGER:
        return json.dumps(
            {"ok": True, "data": _REFUND_LEDGER[key], "duplicate": True},
            ensure_ascii=False,
        )

    order = next(
        (o for o in _ORDERS_DB.get("U000123", []) if o["order_id"] == refund_args.order_id),
        None,
    )
    if order is None:
        return _err(
            "business_rule",
            "订单号在系统中不存在",
            hint="先用 search_orders 查出真实订单号，再发起退款",
        )
    if order["status"] == "refunded":
        return _err(
            "business_rule",
            "该订单已退款，不能重复退款",
            hint="直接告知用户已退款，不要再次调用本工具",
        )
    if refund_args.amount > order["amount"]:
        return _err(
            "business_rule",
            f"退款金额 {refund_args.amount} 超过实付 {order['amount']}",
            hint=f"把 amount 调整为不超过 {order['amount']} 的值再说",
        )
    return None


# 练习 2：调用日志（时间/工具/参数/耗时/错误类型），内存环形队列，重启即清
TOOL_CALL_LOG: deque[dict[str, Any]] = deque(maxlen=200)

# 自愈上限：同一工具同一类错误连续超过该次数就熔断升级（由 agent_loop 消费）
MAX_SAME_ERROR: int = 2


def dispatch(name: str, raw_args: str, *, user_confirmed: bool = False) -> str:
    """统一分发器：三层校验流水线 + 执行（Day 04 核心入口）。

    流程：① JSON 解析 → ② Pydantic Schema 校验 → ③ 业务规则 →
    ④ 副作用确认门 → ⑤ 执行（校验并清洗过的参数，不是原始脏数据）。
    任何一步失败都不抛异常，返回结构化错误 JSON 让模型自我纠正。
    """
    entry = REGISTRY.get(name)
    if entry is None:
        result = _err(
            "unknown_tool",
            f"没有名为 {name} 的工具",
            hint="只能使用这些工具：" + "、".join(REGISTRY),
        )
        _log_call(name, {}, 0, result)
        return result

    started = time.perf_counter()
    error_type: str | None = None
    args: dict[str, Any] = {}
    result: str

    # ① JSON 解析：arguments 是字符串，模型偶尔给出非法 JSON
    try:
        args = json.loads(raw_args or "{}")
        if not isinstance(args, dict):
            raise ValueError("顶层必须是 JSON 对象")
    except (json.JSONDecodeError, ValueError) as exc:
        result = _err(
            "bad_json",
            f"arguments 无法解析：{exc}",
            hint='输出严格 JSON 对象，键和字符串都用双引号，例：{"user_id": "U000123"}',
        )
        error_type = "bad_json"

    # ② Schema + 约束校验（参数收窄四板斧的落地层）
    if error_type is None:
        try:
            parsed = entry["args_model"](**args)
        except ValidationError as exc:
            result = _err(
                "invalid_args",
                "参数未通过校验",
                hint="按 errors 逐条修正后重新调用；不要更换参数名，不要省略必填项",
                errors=_format_validation(exc),
            )
            error_type = "invalid_args"
        else:
            result, error_type = _run_validated(entry, name, parsed, user_confirmed)

    duration_ms = round((time.perf_counter() - started) * 1000)
    _log_call(name, args, duration_ms, result, error_type)
    return result


def _run_validated(
    entry: dict[str, Any], name: str, parsed: BaseModel, user_confirmed: bool
) -> tuple[str, str | None]:
    """校验通过后的 ③业务规则 → ④确认门 → ⑤执行 三步，返回 (结果, 错误类型)。"""
    # ③ 业务规则校验（可能直接返回幂等命中的 duplicate 成功）
    business = _business_check(name, parsed)
    if business is not None:
        try:
            ok = bool(json.loads(business).get("ok"))
        except json.JSONDecodeError:
            ok = False
        return business, None if ok else "business_rule"

    # ④ 副作用确认门：写操作未经确认一律拦截，preview 让模型复述时不出错
    if entry["require_confirm"] and not user_confirmed:
        return (
            _err(
                "need_confirmation",
                "该操作会产生真实资金变动，需用户确认",
                hint="先把参数整理成一句话请用户确认；用户明确同意后再以相同参数调用",
                preview=parsed.model_dump(),
            ),
            "need_confirmation",
        )

    # ⑤ 执行：传 model_dump()（校验并清洗过），不是模型原样的脏参数
    try:
        data = entry["fn"](**parsed.model_dump())
    except ValueError as exc:
        return (
            _err(
                "business_rule",
                f"执行失败：{exc}",
                hint="按提示修正参数后重试一次；再失败就如实告知用户",
            ),
            "business_rule",
        )
    except Exception as exc:  # noqa: BLE001
        return (
            _err(
                "tool_crash",
                f"工具内部异常：{type(exc).__name__}: {exc}",
                hint="这是工具缺陷，用同样参数重试无用；如实告知用户或换其他途径",
            ),
            "tool_crash",
        )
    return json.dumps({"ok": True, "data": data}, ensure_ascii=False), None


def _log_call(
    name: str,
    args: dict[str, Any],
    duration_ms: int,
    result: str,
    error_type: str | None = None,
) -> None:
    """把一次 dispatch 记入调用日志（练习 2 的落点，/tools/calls 接口读取）。"""
    is_error = error_type is not None
    TOOL_CALL_LOG.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "name": name,
            "arguments": args,
            "duration_ms": duration_ms,
            "is_error": is_error,
            "error_type": error_type,
        }
    )


# ============================================================
# 6. 练习 2：说明书预算——工具说明每轮请求都要重新付费
# ============================================================

SPEC_BUDGET_LIMIT: int = 2000


def spec_budget() -> dict[str, Any]:
    """统计 tools 清单序列化后的总字符数与最长的 3 个 description。

    超过预算（2000 字符）意味着每轮对话都在为说明书付超额 token 成本，
    应优先精简最长的 description 或考虑工具分组。
    """
    total = len(json.dumps(TOOL_SCHEMAS, ensure_ascii=False))
    costs = sorted(
        (
            {"name": entry["spec"]["function"]["name"], "chars": len(entry["spec"]["function"]["description"])}
            for entry in REGISTRY.values()
        ),
        key=lambda item: item["chars"],
        reverse=True,
    )
    return {
        "total_chars": total,
        "limit": SPEC_BUDGET_LIMIT,
        "over_limit": total > SPEC_BUDGET_LIMIT,
        "top_tools": costs[:3],
    }
