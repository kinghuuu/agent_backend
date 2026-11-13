"""Day 02 结构化输出实验：客户意图提取器（三级方案对比 + 失败现场回填重试）。

对应课程 Day 02 的核心知识点：
- 消息结构的实战价值：解析/校验失败时，把模型的错误输出和报错信息 append 回
  messages 再重试——消息列表不只是历史，还是**纠错通道**，模型"看到"自己错在哪，
  第二次成功率极高；
- 三级进化：口头要求 JSON（反面教材，解析必崩）→ JSON Mode（API 级开关，
  只保证合法 JSON）→ Pydantic Schema（合同级约束，保证字段/类型/枚举/取值范围）；
- 练习 1：intent 枚举扩展为 6 类（含"其他"）；练习 2：嵌套字段 items；
  练习 3："校验失败 → 回填 → 重试"泛化为通用的 ``retry_until_valid``。

兼容性（真实工程经验）：``response_format={"type": "json_schema"}`` 并非所有
OpenAI 兼容服务都支持（如 DeepSeek 只支持 json_object），且 strict 模式对
Schema 有额外要求（所有字段必填、禁 additionalProperties、不允许 default），
遇到 400 自动降级为 json_object + Pydantic 校验兜底，降级动作记入轨迹。
"""

import json
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import ExtractRunRequest
from app.modules.study.services.llm import build_client, resolve_model


# ---------- 工单抽取合同：Pydantic Schema（练习 1 + 练习 2） ----------

# 意图枚举取值（练习 1：6 类，含"其他"）。Schema prompt 与 Pydantic 模型共用，
# 改这里一处即可——枚举取值只许出现一个事实来源，否则两处必然漂移
INTENT_VALUES = ("咨询", "催发货", "投诉", "退款", "换货", "其他")

# Literal 接收元组即可展开为字面量联合类型，与上面的取值保持同源
IntentType = Literal[INTENT_VALUES]  # type: ignore[valid-type]


class ItemInfo(BaseModel):
    """工单里提到的单个商品（练习 2：嵌套字段，观察模型对复杂 Schema 的遵循）。"""

    product: str = Field(description="商品名称")
    quantity: int = Field(default=1, ge=1, description="数量，默认 1")


class TicketInfo(BaseModel):
    """工单抽取结果的严格合同：字段、类型、枚举、取值范围全部约束。

    这份合同对应课程 2.3 节——JSON Mode 只保证"合法 JSON"，
    字段对不对、类型对不对、取值合不合法，全靠 Pydantic 验收。
    """

    name: str = Field(description="客户姓名")
    phone: str | None = Field(default=None, description="11 位手机号，未提供则为 null")
    intent: IntentType = Field(description="客户意图分类")
    urgency: int = Field(ge=1, le=5, description="紧急程度 1-5")
    items: list[ItemInfo] = Field(default_factory=list, description="涉及的商品列表")


SCHEMA_PROMPT = f"""你是工单信息抽取器。从用户输入中提取字段，只输出一个 JSON 对象，
不要任何解释、不要 markdown 代码块。字段定义：
{{"name": str, "phone": str|null, "intent": "{"|".join(INTENT_VALUES)}", "urgency": 1-5, "items": [{{"product": str, "quantity": int}}]}}
用户没提到的字段填 null、空列表或合理默认值，禁止编造。"""


# ---------- 校验器：失败抛 ValueError，异常信息就是"回填给模型的纠错 prompt" ----------


def _validate_json_object(content: str) -> dict[str, Any]:
    """把模型原文校验成 JSON 对象（json_mode 模式用，只保证合法、不保证字段）。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"上面的输出不是合法 JSON（{exc}）。"
            "重新输出，只给一个纯 JSON 对象，不要解释、不要 markdown 代码块。"
        ) from exc


def _validate_ticket(content: str) -> TicketInfo:
    """把模型原文校验成 TicketInfo（json_schema 模式用，字段/类型/枚举/范围全验）。"""
    try:
        return TicketInfo.model_validate_json(content)
    except ValidationError as exc:
        # 截断错误详情：Pydantic 全量报错可能很长，回填太长反而稀释关键信息
        raise ValueError(
            f"校验失败：{str(exc)[:400]}。"
            "请严格按字段定义重新输出，字段名、类型、枚举取值必须完全一致，"
            "用户没提到的字段填 null 或空列表。"
        ) from exc


async def retry_until_valid(
    messages: list[dict[str, Any]],
    validate: Callable[[str], Any],
    fetch: Callable[[int], Any],
    *,
    max_retries: int,
) -> tuple[Any | None, list[dict[str, Any]]]:
    """通用的"调用 → 校验 → 失败回填 → 重试"循环（练习 3 的泛化）。

    可套在任意"LLM 函数"上：
    - ``validate``：把模型原文转成目标对象，校验失败抛 ValueError，
      异常信息会原样作为 user 消息回填（错误信息本身就是最有价值的 prompt）；
    - ``fetch``：执行一次补全请求，返回 (原文, 本轮输出约束名, 降级说明或 None)，
      接收 attempt 序号，便于实现"第一次带严格约束、之后降级"之类的策略。
    返回 (最终对象或 None, 尝试轨迹)。轨迹结构与 ExtractAttemptTrace 对应。
    """
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_retries + 2):
        content, fmt, note = await fetch(attempt)
        entry: dict[str, Any] = {
            "attempt": attempt,
            "raw_content": content,
            "parsed": None,
            "error": None,
            "response_format": fmt,
            "note": note,
        }
        try:
            obj = validate(content)
        except ValueError as exc:
            entry["error"] = str(exc)
            attempts.append(entry)
            # 失败现场回填：assistant 原文 + 报错说明，让模型看到自己错在哪
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": str(exc)})
            continue

        entry["parsed"] = obj if isinstance(obj, dict) else obj.model_dump()
        attempts.append(entry)
        return obj, attempts

    # 连续失败：返回 None 交给上层如实告知，绝不把编造的结果混过去
    return None, attempts


def _make_json_schema_fetch(client: Any, model: str, messages: list[dict[str, Any]]):
    """构造 json_schema 模式的 fetch：第一次尝试严格约束，不支持则降级 json_object。

    Pydantic 生成的 Schema 带默认值、不带 additionalProperties: false，
    OpenAI strict 模式可能直接 400——这正是课程"坑 3"，所以必须留降级通道。
    """
    schema_tried = False
    degraded_note: str | None = None

    async def fetch(attempt: int) -> tuple[str, str, str | None]:
        nonlocal schema_tried, degraded_note
        if not schema_tried:
            schema_tried = True
            try:
                from openai import BadRequestError  # 延迟导入，与 llm.py 的约定一致

                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "ticket_info",
                            "strict": True,
                            "schema": TicketInfo.model_json_schema(),
                        },
                    },
                )
                return resp.choices[0].message.content or "", "json_schema", None
            except BadRequestError as exc:
                # 仅对"约束不被支持"类的 400 降级；网络/鉴权等其他错误照常抛出
                degraded_note = (
                    f"服务不支持 json_schema 严格约束（{type(exc).__name__}），"
                    "已降级为 json_object + Pydantic 校验兜底"
                )
                # 走到这里说明降级：本轮改用 json_object 完成请求，note 随轨迹展示
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )
        # 降级说明只在发生降级的那一轮附带一次，避免轨迹里重复刷屏
        note, degraded_note = degraded_note, None
        return resp.choices[0].message.content or "", "json_object", note

    return fetch


async def run(db: AsyncSession, payload: ExtractRunRequest) -> dict[str, Any]:
    """运行一次意图抽取实验，返回尝试轨迹与最终工单（views 调用的入口）。"""
    cfg = await resolve_model(db, payload.model_config_id)
    client = build_client(cfg)

    # ---------- 第一级（反面教材）：口头要求"请输出 JSON"，无约束、不重试 ----------
    if payload.mode == "naive":
        messages = [
            {"role": "system", "content": "你是信息提取助手，请以 JSON 格式输出结果。"},
            {"role": "user", "content": payload.text},
        ]
        resp = await client.chat.completions.create(
            model=cfg.model_name, messages=messages, temperature=0
        )
        content = resp.choices[0].message.content or ""
        entry: dict[str, Any] = {
            "attempt": 1,
            "raw_content": content,
            "parsed": None,
            "error": None,
            "response_format": "none",
            "note": None,
        }
        try:
            entry["parsed"] = json.loads(content)
            entry["note"] = (
                "这次运气好，模型没加废话。多跑几次就会碰到'好的，以下是 JSON：'——"
                "对 LLM 说'请输出 JSON'不是合同，是许愿"
            )
        except json.JSONDecodeError as exc:
            entry["error"] = f"JSONDecodeError: {exc}"
        return {
            "mode": payload.mode,
            "attempts": [entry],
            "ticket": entry["parsed"],
            "success": entry["parsed"] is not None,
            "model": cfg.model_name,
        }

    # ---------- 第二级：JSON Mode + 失败现场回填重试（只保证合法 JSON） ----------
    if payload.mode == "json_mode":
        messages = [
            {
                "role": "system",
                "content": (
                    "你是信息提取器。只输出一个 JSON 对象，不要任何解释、不要 markdown 代码块。"
                    "字段：name, phone, intent, urgency(1-5)。"
                ),
            },
            {"role": "user", "content": payload.text},
        ]

        async def fetch_json_mode(attempt: int) -> tuple[str, str, str | None]:
            # prompt 里必须出现"JSON"一词，否则 json_object 模式会被 API 拒绝（课程坑 2）
            resp = await client.chat.completions.create(
                model=cfg.model_name,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or "", "json_object", None

        obj, attempts = await retry_until_valid(
            messages, _validate_json_object, fetch_json_mode,
            max_retries=payload.max_retries,
        )
        return {
            "mode": payload.mode,
            "attempts": attempts,
            "ticket": obj,
            "success": obj is not None,
            "model": cfg.model_name,
        }

    # ---------- 第三级：Pydantic Schema 合同 + 结构化输出（生产标准做法） ----------
    messages = [
        {"role": "system", "content": SCHEMA_PROMPT},
        {"role": "user", "content": payload.text},
    ]
    obj, attempts = await retry_until_valid(
        messages,
        _validate_ticket,
        _make_json_schema_fetch(client, cfg.model_name, messages),
        max_retries=payload.max_retries,
    )
    return {
        "mode": payload.mode,
        "attempts": attempts,
        "ticket": obj.model_dump() if obj is not None else None,
        "success": obj is not None,
        "model": cfg.model_name,
    }
