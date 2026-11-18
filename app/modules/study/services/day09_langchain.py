"""Day 09 LangChain 初探：给手写管线换上工业零件。

五个实验文件的内容在这里收成一个模块，每个函数都对应课程 ``day09`` 的一个脚本：

    消息抽象   → ``messages_demo``      （裸 dict → 类型对象 + 双向转换）
    模板变量   → ``prompt_demo``        （ChatPromptTemplate + partial + 缺变量 fail-fast）
    结构化输出 → ``run_structured``     （with_structured_output 收编 Day 02 的三级进化）
    LCEL 管道  → ``run_lcel``           （invoke / stream / batch / with_retry / RunnableLambda）
    框架版循环 → ``run_agent``          （@tool + bind_tools + ToolMessage 回填 + MAX_STEPS）

三条贯穿全篇的原则：
1. 每学一个 API，先找到手写对应物（见 ``PART_MAPPING``）——背 API 换个框架就归零；
2. 框架替你做"零件标准化"（schema 生成、消息配对、重试），**不替你做"决策"**
   （终止条件、上下文预算）——这两件事仍然是 Handwritten 阶段的活；
3. langchain 全部延迟导入：没装依赖时接口报可读的 400，而不是让整个应用起不来。
"""

import asyncio
import time
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import (
    Day09AgentRequest,
    Day09CustomerReply,
    Day09LcelRequest,
    Day09PromptRequest,
    Day09StructuredRequest,
)
from app.modules.study.services.llm import openai_base_url, resolve_model

# 安装提示：未装依赖时所有 Day 09 接口都返回这句话
INSTALL_HINT = "pip install -U langchain langchain-openai"

# 课程 2.5 的对照表：手写零件 ↔ 框架零件（前端直接渲染这张表）
PART_MAPPING: list[dict[str, str]] = [
    {"handwritten": "手写 tools JSON Schema（Day 03）", "framework": "@tool 从 docstring 自动生成", "winner": "framework"},
    {"handwritten": "手写 dispatch 注册表（Day 03）", "framework": "tool.invoke(tool_call)", "winner": "framework"},
    {"handwritten": "手写 Pydantic 校验（Day 04）", "framework": "签名类型注解自动校验", "winner": "framework"},
    {"handwritten": "MAX_STEPS 终止条件（Day 05）", "framework": "自己写（框架不替你做决策）", "winner": "you"},
    {"handwritten": "消息配对校验（Day 08 自检 #5）", "framework": "类型层面部分兜底", "winner": "half"},
    {"handwritten": "上下文预算与裁剪（Day 08）", "framework": "自己写（LangChain 不管编排策略）", "winner": "you"},
]

# 裸 dict 与类型对象的差别（前端用两列对照展示）
DICT_VS_TYPED: list[dict[str, str]] = [
    {"point": "role 拼写", "raw": "拼成 assitant 不报错，直到模型行为异常才被发现", "typed": "role 由类决定，写错这个错误不存在"},
    {"point": "字段名", "raw": "tool_calls / tool_call_id 记错，运行到深处才炸", "typed": "构造 / 序列化时就报错"},
    {"point": "互转", "raw": "自己写映射", "typed": "convert_to_openai_messages 一行"},
]

# 结构化抽取的系统提示词（三字段 = Day 02 的 CustomerReply）
EXTRACT_SYSTEM = """你是客服质检员。阅读用户原话，产出结构化回执：
- reply：给用户的回复，一句话给出关键结论；
- sentiment：用户情绪，只能是 positive / neutral / negative；
- needs_escalation：是否需要转人工（true/false，涉及投诉、赔偿、监管风险时为 true）。
只依据用户原话，不要编造事实。"""

# 模型选错的示例问题（练习 2 用：一个问订单、一个问政策）
AGENT_PRESETS = [
    "订单 SO20261101001 发货了吗？",
    "XR-500 Pro 能退吗？",
    "我买的耳机坏了，能退吗？顺便帮我看看 SO20261101002 到哪了",
]


def _require_langchain() -> None:
    """依赖探测：没装 langchain 时给安装命令，而不是 ImportError 堆栈。"""
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 只在缺依赖的环境触发
        raise ValueError(f"未安装 langchain，请先执行：{INSTALL_HINT}") from exc


def build_chat_model(cfg: Any, temperature: float = 0.0, run_name: str = "study-day09"):
    """按「模型管理」里的配置构造 ChatOpenAI。

    换供应商只改这一行（ChatAnthropic / ChatDeepSeek 接口同构）——
    这正是课程 1.1 对照表里"多供应商适配"那一条的价值。
    ``max_retries=0``：把重试交给显式的 ``with_retry``，方便课堂观察。
    """
    _require_langchain()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=cfg.model_name,
        api_key=cfg.api_key or "EMPTY",
        base_url=openai_base_url(cfg.base_url),
        temperature=temperature,
        timeout=60,
        max_retries=0,
        tags=["study", run_name],
    )


def _message_text(message: Any) -> str:
    """取消息正文：兼容 str 与"内容块 list"两种返回形态。"""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


# ============================================================
# 1. 消息抽象：裸 dict → 类型对象
# ============================================================


def messages_demo() -> dict[str, Any]:
    """构造四种角色的类型化消息，并演示"role 拼错"在两种世界里的不同结局。"""
    _require_langchain()
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
        convert_to_openai_messages,
    )

    messages = [
        SystemMessage(content="你是客服 Agent，实时数据必须调工具查询。"),
        HumanMessage(content="XR-500 Pro 保修多久？"),
        AIMessage(content="整机保修两年 [来源: 产品手册]"),
        ToolMessage(content="order_status: 已发货", tool_call_id="t_1"),
    ]
    notes = {
        "system": "第 1 层身份与指令（Day 08）：角色 + 硬约束",
        "human": "用户输入；Day 03 起 history 里会出现它的多轮副本",
        "ai": "模型回复；带 tool_calls 时是本轮决策，content 可能为空",
        "tool": "工具结果；必须带 tool_call_id，否则 API 直接 400（Day 03 的坑）",
    }
    openai_roles = {"system": "system", "human": "user", "ai": "assistant", "tool": "tool"}

    rows = [
        {
            "type": message.type,
            "openai_role": openai_roles.get(message.type, message.type),
            "content": _message_text(message),
            "note": notes.get(message.type, ""),
        }
        for message in messages
    ]

    # 反例：裸 dict 拼错 role 静默通过（直到调用 API 才炸）
    raw_typo = {"role": "assitant", "content": "保修两年"}
    raw_typo_note = "裸 dict：{} 能顺利构造，错误被推迟到调用模型时才暴露".format(raw_typo)

    # 正例：必填字段缺失在构造时就炸——ToolMessage 少了 tool_call_id 正是 Day 03 的配对坑
    typed_error = ""
    try:
        ToolMessage(content="order_status: 已发货")  # type: ignore[call-arg]
    except Exception as exc:  # noqa: BLE001  不同版本是 KeyError 或 ValidationError
        typed_error = f"{type(exc).__name__}: {exc}"

    # 反例补一刀：pydantic v2 默认 extra=ignore，多传的未知 kwargs 会被静默忽略，
    # 所以"类型对象防拼错"靠的是**必填字段校验 + IDE 类型提示**，不是"任何拼错都报错"。
    extra_ignored = ""
    try:
        HumanMessage(content="保修两年", role="assitant")  # type: ignore[call-arg]
        extra_ignored = "HumanMessage 接受了未知参数 role（pydantic extra=ignore），说明多传不会报错"
    except TypeError as exc:
        extra_ignored = f"多传参数也会报错：{type(exc).__name__}: {exc}"

    return {
        "messages": rows,
        "openai_payload": convert_to_openai_messages(messages),
        "roundtrip_note": "convert_to_openai_messages 的输出与 Day 01 手写的 messages 完全同构——框架只是包了一层类型",
        "dict_vs_typed": DICT_VS_TYPED,
        "raw_typo": {"payload": raw_typo, "note": raw_typo_note},
        "typed_error": typed_error or "（没有触发校验错误，请检查 langchain 版本）",
        "extra_ignored": extra_ignored,
        "part_mapping": PART_MAPPING,
    }


# ============================================================
# 2. Prompt 模板：变量、partial、fail-fast
# ============================================================


def prompt_demo(payload: Day09PromptRequest) -> dict[str, Any]:
    """ChatPromptTemplate 演示：变量占位、partial 冻结、漏填变量立即 KeyError。"""
    _require_langchain()
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是{company}的客服 Agent。\n"
                "回答规则：\n"
                "1. 优先使用【检索到的资料】，标注来源。\n"
                "2. 实时数据（订单/库存）一律调用工具，禁止编造。",
            ),
            ("placeholder", "{history}"),
            ("human", "{question}"),
        ]
    )
    # partial 冻结：多租户场景里公司名固定，问题每轮变
    frozen = prompt.partial(company=payload.company or "极星科技")

    history = [
        HumanMessage(content=str(item.get("content", "")))
        if item.get("role") == "user"
        else AIMessage(content=str(item.get("content", "")))
        for item in payload.history
    ]

    error = ""
    if payload.omit_variable:
        # 故意漏填必需变量 question：裸字符串时代这里只会"少一段话"，没人报错。
        # 注意 placeholder 变量（history）是**可选**的——缺失就跳过，这是 placeholder 的设计，
        # 因为"没有历史"是合法状态；真正会 fail-fast 的是普通占位符。
        try:
            frozen.invoke({"history": history})
        except KeyError as exc:
            error = f"KeyError: {exc}"

    filled = frozen.invoke({"history": history, "question": payload.question})
    messages = filled.to_messages()

    # 给每条消息标明"它由模板哪一段产生"，方便肉眼对上课程代码
    sources = ["模板 system（角色 + 规则）"]
    sources += ["placeholder 占位的对话历史"] * len(history)
    sources += ["模板 human（当前问题）"]
    rows = [
        {
            "type": message.type,
            "content": _message_text(message),
            "source": sources[index] if index < len(sources) else "模板",
        }
        for index, message in enumerate(messages)
    ]

    return {
        "input_variables": sorted(prompt.input_variables),
        "optional_variables": ["history"],
        "partial_frozen": ["company"],
        "messages": rows,
        "error": error or None,
        "error_hint": (
            "必需变量漏填 → invoke 阶段直接 KeyError，而不是静默拼出一个残缺 prompt；"
            "placeholder 变量（history）例外：它是可选的，缺失即跳过——"
            "『没有历史』本身是合法状态，这一条和课程原文略有出入（框架版本行为）"
            if error
            else "本次变量齐全，invoke 成功（placeholder 的 history 缺了也不会报错）"
        ),
        "part_mapping": PART_MAPPING,
    }


# ============================================================
# 3. 结构化输出：with_structured_output
# ============================================================


# with_structured_output 的三种下发方式：default（框架自挑，多数供应商走 json_schema）
# / function_calling（工具调用，兼容性最好）/ json_mode（模型必须支持 JSON Mode）
_STRUCTURED_METHODS = ("default", "function_calling", "json_mode")


def _structured_runner(llm: Any, schema: Any, method: str) -> Any:
    """按指定方式构造结构化输出 Runnable。"""
    if method == "default":
        return llm.with_structured_output(schema)
    return llm.with_structured_output(schema, method=method)


def _structured_messages(text: str, method: str) -> list[Any]:
    """按方式组装输入消息。

    json_mode 有硬性要求：prompt 里必须出现 "json" 字样，否则供应商直接 400
    （``Prompt must contain the word 'json'``）。这条要求藏在供应商文档里，
    封装层不会替你加，所以降级路径上要自己补一句字段说明。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages: list[Any] = [SystemMessage(content=EXTRACT_SYSTEM)]
    if method == "json_mode":
        messages.append(
            SystemMessage(
                content=(
                    "以 JSON 对象输出，字段：reply（字符串）、sentiment（positive/neutral/negative）、"
                    "needs_escalation（布尔）。只输出 JSON，不要解释。"
                )
            )
        )
    messages.append(HumanMessage(content=text))
    return messages


async def run_structured(db: AsyncSession, payload: Day09StructuredRequest) -> dict[str, Any]:
    """用 ``with_structured_output`` 抽取三字段回执；可重复跑同一输入观察抖动。

    自动降级：供应商不支持 json_schema 时（如部分 OpenAI 兼容服务返回
    ``This response_format type is unavailable``），依次回退 function_calling → json_mode，
    并把失败现场记进轨迹——这正是课程对照表里"strict 模式适配各家 API"那一行。
    """
    _require_langchain()
    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=payload.temperature, run_name="structured")

    order = list(_STRUCTURED_METHODS) if payload.method == "auto" else [payload.method]
    locked: str | None = None
    attempts: list[dict[str, Any]] = []
    first_trace: list[str] = []
    started = time.perf_counter()

    for index in range(1, payload.repeat + 1):
        begin = time.perf_counter()
        parsed: dict[str, Any] | None = None
        method_trace: list[str] = []
        used = ""
        for method in [locked] if locked else order:
            try:
                runner = _structured_runner(llm, Day09CustomerReply, method)
                result = await runner.ainvoke(_structured_messages(payload.text, method))
                parsed = result.model_dump()
                used = method
                break
            except Exception as exc:  # noqa: BLE001  降级失败也是课堂素材
                method_trace.append(f"{method} ✗ {type(exc).__name__}: {str(exc)[:200]}")
        if index == 1:
            first_trace = method_trace
        locked = locked or used or None
        attempts.append(
            {
                "index": index,
                "reply": str((parsed or {}).get("reply", "")),
                "sentiment": str((parsed or {}).get("sentiment", "")),
                "needs_escalation": bool((parsed or {}).get("needs_escalation", False)),
                "method": used or "全部失败",
                "methods_tried": method_trace,
                "elapsed_ms": int((time.perf_counter() - begin) * 1000),
                "error": None if parsed is not None else (method_trace[-1] if method_trace else "未知错误"),
            }
        )

    unstable = sorted(
        {
            field
            for field in ("reply", "sentiment", "needs_escalation")
            if len({str(item[field]) for item in attempts}) > 1
        }
    )
    if unstable and payload.temperature > 0:
        note = "temperature 越高，同一输入的字段越容易抖——结构化输出请把 temperature 设 0"
    elif locked and locked != "default":
        note = (
            f"默认方式不可用，已自动降级到 {locked}：不同供应商支持的 response_format 不一样，"
            "with_structured_output 的 method 参数就是适配开关"
        )
    else:
        note = "Field(description=) 就是下发给模型的说明书，写得越具体抽取越稳（与 Day 04 工具 schema 同理）"
    return {
        "schema_sent": Day09CustomerReply.model_json_schema(),
        "attempts": attempts,
        "unstable_fields": unstable,
        "method_used": locked or "全部失败",
        "methods_tried": first_trace,
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "note": note,
        "part_mapping": PART_MAPPING,
    }


# ============================================================
# 4. LCEL 管道：一个竖线带来的横切能力
# ============================================================


CHAIN_STEPS = [
    {"index": 1, "name": "ChatPromptTemplate", "role": "变量填充 → PromptValue"},
    {"index": 2, "name": "ChatOpenAI", "role": "发给模型 → AIMessage"},
    {"index": 3, "name": "StrOutputParser", "role": "AIMessage → str"},
]


async def run_lcel(db: AsyncSession, payload: Day09LcelRequest) -> dict[str, Any]:
    """按 mode 演示 invoke / stream / batch / with_retry / RunnableLambda 五种用法。"""
    _require_langchain()
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    mode = payload.mode
    chain_steps = list(CHAIN_STEPS)

    # with_retry 是纯本地演示：不花 token，也能看清指数退避的包装层
    if mode == "retry":
        attempts: list[dict[str, Any]] = []
        state = {"count": 0}

        def flaky(text: str) -> str:
            """前两次模拟上游 502，第三次成功——观察 with_retry 的重试轨迹。"""
            state["count"] += 1
            attempts.append({"attempt": state["count"], "ok": state["count"] >= 3, "text": text})
            if state["count"] < 3:
                raise RuntimeError("模拟上游 502 Bad Gateway")
            return f"{text} → 第 3 次终于成功，框架替你指数退避"

        safe = RunnableLambda(flaky).with_retry(stop_after_attempt=3)
        begin = time.perf_counter()
        answer = safe.invoke("订单查询")
        retry_ms = int((time.perf_counter() - begin) * 1000)
        return {
            "mode": mode,
            "chain": chain_steps,
            "answer": answer,
            "chunks": [],
            "chunk_count": 0,
            "elapsed_ms": retry_ms,
            "batch": [],
            "batch_ms": None,
            "serial_ms": None,
            "retry_trace": attempts,
            "note": "with_retry 挂在任意 Runnable 上：重试、退避、最终失败都由框架处理，业务函数只写正常路径",
            "part_mapping": PART_MAPPING,
        }

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, run_name=f"lcel-{mode}")
    prompt = ChatPromptTemplate.from_template("用一句话回答：{question}")
    parser = StrOutputParser()
    chain = prompt | llm | parser

    answer = ""
    chunks: list[str] = []
    batch: list[dict[str, Any]] = []
    batch_ms: int | None = None
    serial_ms: int | None = None
    started = time.perf_counter()

    if mode == "invoke":
        answer = await chain.ainvoke({"question": payload.question})
    elif mode == "stream":
        # 手写 Day 01 时要自己处理 delta 拼接，这里一行拿完所有分片
        async for chunk in chain.astream({"question": payload.question}):
            if chunk:
                chunks.append(chunk)
        answer = "".join(chunks)
    elif mode == "batch":
        questions = payload.batch_questions or [
            "什么是短期记忆？",
            "什么是长期记忆？",
            "什么是上下文工程？",
        ]
        begin = time.perf_counter()
        answers = await chain.abatch([{"question": item} for item in questions])
        batch_ms = int((time.perf_counter() - begin) * 1000)
        batch = [
            {"question": question, "answer": text}
            for question, text in zip(questions, answers)
        ]
        if payload.compare_serial:
            begin = time.perf_counter()
            for item in questions:
                await chain.ainvoke({"question": item})
            serial_ms = int((time.perf_counter() - begin) * 1000)
        answer = batch[0]["answer"] if batch else ""
    elif mode == "lambda":
        # 管道中途插手写逻辑：普通函数必须用 RunnableLambda 包一层（坑 5）
        tag_source = RunnableLambda(lambda text: f"{text}\n[来源: LCEL 管道]")
        full_chain = chain | tag_source
        chain_steps = [
            *CHAIN_STEPS,
            {"index": 4, "name": "RunnableLambda", "role": "手写后处理（非 Runnable 要包一层）"},
        ]
        answer = await full_chain.ainvoke({"question": payload.question})

    notes = {
        "invoke": "单次调用：prompt | llm | parser，一个竖线就是全部语法",
        "stream": "stream 自动处理 delta 拼接：手写版要自己维护 buffer 与 flush",
        "batch": "batch 并发执行：三个问题 ≈ 单问耗时，而不是三倍（生产里批量评估就靠它）",
        "lambda": "管道里插手写函数要用 RunnableLambda，直接塞普通函数会报类型错",
    }
    return {
        "mode": mode,
        "chain": chain_steps,
        "answer": answer,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "batch": batch,
        "batch_ms": batch_ms,
        "serial_ms": serial_ms,
        "retry_trace": [],
        "note": notes.get(mode, ""),
        "part_mapping": PART_MAPPING,
    }


# ============================================================
# 5. 框架版 Agent 循环：@tool + bind_tools + ToolMessage
# ============================================================


def _build_agent_tools(variant: str) -> list[Any]:
    """按变体构造工具：full（详细 docstring）/ vague（模糊 docstring）/ missing（无 docstring）。"""
    _require_langchain()
    from typing import Annotated

    from langchain_core.tools import tool
    from pydantic import Field

    if variant == "missing":
        # 故意不给 docstring：框架直接 fail-fast，比"模型乱调"更容易排查
        @tool
        def get_order_status(order_id: str) -> str:  # noqa: D401
            return "已发货"

        return [get_order_status]

    if variant == "vague":

        @tool
        def get_order_status(order_id: str) -> str:
            """查询。"""
            return _order_status(order_id)

        @tool
        def get_return_policy() -> str:
            """查询。"""
            return _return_policy()

        return [get_order_status, get_return_policy]

    @tool(parse_docstring=True)
    def get_order_status(
        order_id: Annotated[
            str, Field(description="订单号，必须以 SO 开头，如 SO20261101001")
        ],
    ) -> str:
        """查询订单的物流与发货状态。

        何时用：用户问"发货了吗""到哪了""物流状态"这类实时事实时。
        何时不用：用户问的是退款政策、保修期限等静态规则时，直接查政策工具。
        返回：订单号 + 当前状态（已发货/待发货/已签收）。

        Args:
            order_id: 订单号，必须以 SO 开头，如 SO20261101001。
        """
        return _order_status(order_id)

    @tool
    def get_return_policy() -> str:
        """查询退款政策（静态规则，不需要订单号）。

        何时用：用户问"能退吗""退款要运费吗""几天内可退"这类规则问题时。
        何时不用：用户问的是某个订单的具体状态时，用 get_order_status。
        返回：退款期限 + 运费承担方。
        """
        return _return_policy()

    return [get_order_status, get_return_policy]


def _order_status(order_id: str) -> str:
    """假订单数据：真实工程里这里是数据库/HTTP 调用。"""
    if not order_id.startswith("SO"):
        return "错误: 订单号必须以 SO 开头（错误信息即 prompt，Day 04）"
    return f"{order_id}: 已发货（今日 09:12 查询）"


def _return_policy() -> str:
    """假政策数据：与 Day 07 知识库里的条款保持一致。"""
    return "自签收之日起 30 天内可无理由退款，运费 10 元由买家承担；年卡会员免运费。"


async def run_agent(db: AsyncSession, payload: Day09AgentRequest) -> dict[str, Any]:
    """用 LangChain 消息对象重写 Day 01 的最小循环（结构完全同构）。"""
    _require_langchain()
    from langchain_core.messages import HumanMessage, ToolMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, run_name="agent-loop")

    try:
        tools = _build_agent_tools(payload.variant)
    except ValueError as exc:
        # variant=missing 时 @tool 直接报错：把这条 fail-fast 当作结论返回
        return {
            "steps": [],
            "final_answer": "",
            "stop_reason": "tool_schema_error",
            "tool_schemas": [],
            "variant_note": f"框架在定义工具时就拦住了：{exc}。"
            "这比'模型乱调工具'好排查——description 就是模型的说明书（Day 04）",
            "messages_count": 0,
            "model": cfg.model_name,
            "elapsed_ms": 0,
        }

    llm_with_tools = llm.bind_tools(tools)
    # bind_tools 真实下发的 schema：观察 description 权重的最好入口
    tool_schemas = llm_with_tools.kwargs.get("tools", [])

    messages: list[Any] = [HumanMessage(content=payload.question)]
    steps: list[dict[str, Any]] = []
    final_answer = ""
    stop_reason = "max_steps"
    started = time.perf_counter()

    for step in range(1, payload.max_steps + 1):
        ai_message = await llm_with_tools.ainvoke(messages)
        messages.append(ai_message)
        tool_calls = list(getattr(ai_message, "tool_calls", []) or [])

        if not tool_calls:  # 无工具调用 = 语义终止（Day 05 第 1 层）
            final_answer = _message_text(ai_message)
            steps.append({"step": step, "tool_calls": [], "content": final_answer})
            stop_reason = "completed"
            break

        traces: list[dict[str, Any]] = []
        for call in tool_calls:  # 并行工具调用逐个回填
            tool_obj = next((item for item in tools if item.name == call["name"]), None)
            if tool_obj is None:
                content, is_error = f"未知工具 {call['name']}（模型幻觉）", True
            else:
                try:
                    result = tool_obj.invoke(call)  # 传 tool_call dict 即可：本地执行
                    content = _message_text(result) if hasattr(result, "content") else str(result)
                    is_error = getattr(result, "status", "success") == "error"
                except Exception as exc:  # noqa: BLE001
                    content, is_error = f"工具执行失败：{exc}", True
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
            traces.append(
                {
                    "id": call["id"],
                    "name": call["name"],
                    "args": call.get("args", {}),
                    "result": content,
                    "is_error": is_error,
                }
            )
        steps.append({"step": step, "tool_calls": traces, "content": None})

    if stop_reason == "max_steps":
        # 框架不装保险丝：跑满上限必须由我们自己给一个降级答复
        final_answer = (
            f"已达 {payload.max_steps} 步上限（LangChain 不自带终止条件，"
            "MAX_STEPS 是 Day 05 的活）。基于已有信息：信息不足，未能确认。"
        )

    variant_notes = {
        "full": "description 写全（做什么/何时用/何时不用/返回什么），模型才选得准",
        "vague": "description 只剩『查询。』——观察模型是否还分得清两个工具（练习 2）",
        "missing": "无 docstring 的变体在定义工具时就会报错",
    }
    return {
        "steps": steps,
        "final_answer": final_answer,
        "stop_reason": stop_reason,
        "tool_schemas": tool_schemas,
        "variant_note": variant_notes.get(payload.variant, ""),
        "messages_count": len(messages),
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


async def mapping() -> dict[str, Any]:
    """Day 09 的概念对照表（静态数据，供前端渲染）。"""
    return {
        "part_mapping": PART_MAPPING,
        "dict_vs_typed": DICT_VS_TYPED,
        "agent_presets": AGENT_PRESETS,
        "install_hint": INSTALL_HINT,
        "langchain_version": await asyncio.to_thread(_version),
    }


def _version() -> str:
    """读已安装版本：版本差异是框架篇最常见的坑。"""
    try:
        from importlib.metadata import version

        return version("langchain-core")
    except Exception:  # noqa: BLE001  没装就是没装，前端照常展示
        return "未安装"
