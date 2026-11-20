"""Day 13 流式输出：把 Agent 的思考过程变成可见事件。

课程 day13 的五个脚本在这里收成一个模块：

    stream_mode 四模式对照   → ``run_modes``            （01：同一张图分别按 values / updates / messages 订阅）
    多 Agent token 分流      → ``run_token_stream``     （02：按 ``metadata["langgraph_node"]`` 把 token 路由到各自气泡）
    custom 工具进度上报      → ``run_custom_stream``    （03：工具内 ``get_stream_writer()`` 上报进度）
    interrupt + 流式同开     → ``run_interrupt_stream`` （04：只订 messages 会漏掉 ``__interrupt__`` 这个坑）
    生产最后一公里           → ``sse_chat``             （05：LangGraph 事件 → SSE 帧 → 浏览器 EventSource）

贯穿全篇的四条结论：

1. 流的本质是**分层**的：一个超步里可以同时有 values（状态级）、updates（节点级）、
   messages（token 级）、custom（自定义）四层事件，``stream_mode`` 传列表就能一次订多层；
2. 多 Agent 场景下 ``metadata["langgraph_node"]`` 是唯一能分清「谁在说话」的字段；
   主管的结构化输出 token 是 JSON 片段，必须过滤掉，绝不能给用户看；
3. ``interrupt()`` 的暂停信号**只出现在 updates 流的 ``__interrupt__`` 键里**，
   只订 messages 的前端会永远等不到审批框——这是今天的主坑；
4. 生产三件套：帧尾 ``\\n\\n`` + ``ensure_ascii=False`` + ``X-Accel-Buffering: no``，
   事件协议固定成 token / interrupt / node_start / done 四类，前端按 ``type`` 分发。

两个与课程代码不同的工程决定：

- 课程 02/03 直接 ``llm.invoke`` 在节点里产生 token；本模块同样如此（不走专家子图），
  因为一旦 token 产生在子图内部，``langgraph_node`` 会变成子图里的 ``model``，
  "按专家分流"这个教学点就演示不出来了；
- 课程 05 的 SSE 接口里 ``thread_id`` 是为了续跑；本模块的 SSE 只做单轮打字机演示，
  模型在**返回响应之前**解析好再传给生成器（否则请求级 DB 会话在流开始时就已关闭）。

langgraph / langchain 全部延迟导入：缺依赖时接口返回可读的 400，而不是让应用起不来。
"""

import asyncio
import json
import operator
import re
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import (
    Day13CustomRequest,
    Day13InterruptRequest,
    Day13ModesRequest,
    Day13TokenRequest,
)
from app.modules.study.services.day09_langchain import build_chat_model
from app.modules.study.services.day10_langgraph import _graph_structure, _jsonable, _text
from app.modules.study.services.day11_checkpointer import _last_answer
from app.modules.study.services.llm import resolve_model

INSTALL_HINT = "pip install -U langgraph langchain-openai"

# ============================================================
# 静态概念数据（供前端渲染，全部来自课程原文）
# ============================================================

# 课程 2.1：stream_mode 四值全解（外加 debug 的说明）
STREAM_MODES: list[dict[str, str]] = [
    {
        "mode": "values",
        "emits": "每个超步结束后的完整 state",
        "granularity": "状态级",
        "usage": "调试：看状态怎么一步步长出来（第一条是初始 state）",
        "shape": "state_dict",
    },
    {
        "mode": "updates",
        "emits": "每个节点返回的增量（含 __interrupt__）",
        "granularity": "节点级",
        "usage": "前端显示『当前在哪个节点、干了什么』+ 探测暂停",
        "shape": "{node_name: update_dict}",
    },
    {
        "mode": "messages",
        "emits": "LLM 产生的 token 块",
        "granularity": "token 级",
        "usage": "打字机效果；meta.langgraph_node 决定流进哪个气泡",
        "shape": "(AIMessageChunk, metadata)",
    },
    {
        "mode": "custom",
        "emits": "节点内 get_stream_writer() 手动上报的任意数据",
        "granularity": "自定义",
        "usage": "工具进度条、检索『查到 5 条』",
        "shape": "writer() 写什么就是什么",
    },
    {
        "mode": "debug",
        "emits": "最详细的内部事件（任务调度、重试）",
        "granularity": "框架级",
        "usage": "只用于排查框架本身的问题，日常不用",
        "shape": "内部事件 dict",
    },
]

# 课程 2.3：三个 API 怎么选
API_CHOICES: list[dict[str, str]] = [
    {
        "api": "graph.invoke()",
        "job": "一次拿全部结果",
        "when": "后台批处理、测试",
        "note": "用户盯着转圈，感知延迟 = 真实延迟",
    },
    {
        "api": "graph.stream(stream_mode=...)",
        "job": "按模式吐事件",
        "when": "生产主力，大多数场景够用",
        "note": "单模式 for chunk；多模式 for (mode, chunk)，解包方式必须跟着改",
    },
    {
        "api": "graph.astream_events(v=2)",
        "job": "最细粒度事件流（含 on_chat_model_start 等回调）",
        "when": "要统计每次 LLM 调用的首 token 延迟时",
        "note": "初学先用 stream 解决 90% 的需求",
    },
]

# 课程 3.4 + 4：interrupt 与流式同开的机理与三种探测方式
INTERRUPT_RULES: list[dict[str, str]] = [
    {
        "point": "为什么 messages 流里没有暂停信号",
        "detail": "interrupt() 的实现是抛出一个特殊暂停信号，图在该超步挂起，节点没有返回值；"
        "messages 流只承载 LLM token，暂停不是消息",
    },
    {
        "point": "暂停信号在哪",
        "detail": "作为特殊键 __interrupt__ 塞进 updates 流。标配 stream_mode=['messages','updates']，"
        "并在 updates 里检查 '__interrupt__'",
    },
    {
        "point": "另外两种探测方式",
        "detail": "astream_events v2 里是 on_interrupt 事件；graph.get_state(config) 里体现为 "
        ".next 非空 + .tasks 中的 interrupted 标记——三种任选其一，但别只用 token 流硬等",
    },
    {
        "point": "续跑语义不变",
        "detail": "暂停后仍然用 Command(resume=...) 续跑（Day 11）；interrupt 之前的零副作用铁律照旧",
    },
]

# 课程 3.5：SSE 三件套 + 事件协议
SSE_RULES: list[dict[str, str]] = [
    {
        "rule": "帧尾必须 \\n\\n",
        "why": "SSE 协议用空行分隔事件块；少了空行浏览器会一直等待",
        "how": "f'data: {json.dumps(...)}\\n\\n'",
    },
    {
        "rule": "ensure_ascii=False",
        "why": "否则中文被转成 \\uXXXX，前端还得再解码一遍",
        "how": "json.dumps(data, ensure_ascii=False)",
    },
    {
        "rule": "X-Accel-Buffering: no",
        "why": "Nginx 默认 proxy_buffering on，会把 SSE 帧攒满 buffer 才转发，流式变假流式",
        "how": "响应头加这一项（或网关配 proxy_buffering off）",
    },
    {
        "rule": "协议先行",
        "why": "别让前端解析 LangGraph 原始结构，框架升级就全崩",
        "how": "固定四类事件：token / interrupt / node_start / done（外加 error）",
    },
]

EVENT_PROTOCOL: list[dict[str, str]] = [
    {"type": "token", "payload": "node + content", "front": "追加到对应节点的气泡（打字机）"},
    {"type": "node_start", "payload": "node", "front": "状态栏显示『researcher 检索中…』"},
    {"type": "interrupt", "payload": "interrupt 的 value", "front": "弹审批框，把决定回传给 Command(resume=)"},
    {"type": "done", "payload": "chars + first_token_ms", "front": "关掉 EventSource，顺便上报首 token 延迟"},
    {"type": "error", "payload": "message", "front": "HTTP 状态码已是 200，只能靠事件体告知失败"},
]

# 课程第 5 节：五个常见坑
PITFALLS: list[dict[str, str]] = [
    {
        "no": "1",
        "title": "只订 messages，错过 interrupt",
        "symptom": "图悄悄停住，前端还在等 token，审批框永远弹不出来",
        "fix": "stream_mode=['messages','updates']，两路都检查；updates 里认 __interrupt__ 键",
    },
    {
        "no": "2",
        "title": "把主管的结构化 token 给用户看",
        "symptom": "屏幕上出现 {\"next\": \"resea 这种 JSON 碎片",
        "fix": "按 metadata['langgraph_node'] 过滤，或判断 content 是否自然语言",
    },
    {
        "no": "3",
        "title": "custom 流静默失效",
        "symptom": "writer() 调了但前端什么都没收到，也不报错",
        "fix": "get_stream_writer() 必须在图执行上下文内（节点函数 / 被图调用的工具）调用",
    },
    {
        "no": "4",
        "title": "多模式时忘了事件是二元组",
        "symptom": "ValueError 或拿错数据（把 mode 当成 chunk）",
        "fix": "单模式 for chunk；多模式 for (mode, chunk)，模式改成列表时解包要跟着改",
    },
    {
        "no": "5",
        "title": "网关缓冲杀死流式",
        "symptom": "本地逐字出字，上网关后 8 秒不出字然后整段砸出",
        "fix": "X-Accel-Buffering: no；把首 token 延迟做成监控指标，超过 2 秒告警",
    },
]

# 课程第 8 节：今日必掌握
CHECKLIST: list[str] = [
    "说清感知延迟与真实延迟的区别，知道流式是上线必备项",
    "四种 stream_mode 各自吐什么、什么粒度、拿什么活",
    "一次订阅多层：for mode, payload in graph.stream(..., stream_mode=[...])",
    "用 metadata['langgraph_node'] 把 token 分流到各专家气泡，并过滤结构化输出碎片",
    "用 get_stream_writer() 在工具内上报进度，payload 固定成 {type, stage, data}",
    "知道 interrupt 只在 updates 流的 __interrupt__ 里，标配两路一起订",
    "能写出带 SSE 三件套的服务端，并定义四类事件协议给前端消费",
]

# 各实验的预设（前端一键填入）
PRESETS: dict[str, list[str]] = {
    "modes": [
        "用一句话解释什么是 RAG",
        "向量检索和 BM25 有什么区别？",
    ],
    "token": [
        "对比一下向量检索和 BM25 的优劣",
        "多智能体为什么要共享账本而不是共享全部消息？",
    ],
    "custom": [
        "LangGraph 的断点续跑是怎么实现的？",
        "LangGraph 的 Checkpointer 有什么用？",
    ],
    "sse": [
        "用三句话介绍一下 SSE 流式输出",
        "为什么 Agent 产品必须做流式？",
    ],
}

# 课程 3.1：对照实验用的主管（只输出 json，token 会给前端制造"乱码"）
PROBE_SUPERVISOR_PROMPT = """你是调度员，只决定下一步交给谁，不回答用户问题。
只输出一个 json 对象，形如 {"next": "expert"}：
- 需要查资料 / 需要解释概念 → expert
- 完全不需要处理 → FINISH
不要输出任何解释文字。"""

# 课程 3.2：多 Agent token 分流用的主管（Day 12 的三角色简化版）
SUPERVISOR_PROMPT = """你是团队主管，只负责派活，不回答用户。你本人不使用任何工具。

专家名单：
- researcher：查资料、列要点。不写结论性文案。
- writer：把已有要点整理成一句结论。不查资料、不新增事实。

只输出一个 json 对象，形如 {"next": "researcher"}，next 只能是 researcher / writer / FINISH。
不要输出任何解释文字。"""

RESEARCHER_PROMPT = "你是资料研究员。只做检索与事实核对，用 2 个要点回答，不写对客话术。"

WRITER_PROMPT = "你是撰稿人。把对话里已有的要点整理成一句话结论，不新增事实。"

# 课程 3.3：文档助手的系统提示词
DOC_SYSTEM = """你是文档助手。回答前必须先调用 search_docs 检索文档库，只依据检索结果回答。
检索不到就直说没找到，不要凭记忆编造。"""

# 课程 3.3 的假文档库（工具内部进度上报的实验对象）
FAKE_DOCS: dict[str, str] = {
    "doc1": "LangGraph 是基于状态图的多 Agent 编排框架，节点读写共享 state。",
    "doc2": "LangGraph 的 Checkpointer 支持断点续跑与人机协同，一个超步存一帧。",
    "doc3": "MCP 协议统一了工具接入标准，把 N×M 的适配问题降成 N+M。",
}

# 检索工具的模拟耗时（每命中一条 sleep 一次，用来让前端看见进度条在动）
SEARCH_STEP_SECONDS = 0.5

# 退款演示的固定金额兜底
DEFAULT_REFUND_AMOUNT = 199.0


# ============================================================
# 0. 公共设施：依赖探测 / 输入构造 / 事件投影 / SSE 帧
# ============================================================


def _require_langgraph() -> None:
    """依赖探测：没装 langgraph 时给安装命令，而不是 ImportError 堆栈。"""
    try:
        import langgraph.graph  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 只在缺依赖的环境触发
        raise ValueError(f"未安装 langgraph，请先执行：{INSTALL_HINT}") from exc


def _version(name: str = "langgraph") -> str:
    """读已安装包版本：框架篇的坑大半和版本有关。"""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001  没装就是没装，前端照常展示
        return "未安装"


def _thread_id(prefix: str) -> str:
    """生成一次性的演示会话 ID（生产应从业务键派生）。"""
    return f"{prefix}-{uuid4().hex[:6]}"


def _elapsed_ms(started: float) -> int:
    """从起点到现在的毫秒数（事件时间戳全靠它，前端能看到"卡在第几秒"）。"""
    return int((time.perf_counter() - started) * 1000)


def _is_ai_chunk(chunk: Any) -> bool:
    """只认 LLM 产出的块：ToolMessage 也会出现在 messages 流里，不能当 token 处理。"""
    return getattr(chunk, "type", "") in {"AIMessageChunk", "ai", "AIMessage"}


def _should_show_token(node: str, content: str) -> bool:
    """决定这个 token 该不该进用户气泡。

    两条规则：
    1. 主管节点的 token 是 ``with_structured_output`` / JSON 片段，给用户看就是乱码；
    2. 空白块（部分网关会吐空 content）直接丢掉，免得前端渲染出空气泡。
    """
    if node == "supervisor":
        return False
    return bool((content or "").strip())


def _short(value: Any, limit: int = 80) -> str:
    """把任意值压成一行的短摘要（事件表里一格放得下）。"""
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value)
    else:
        try:
            text = json.dumps(_jsonable(value), ensure_ascii=False)
        except Exception:  # noqa: BLE001  兜底：真序列化不了就 str
            text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def _update_digest(update: Any) -> str:
    """把节点补丁压成一句人话（前端『updates 看节点干了什么』那一列）。"""
    if not isinstance(update, dict):
        return _short(update)
    parts: list[str] = []
    for key, value in update.items():
        if key == "messages" and isinstance(value, list):
            parts.append(f"messages +{len(value)}")
        elif key in {"findings", "events", "refunds", "dispatches"} and isinstance(value, list):
            parts.append(f"{key} +{len(value)}")
        else:
            parts.append(f"{key}={_short(value, 50)}")
    return "；".join(parts) or "（空补丁）"


def _first_human_text(messages: list[Any]) -> str:
    """取第一条 HumanMessage 的正文（用户问题），找不到就空串。"""
    for message in messages or []:
        if getattr(message, "type", "") == "human":
            return _text(message)
    return ""


def _parse_route(text: str) -> str:
    """从模型输出里抠 next 字段。

    容忍 ```json 围栏、前后解释文字、全角引号——thinking 模型经常不老实，
    解析失败返回空串由调用方兜底（这里不抛异常，流式过程里抛异常代价太大）。
    """
    raw = (text or "").replace("“", '"').replace("”", '"').replace("：", ":")
    match = re.search(r'"next"\s*:\s*"([A-Za-z]+)"', raw)
    if match:
        return match.group(1)
    for candidate in ("researcher", "writer", "expert", "FINISH"):
        if candidate in raw:
            return candidate
    return ""


def _normalize_decision(value: Any) -> tuple[bool, float | None, str]:
    """把 resume 值归一成 (是否放行, 人工改的金额, 备注)。

    三种形态与 Day 11/12 完全一致：``True`` 放行 / ``False`` 驳回 /
    ``{"approved": True, "amount": 150, "note": "..."}`` 改参数后放行。
    """
    if isinstance(value, dict):
        approved = bool(value.get("approved", True))
        raw_amount = value.get("amount")
        amount = float(raw_amount) if raw_amount is not None else None
        return approved, amount, str(value.get("note") or "")
    return bool(value), None, ""


def sse(data: dict[str, Any]) -> str:
    """拼一个 SSE 帧：``data: <json>\\n\\n``。

    两个协议细节都在这一行里：帧尾两个换行（否则浏览器一直等）、
    ``ensure_ascii=False``（否则中文变 \\uXXXX）。
    """
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def sse_headers() -> dict[str, str]:
    """SSE 响应头三件套（``X-Accel-Buffering`` 是给 Nginx 看的）。"""
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def _group_bubbles(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把逐 token 记录按节点聚合成气泡：节点一变就开新泡。

    "每个专家一个气泡、token 只流进自己的气泡"——就是这一步在数据层的形态。
    """
    bubbles: list[dict[str, Any]] = []
    for token in tokens:
        node = str(token.get("node") or "?")
        if not bubbles or bubbles[-1]["node"] != node:
            bubbles.append(
                {
                    "node": node,
                    "text": "",
                    "tokens": 0,
                    "first_offset_ms": int(token.get("offset_ms") or 0),
                    "last_offset_ms": int(token.get("offset_ms") or 0),
                }
            )
        bubble = bubbles[-1]
        bubble["text"] += str(token.get("content") or "")
        bubble["tokens"] += 1
        bubble["last_offset_ms"] = int(token.get("offset_ms") or 0)
    return bubbles


def _collect_tokens(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把 token 行收成"气泡 + 每节点耗时 + 首 token 延迟"三件套。"""
    bubbles = _group_bubbles(rows)
    timing: dict[str, dict[str, Any]] = {}
    for row in rows:
        node = str(row.get("node") or "?")
        item = timing.setdefault(
            node,
            {"node": node, "tokens": 0, "chars": 0, "first_offset_ms": int(row.get("offset_ms") or 0)},
        )
        item["tokens"] += 1
        item["chars"] += len(str(row.get("content") or ""))
    return {
        "bubbles": bubbles,
        "node_timing": list(timing.values()),
        "first_token_ms": int(rows[0].get("offset_ms") or 0) if rows else 0,
    }


# ============================================================
# 1. 课程 01：最小图 + 四模式对照
# ============================================================


def _build_probe_graph(llm: Any) -> Any:
    """课程 01 的最小图：supervisor（吐 JSON 决策）→ expert（吐自然语言）→ END。

    刻意保留两个会调模型的节点：主管产生"该被过滤的 JSON token"，
    专家产生"该进气泡的自然语言 token"，四种模式的差别才看得出来。
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class ProbeState(TypedDict):
        messages: Annotated[list, add_messages]
        next: str

    async def supervisor_node(state: ProbeState) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        question = _first_human_text(state.get("messages") or [])
        response = await llm.ainvoke(
            [
                SystemMessage(content=PROBE_SUPERVISOR_PROMPT),
                HumanMessage(content=f"用户问题：{question}"),
            ]
        )
        return {"next": _parse_route(_text(response)) or "expert"}

    async def expert_node(state: ProbeState) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage

        question = _first_human_text(state.get("messages") or [])
        response = await llm.ainvoke([HumanMessage(content=f"请用两句话回答：{question}")])
        return {"messages": [response]}

    builder = StateGraph(ProbeState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("expert", expert_node)
    builder.add_edge(START, "supervisor")
    builder.add_edge("supervisor", "expert")
    builder.add_edge("expert", END)
    return builder.compile(name="day13-probe")


def _probe_input(question: str) -> dict[str, Any]:
    """最小图的初始 state（next 显式给空串，values 模式第一条就能看出来）。"""
    from langchain_core.messages import HumanMessage

    return {"messages": [HumanMessage(content=question)], "next": ""}


def _values_view(index: int, state: dict[str, Any]) -> dict[str, Any]:
    """values 事件的一行：消息条数 / next / 最后一条消息摘要。"""
    messages = state.get("messages") or []
    return {
        "index": index,
        "messages_count": len(messages),
        "next": str(state.get("next") or ""),
        "last": _text(messages[-1])[:80] if messages else "",
    }


def _updates_view(index: int, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """updates 事件展开成"每个节点一行"（一个超步可能同时返回多个节点）。"""
    rows: list[dict[str, Any]] = []
    for node, update in (payload or {}).items():
        rows.append(
            {
                "index": index,
                "node": str(node),
                "keys": sorted(update.keys()) if isinstance(update, dict) else [],
                "digest": _update_digest(update),
            }
        )
    return rows


async def run_modes(db: AsyncSession, payload: Day13ModesRequest) -> dict[str, Any]:
    """课程 01：同一张图分别按 values / updates / messages 订阅，再演示多层同订。

    为了省一半调用，messages 与"多模式"共用同一趟执行：
    多模式那趟同时产出 updates 与 messages 两层，这恰好也是课程 2.2
    "一个超步，多层事件"最直观的演示。
    """
    _require_langgraph()

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day13-modes")
    graph = _build_probe_graph(llm)

    total_started = time.perf_counter()
    mode_stats: list[dict[str, Any]] = []

    # ① values：每个超步后的完整 state（第一条是初始 state）
    started = time.perf_counter()
    values: list[dict[str, Any]] = []
    async for state in graph.astream(_probe_input(payload.question), stream_mode="values"):
        values.append(_values_view(len(values) + 1, state))
    mode_stats.append(
        {"mode": "values", "events": len(values), "elapsed_ms": _elapsed_ms(started), "runs": 1}
    )

    # ② updates：每个节点返回的增量（键名直接告诉你它改了什么）
    started = time.perf_counter()
    updates: list[dict[str, Any]] = []
    async for chunk in graph.astream(_probe_input(payload.question), stream_mode="updates"):
        updates.extend(_updates_view(len(updates) + 1, chunk))
    mode_stats.append(
        {"mode": "updates", "events": len(updates), "elapsed_ms": _elapsed_ms(started), "runs": 1}
    )

    # ③ messages + updates：一次订阅两层，event 变成 (mode, payload) 二元组
    started = time.perf_counter()
    multi: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    async for mode, chunk in graph.astream(
        _probe_input(payload.question), stream_mode=["updates", "messages"]
    ):
        offset = _elapsed_ms(started)
        if mode == "updates":
            for row in _updates_view(len(multi) + 1, chunk):
                multi.append(
                    {"index": len(multi) + 1, "mode": "updates", "node": row["node"], "content": row["digest"]}
                )
        else:
            message_chunk, meta = chunk
            if not _is_ai_chunk(message_chunk):
                continue
            text = _text(message_chunk)
            if not text:
                continue
            node = str((meta or {}).get("langgraph_node") or "?")
            row = {
                "index": len(messages) + 1,
                "node": node,
                "content": text,
                "step": int((meta or {}).get("langgraph_step") or 0),
                "offset_ms": offset,
            }
            messages.append(row)
            multi.append(
                {"index": len(multi) + 1, "mode": "messages", "node": node, "content": text}
            )
    mode_stats.append(
        {
            "mode": "messages + updates（一趟执行，两层事件）",
            "events": len(multi),
            "elapsed_ms": _elapsed_ms(started),
            "runs": 1,
        }
    )

    # token 分流的统计（谁在说话、首 token 多晚）
    grouped = _collect_tokens([row for row in messages if _should_show_token(row["node"], row["content"])])
    filtered = [row for row in messages if not _should_show_token(row["node"], row["content"])]

    return {
        "question": payload.question,
        "values": values,
        "updates": updates,
        "messages": messages,
        "multi": multi,
        "mode_stats": mode_stats,
        "bubbles": grouped["bubbles"],
        "node_timing": grouped["node_timing"],
        "filtered_tokens": len(filtered),
        "takeaways": [
            "values 第一条就有数据（初始 state 也会吐一次），适合调试重放。",
            "updates 的增量键直接告诉你节点改了什么：主管只改 next，专家只改 messages。",
            "messages 吐的是 (chunk, metadata) 元组，metadata['langgraph_node'] 决定 token 归谁。",
            "多模式订阅时事件是 (mode, payload) 二元组——模式改成列表，解包就必须跟着改。",
            "主管的 json token 必须过滤（本次被过滤 {} 条），否则用户看到的是乱码。".format(
                len(filtered)
            ),
        ],
        "note": (
            "本实验跑了 3 趟图（6 次 LLM 调用）：values / updates 各一趟，"
            "messages 与多模式共用第三趟——省调用，也更能说明『一个超步、多层事件』。"
        ),
        "graph": _graph_structure(
            graph,
            labels={
                ("__start__", "supervisor"): "入口",
                ("supervisor", "expert"): "静态边（演示用）",
                ("expert", "__end__"): "终点",
            },
        ),
        "model": cfg.model_name,
        "elapsed_ms": _elapsed_ms(total_started),
    }


# ============================================================
# 2. 课程 02：多 Agent 下的 token 分流
# ============================================================


def _build_token_graph(llm: Any, *, max_dispatch: int = 3) -> Any:
    """课程 02 的三角色图：supervisor 路由 → 专家干活 → 回主管 → FINISH。

    专家节点**直接** ``llm.ainvoke``（不包子图），这样 token 的
    ``metadata["langgraph_node"]`` 才是专家名；包子图会变成子图里的 model，
    前端就分不清是谁在说话了。
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class TeamState(TypedDict):
        messages: Annotated[list, add_messages]
        next: str
        findings: Annotated[list[str], operator.add]
        dispatches: Annotated[int, operator.add]

    async def supervisor_node(state: TeamState) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        question = _first_human_text(state.get("messages") or [])
        done = state.get("findings") or ["（暂无）"]
        response = await llm.ainvoke(
            [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(
                    content=(
                        f"用户问题：{question}\n已完成：\n"
                        + "\n".join(done)
                        + "\n下一步交给谁？只输出 json 对象。"
                    )
                ),
            ]
        )
        nxt = _parse_route(_text(response))
        done_text = " ".join(state.get("findings") or [])

        # 三道硬守卫：模型抽风也不允许无限乒乓（Day 12 预算精神）
        if nxt not in {"researcher", "writer", "FINISH"}:
            nxt = "FINISH"
        if nxt == "researcher" and "[researcher]" in done_text:
            nxt = "writer"
        if nxt == "writer" and "[writer]" in done_text:
            nxt = "FINISH"
        if int(state.get("dispatches") or 0) >= max_dispatch:
            nxt = "FINISH"
        return {"next": nxt, "dispatches": 1}

    async def researcher_node(state: TeamState) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        question = _first_human_text(state.get("messages") or [])
        response = await llm.ainvoke(
            [
                SystemMessage(content=RESEARCHER_PROMPT),
                HumanMessage(content=f"用户问题：{question}\n请列出 2 个要点。"),
            ]
        )
        answer = _text(response)
        return {
            "messages": [AIMessage(content=answer, name="researcher")],
            "findings": [f"[researcher] {answer}"],
            "next": "supervisor",
        }

    async def writer_node(state: TeamState) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        findings = "\n".join(state.get("findings") or [])
        response = await llm.ainvoke(
            [
                SystemMessage(content=WRITER_PROMPT),
                HumanMessage(content=f"已有点：\n{findings}\n请写成一句结论。"),
            ]
        )
        answer = _text(response)
        return {
            "messages": [AIMessage(content=answer, name="writer")],
            "findings": [f"[writer] {answer}"],
            "next": "supervisor",
        }

    def route_after_supervisor(state: TeamState) -> str:
        """路由函数只返回节点名（不返回 Command，保持与静态边的写法一致）。"""
        nxt = str(state.get("next") or "FINISH")
        return END if nxt in {"FINISH", "__end__"} else nxt

    builder = StateGraph(TeamState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"researcher": "researcher", "writer": "writer", END: END},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("writer", "supervisor")
    return builder.compile(name="day13-token")


async def run_token_stream(db: AsyncSession, payload: Day13TokenRequest) -> dict[str, Any]:
    """课程 02：三个 LLM 一起吐 token，按 ``langgraph_node`` 分流到各自气泡。

    主管的 token 全部进 ``filtered``（保留样例给前端看"过滤掉了什么"），
    专家的 token 按节点聚合成气泡——这就是多 Agent 前端的标配渲染模型。
    """
    _require_langgraph()
    from langchain_core.messages import HumanMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day13-token")
    graph = _build_token_graph(llm)

    started = time.perf_counter()
    visible: list[dict[str, Any]] = []
    hidden: list[dict[str, Any]] = []
    dispatches: list[dict[str, Any]] = []

    async for mode, chunk in graph.astream(
        {"messages": [HumanMessage(content=payload.question)], "findings": [], "dispatches": 0},
        stream_mode=["messages", "updates"],
    ):
        offset = _elapsed_ms(started)
        if mode == "updates":
            for node, update in (chunk or {}).items():
                if node == "supervisor" and isinstance(update, dict):
                    dispatches.append(
                        {"next": str(update.get("next") or ""), "offset_ms": offset}
                    )
            continue
        message_chunk, meta = chunk
        if not _is_ai_chunk(message_chunk):
            continue
        text = _text(message_chunk)
        if not text:
            continue
        node = str((meta or {}).get("langgraph_node") or "?")
        row = {"node": node, "content": text, "offset_ms": offset}
        if _should_show_token(node, text):
            visible.append(row)
        else:
            hidden.append(row)

    grouped = _collect_tokens(visible)
    hidden_text = "".join(row["content"] for row in hidden)
    return {
        "question": payload.question,
        "bubbles": grouped["bubbles"],
        "node_timing": grouped["node_timing"],
        "first_token_ms": grouped["first_token_ms"],
        "filtered": {
            "nodes": sorted({row["node"] for row in hidden}),
            "tokens": len(hidden),
            "chars": len(hidden_text),
            "sample": hidden_text[:160],
        },
        "dispatches": dispatches,
        "filter_rule": (
            "node == 'supervisor' 的 token 一律不进气泡：它是 json 片段，"
            "用户看到就是乱码；专家 token 按 node 聚合，节点一变就开新气泡。"
        ),
        "note": (
            "主管一共派单 {} 次；它的 token 全部被过滤掉（{} 条），"
            "用户只看到专家的自然语言——这就是 Day 12 场景流式化的完整形态。"
        ).format(len(dispatches), len(hidden)),
        "graph": _graph_structure(
            graph,
            labels={
                ("__start__", "supervisor"): "入口",
                ("supervisor", "researcher"): "路由：查资料",
                ("supervisor", "writer"): "路由：写结论",
                ("supervisor", "__end__"): "FINISH",
                ("researcher", "supervisor"): "回主管复命",
                ("writer", "supervisor"): "回主管复命",
            },
        ),
        "model": cfg.model_name,
        "elapsed_ms": _elapsed_ms(started),
    }


# ============================================================
# 3. 课程 03：custom 模式——工具内部进度上报
# ============================================================


def _build_doc_tools() -> list[Any]:
    """课程 03 的检索工具：内部用 ``get_stream_writer()`` 上报每条命中进度。

    ``get_stream_writer()`` 必须在**图执行上下文内**调用（节点函数或被图调用的工具），
    在普通函数里调用拿不到 writer，``writer()`` 会变成静默的无操作（坑 3）。
    """
    from langchain_core.tools import tool
    from langgraph.config import get_stream_writer

    @tool(parse_docstring=True)
    def search_docs(query: str) -> str:
        """在文档库中检索相关内容，并实时上报检索进度。

        何时用：回答任何 LangGraph / 协议相关问题之前，必须先检索文档库。
        何时不用：纯寒暄、不需要事实依据时。

        Args:
            query: 检索关键词，如"断点续跑"、"Checkpointer"。
        """
        writer = get_stream_writer()
        writer({"type": "progress", "stage": "开始检索", "query": query})

        words = [word for word in (query or "").replace("？", " ").split() if word]
        hits = [key for key, value in FAKE_DOCS.items() if any(word in value for word in words)]
        hits = hits or list(FAKE_DOCS)  # 没命中任何关键词就返回全部，保证进度条有东西可动
        for index, key in enumerate(hits, 1):
            time.sleep(SEARCH_STEP_SECONDS)  # 模拟每条检索耗时，让进度条真的在动
            writer({"type": "progress", "stage": f"命中第 {index} 条", "doc": key})

        writer({"type": "progress", "stage": "检索完成", "total": len(hits)})
        return "\n".join(FAKE_DOCS[key] for key in hits) or "未找到相关文档"

    return [search_docs]


def _build_doc_agent(llm: Any, tools: list[Any], *, max_steps: int = 3) -> Any:
    """文档助手：model ⇄ tools 的最小 ReAct 图（与 Day 12 专家子图同构）。"""
    from langchain_core.messages import SystemMessage, ToolMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    tool_map = {item.name: item for item in tools}
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    class DocState(TypedDict):
        messages: Annotated[list, add_messages]
        steps: Annotated[int, operator.add]

    async def agent_node(state: DocState) -> dict[str, Any]:
        response = await llm_with_tools.ainvoke(
            [SystemMessage(content=DOC_SYSTEM), *(state.get("messages") or [])]
        )
        return {"messages": [response], "steps": 1}

    async def tools_node(state: DocState) -> dict[str, Any]:
        last = (state.get("messages") or [None])[-1]
        outputs: list[Any] = []
        for call in list(getattr(last, "tool_calls", None) or []):
            name = str(call.get("name") or "")
            fn = tool_map.get(name)
            if fn is None:
                content = f"未知工具 {name}，可用：{sorted(tool_map)}"
            else:
                try:
                    content = str(await fn.ainvoke(dict(call.get("args") or {})))
                except Exception as exc:  # noqa: BLE001  错误也包成 ToolMessage，配对不能断
                    content = f"工具执行失败：{type(exc).__name__}: {exc}"
            outputs.append(ToolMessage(content=content, tool_call_id=str(call.get("id") or "")))
        return {"messages": outputs}

    def after_agent(state: DocState) -> str:
        """有 tool_calls 且预算没用完就继续；否则收工（语义收敛 + 步数刹车）。"""
        last = (state.get("messages") or [None])[-1]
        if not getattr(last, "tool_calls", None):
            return END
        if int(state.get("steps") or 0) >= max_steps:
            return END
        return "tools"

    builder = StateGraph(DocState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", after_agent, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")
    return builder.compile(name="day13-custom")


async def run_custom_stream(db: AsyncSession, payload: Day13CustomRequest) -> dict[str, Any]:
    """课程 03：订阅 ``custom`` 进度 + ``messages`` token + ``updates`` 节点增量。

    工具执行是黑盒：检索类工具跑 3 秒，用户只看到卡住。``get_stream_writer()``
    让工具内部主动上报进度，前端把它渲染成进度条 / 日志行。
    """
    _require_langgraph()
    from langchain_core.messages import HumanMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day13-custom")
    tools = _build_doc_tools()
    graph = _build_doc_agent(llm, tools)

    started = time.perf_counter()
    progress: list[dict[str, Any]] = []
    tokens: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    final_answer = ""

    async for mode, chunk in graph.astream(
        {"messages": [HumanMessage(content=payload.question)], "steps": 0},
        stream_mode=["custom", "messages", "updates"],
    ):
        offset = _elapsed_ms(started)
        if mode == "custom":
            row = dict(chunk or {})
            row["offset_ms"] = offset
            progress.append(row)
        elif mode == "messages":
            message_chunk, meta = chunk
            node = str((meta or {}).get("langgraph_node") or "?")
            text = _text(message_chunk)
            if getattr(message_chunk, "type", "") == "tool":
                tool_results.append({"node": node, "content": text, "offset_ms": offset})
                continue
            if not _is_ai_chunk(message_chunk) or not text:
                continue
            tokens.append({"node": node, "content": text, "offset_ms": offset})
        else:
            for node, update in (chunk or {}).items():
                if node == "agent" and isinstance(update, dict):
                    messages = update.get("messages") or []
                    if messages:
                        answer = _last_answer(list(messages))
                        if answer:
                            final_answer = answer

    grouped = _collect_tokens([row for row in tokens if _should_show_token(row["node"], row["content"])])
    return {
        "question": payload.question,
        "progress": progress,
        "stages": [
            "{}｜{}".format(row.get("stage"), row.get("doc") or row.get("total") or "")
            for row in progress
        ],
        "tokens": tokens,
        "bubbles": grouped["bubbles"],
        "tool_results": tool_results,
        "final_answer": final_answer
        or "".join(row["content"] for row in tokens if row["node"] not in {"supervisor"}),
        "first_token_ms": grouped["first_token_ms"],
        "protocol_note": (
            "custom 的 payload 是你自己定义的协议：生产里固定成 "
            "{'type': ..., 'stage': ..., 'data': ...} 三键，前端按 type 分发渲染"
        ),
        "note": (
            "进度事件 {} 条、token {} 块、工具结果 {} 条——"
            "三者按 offset_ms 排在同一条时间轴上，前端就能画出『检索中 → 逐字输出』。"
        ).format(len(progress), len(tokens), len(tool_results)),
        "graph": _graph_structure(
            graph,
            labels={
                ("__start__", "agent"): "入口",
                ("agent", "tools"): "有 tool_calls → tools",
                ("agent", "__end__"): "没有 tool_calls → END",
                ("tools", "agent"): "观察结果回填",
            },
        ),
        "model": cfg.model_name,
        "elapsed_ms": _elapsed_ms(started),
    }


# ============================================================
# 4. 课程 04：interrupt + 流式同开（今天最经典的坑）
# ============================================================

# 进程级存档器：interrupt 必须配 checkpointer，多个请求共享才能续跑
_DAY13_SAVER: Any | None = None

# 退款流水（进程内存，演示用）：只有 resume 之后才 append，用来证明重放不重复扣款
REFUND_LOG: list[dict[str, Any]] = []


def _saver() -> Any:
    """进程级 MemorySaver 单例（与 Day 11 / 12 同一套做法）。"""
    global _DAY13_SAVER
    if _DAY13_SAVER is None:
        from langgraph.checkpoint.memory import MemorySaver

        _DAY13_SAVER = MemorySaver()
    return _DAY13_SAVER


def _build_refund_app() -> Any:
    """课程 04 的退款图：START → refund → END（refund 内有 interrupt 确认门）。

    节点里**不调模型**：今天讲的是流的形状与暂停信号，不是模型质量；
    去掉模型依赖后这个实验变成完全确定性的，接口和单测都能稳定复现。
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class RefundState(TypedDict):
        messages: Annotated[list, add_messages]
        amount: float
        thread_id: str
        refunds: Annotated[list[dict[str, Any]], operator.add]
        events: Annotated[list[dict[str, Any]], operator.add]

    async def refund_node(state: RefundState) -> dict[str, Any]:
        from langchain_core.messages import AIMessage
        from langgraph.types import interrupt

        amount = float(state.get("amount") or DEFAULT_REFUND_AMOUNT)

        # ★ interrupt 之前零副作用（Day 11 铁律）★
        decision = interrupt(
            {
                "type": "confirm",
                "action": "refund",
                "amount": amount,
                "prompt": f"即将退款 {amount:g} 元，确认？",
            }
        )
        approved, patched, note = _normalize_decision(decision)
        final_amount = patched if patched is not None else amount
        if not approved:
            return {
                "messages": [AIMessage(content="退款被人工驳回，未执行任何资金操作。")],
                "events": [
                    {"node": "refund", "kind": "approval", "approved": False, "detail": "人工驳回"}
                ],
            }

        # 副作用只在 resume 之后发生：节点被重放也不会扣两次款
        row = {
            "thread_id": str(state.get("thread_id") or ""),
            "amount": final_amount,
            "note": note,
            "patched": patched is not None,
        }
        REFUND_LOG.append(row)
        return {
            "messages": [AIMessage(content=f"已退款 {final_amount:g} 元")],
            "refunds": [{**row, "approved": True}],
            "events": [
                {
                    "node": "refund",
                    "kind": "approval",
                    "approved": True,
                    "detail": f"放行 {final_amount:g} 元" + (f"（人工改过金额）" if patched is not None else ""),
                }
            ],
        }

    builder = StateGraph(RefundState)
    builder.add_node("refund", refund_node)
    builder.add_edge(START, "refund")
    builder.add_edge("refund", END)
    return builder.compile(checkpointer=_saver(), name="day13-refund")


def _refund_input(amount: float, thread_id: str) -> dict[str, Any]:
    """退款图的初始 state（thread_id 进 state 方便事件里标注属于哪次演示）。"""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content="我要退款")],
        "amount": amount,
        "thread_id": thread_id,
    }


def _config(thread_id: str) -> dict[str, Any]:
    """invoke / stream 的配置：thread_id 是会话身份（Day 11 的老朋友）。"""
    return {"configurable": {"thread_id": thread_id}}


async def _collect_messages_only(stream: Any, started: float) -> list[dict[str, Any]]:
    """只订 messages：这就是"为了打字机效果只订阅 token"的前端写法。"""
    events: list[dict[str, Any]] = []
    async for chunk, meta in stream:
        if not _is_ai_chunk(chunk):
            continue
        text = _text(chunk)
        if not text:
            continue
        events.append(
            {
                "seq": len(events) + 1,
                "mode": "messages",
                "node": str((meta or {}).get("langgraph_node") or "?"),
                "kind": "token",
                "content": text,
                "offset_ms": _elapsed_ms(started),
            }
        )
    return events


async def _collect_dual(stream: Any, started: float) -> list[dict[str, Any]]:
    """订 messages + updates：暂停信号会以 ``__interrupt__`` 键出现在 updates 里。"""
    events: list[dict[str, Any]] = []
    async for mode, payload in stream:
        offset = _elapsed_ms(started)
        if mode == "updates":
            if "__interrupt__" in (payload or {}):
                intr = payload["__interrupt__"][0]
                events.append(
                    {
                        "seq": len(events) + 1,
                        "mode": "updates",
                        "node": "__interrupt__",
                        "kind": "interrupt",
                        "content": _jsonable(getattr(intr, "value", intr)),
                        "offset_ms": offset,
                    }
                )
                continue
            for node, update in (payload or {}).items():
                events.append(
                    {
                        "seq": len(events) + 1,
                        "mode": "updates",
                        "node": str(node),
                        "kind": "update",
                        "keys": sorted(update.keys()) if isinstance(update, dict) else [],
                        "content": _update_digest(update),
                        "offset_ms": offset,
                    }
                )
        else:
            chunk, meta = payload
            if not _is_ai_chunk(chunk):
                continue
            text = _text(chunk)
            if not text:
                continue
            events.append(
                {
                    "seq": len(events) + 1,
                    "mode": "messages",
                    "node": str((meta or {}).get("langgraph_node") or "?"),
                    "kind": "token",
                    "content": text,
                    "offset_ms": offset,
                }
            )
    return events


async def _pause_view(app: Any, config: dict[str, Any]) -> dict[str, Any]:
    """读一眼快照：next 非空说明图正停在那里（另一种探测方式）。"""
    snapshot = await app.aget_state(config)
    next_nodes = list(getattr(snapshot, "next", ()) or ())
    interrupts: list[dict[str, Any]] = []
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            interrupts.append(_jsonable(getattr(item, "value", item)))
    return {"paused": bool(next_nodes), "next": [str(node) for node in next_nodes], "tasks": interrupts}


async def run_interrupt_stream(payload: Day13InterruptRequest) -> dict[str, Any]:
    """课程 04：同一张退款图跑三遍，把"只订 messages 会漏暂停"这个坑摆到台面上。

    ① 只订 messages → 事件 0 条、图已暂停、**interrupt_seen=False**（坑的现场）；
    ② messages + updates → 收到 ``__interrupt__``，拿到审批载荷；
    ③ ``Command(resume=...)`` 续跑 → 事件继续流，退款流水只记一次。
    """
    _require_langgraph()
    from langgraph.types import Command

    started = time.perf_counter()
    app = _build_refund_app()
    amount = float(payload.amount)

    # ① 错误示范：只订阅 token 流
    thread_a = _thread_id("stream-msg")
    config_a = _config(thread_a)
    events_a = await _collect_messages_only(
        app.astream(_refund_input(amount, thread_a), config_a, stream_mode="messages"), started
    )
    pause_a = await _pause_view(app, config_a)

    # ② 正确示范：messages + updates 一起订
    thread_b = _thread_id("stream-fix")
    config_b = _config(thread_b)
    events_b = await _collect_dual(
        app.astream(
            _refund_input(amount, thread_b), config_b, stream_mode=["messages", "updates"]
        ),
        started,
    )
    pause_b = await _pause_view(app, config_b)
    interrupt_value = next(
        (row["content"] for row in events_b if row.get("kind") == "interrupt"), None
    )

    # ③ 续跑：Command(resume=...) 语义不变（Day 11）
    resume_view: dict[str, Any] | None = None
    if payload.with_resume and pause_b["paused"]:
        resume_events = await _collect_dual(
            app.astream(Command(resume=payload.decision), config_b, stream_mode=["messages", "updates"]),
            started,
        )
        pause_c = await _pause_view(app, config_b)
        resume_view = {
            "thread_id": thread_b,
            "events": resume_events,
            "paused": pause_c["paused"],
            "next": pause_c["next"],
            "final_answer": "".join(
                row["content"]
                for row in resume_events
                if row.get("kind") == "token"
                and not str(row.get("content", "")).startswith("{")
            )
            or "（本节点不调模型，所以续跑阶段没有 token）",
            "refund_events": [row for row in resume_events if row.get("kind") == "update"],
        }

    refunds = [row for row in REFUND_LOG if row.get("thread_id") == thread_b]
    return {
        "messages_only": {
            "thread_id": thread_a,
            "events": events_a,
            "event_count": len(events_a),
            "interrupt_seen": any(row.get("kind") == "interrupt" for row in events_a),
            "paused": pause_a["paused"],
            "next": pause_a["next"],
            "verdict": "❌ 图已经停住，但 messages 流里一条暂停信号都没有——前端会一直等 token",
        },
        "with_updates": {
            "thread_id": thread_b,
            "events": events_b,
            "event_count": len(events_b),
            "interrupt_seen": interrupt_value is not None,
            "interrupt": interrupt_value,
            "paused": pause_b["paused"],
            "next": pause_b["next"],
            "verdict": "✅ __interrupt__ 出现在 updates 流里，前端可以弹审批框了",
        },
        "resume": resume_view,
        "refunds": _jsonable(refunds),
        "rules": [
            "interrupt() 的实现是抛出暂停信号，节点没有返回值；messages 流只承载 LLM token。",
            "暂停信号以 __interrupt__ 为键塞进 updates 流：stream_mode=['messages','updates']。",
            "另外两种探测：astream_events v2 的 on_interrupt；get_state(config).next 非空 + tasks 有 interrupted。",
            "续跑仍然是 Command(resume=...)；interrupt 之前零副作用，重放不会重复扣款。",
        ],
        "note": (
            "故意让两个 thread 各跑一遍：左边演示坑，右边演示解法；"
            "第三个阶段用 Command(resume=...) 续跑。"
            "当前退款流水 {} 条（只记一次，证明重放安全）。".format(len(refunds))
        ),
        "elapsed_ms": _elapsed_ms(started),
    }


# ============================================================
# 5. 课程 05：生产最后一公里——LangGraph → SSE → 浏览器
# ============================================================


def _build_chat_graph(llm: Any) -> Any:
    """SSE 演示用的单节点对话图（一个超步、一处 token 来源，协议演示更清楚）。"""
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class SseState(TypedDict):
        messages: Annotated[list, add_messages]

    async def chat_node(state: SseState) -> dict[str, Any]:
        response = await llm.ainvoke(state.get("messages") or [])
        return {"messages": [response]}

    builder = StateGraph(SseState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(name="day13-sse")


async def sse_model(db: AsyncSession, model_config_id: int | None) -> Any:
    """SSE 接口用：在返回响应之前把模型配置解析好，生成器里不再碰数据库。"""
    return await resolve_model(db, model_config_id)


async def sse_chat(cfg: Any, question: str) -> AsyncIterator[str]:
    """把图的事件流翻译成四类 SSE 帧：node_start / token / done（error 兜底）。

    模型配置由调用方在**返回响应之前**解析好传进来：请求级 DB 会话在
    StreamingResponse 开始后就关闭了，生成器里不能再碰 db。
    """
    from langchain_core.messages import HumanMessage

    llm = build_chat_model(cfg, temperature=0.0, run_name="day13-sse")
    graph = _build_chat_graph(llm)

    started = time.perf_counter()
    first_token_ms: int | None = None
    chars = 0

    yield sse({"type": "node_start", "node": "chat"})
    try:
        async for chunk, meta in graph.astream(
            {"messages": [HumanMessage(content=question)]}, stream_mode="messages"
        ):
            if not _is_ai_chunk(chunk):
                continue
            text = _text(chunk)
            if not text:
                continue
            if first_token_ms is None:
                first_token_ms = _elapsed_ms(started)
            chars += len(text)
            yield sse(
                {
                    "type": "token",
                    "node": str((meta or {}).get("langgraph_node") or "chat"),
                    "content": text,
                }
            )
    except Exception as exc:  # noqa: BLE001  HTTP 状态码已经是 200，只能靠事件体告知
        yield sse({"type": "error", "message": f"模型调用失败：{type(exc).__name__}: {exc}"})
        return

    yield sse(
        {
            "type": "done",
            "chars": chars,
            "first_token_ms": first_token_ms if first_token_ms is not None else 0,
            "elapsed_ms": _elapsed_ms(started),
        }
    )


# ============================================================
# 6. 静态概念数据接口
# ============================================================


async def mapping() -> dict[str, Any]:
    """Day 13 概念表：四模式 / API 选型 / interrupt 规则 / SSE 三件套 / 事件协议 / 五个坑。"""
    return {
        "stream_modes": STREAM_MODES,
        "api_choices": API_CHOICES,
        "interrupt_rules": INTERRUPT_RULES,
        "sse_rules": SSE_RULES,
        "event_protocol": EVENT_PROTOCOL,
        "pitfalls": PITFALLS,
        "checklist": CHECKLIST,
        "presets": PRESETS,
        "install_hint": INSTALL_HINT,
        "langgraph_version": await asyncio.to_thread(_version, "langgraph"),
    }


__all__ = [
    "API_CHOICES",
    "CHECKLIST",
    "EVENT_PROTOCOL",
    "FAKE_DOCS",
    "INSTALL_HINT",
    "INTERRUPT_RULES",
    "PITFALLS",
    "PRESETS",
    "REFUND_LOG",
    "SSE_RULES",
    "STREAM_MODES",
    "mapping",
    "run_custom_stream",
    "run_interrupt_stream",
    "run_modes",
    "run_token_stream",
    "sse",
    "sse_chat",
    "sse_headers",
    "sse_model",
]
