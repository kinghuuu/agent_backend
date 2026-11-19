"""Day 11 Checkpointer 与人机协同：让 Agent 能暂停、能续跑、能反悔。

课程 day11 的四个脚本 + 一个重放实验在这里收成一个模块：

    存档/续跑/隔离 → ``run_basic``            （01：MemorySaver + thread_id + get_state）
    交互式续聊     → ``chat``                 （01 的交互版：进程级存档器，跨请求读档）
    图级确认门     → ``start_approval`` / ``resume_approval`` / ``expire_approvals``
                                              （02 + 练习 2「修改后放行」+ 练习 3「超时自动驳回」）
    SQLite 重启    → ``run_sqlite``           （03：AsyncSqliteSaver，换连接等价于换进程）
    时间旅行       → ``run_time_travel``      （04：get_state_history → update_state → invoke(None)）
    节点重放       → ``replay_demo``          （坑 1：interrupt 前有副作用会被重放两次，纯本地）

贯穿全篇的三条结论：
1. checkpoint 是**运行时行为**：接上 checkpointer 就自动"一超步一帧"，不用写保存代码；
2. `thread_id` 是会话身份，同 id 续跑、异 id 隔离；新输入靠 `add_messages` reducer 并入历史；
3. **恢复时被 interrupt 的节点会从头重放**——interrupt 之前只许"纯读"，副作用必须放到拿到
   `Command(resume=...)` 之后（或拆成独立节点）。

与 Day 09/10 一样，langgraph 全部延迟导入：缺依赖时接口返回可读的 400，而不是让应用起不来。
"""

import asyncio
import json
import operator
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import (
    Day11ApprovalResumeRequest,
    Day11ApprovalStartRequest,
    Day11BasicRequest,
    Day11ChatRequest,
    Day11ExpireRequest,
    Day11SqliteRequest,
    Day11TimeTravelRequest,
)
from app.modules.study.services.day09_langchain import build_chat_model

# 复用 Day 10 的序列化与出图工具：消息→dict、图结构抽取只该有一份实现
from app.modules.study.services.day10_langgraph import _graph_structure, _jsonable, _text
from app.modules.study.services.llm import resolve_model

# 安装提示：缺依赖时所有 Day 11 接口都返回这句话
INSTALL_HINT = "pip install -U langgraph langgraph-checkpoint-sqlite"

# SQLite 存档文件：放在 agent_backend/var/ 下（运行时生成，已加入 .gitignore）
CHECKPOINT_DIR = Path(__file__).resolve().parents[4] / "var"
SQLITE_FILE = CHECKPOINT_DIR / "day11_checkpoints.db"

# 敏感工具：命中就走 approval 节点等人批
DANGEROUS = {"refund_order"}

# 课程 1.3 的 StateSnapshot 四要素
SNAPSHOT_FIELDS: list[dict[str, str]] = [
    {
        "field": "values",
        "what": "这一帧的完整 State（dict）",
        "use": "读档时看到的全部数据：messages / counters / 业务字段",
    },
    {
        "field": "next",
        "what": "接下来要跑哪些节点（tuple）",
        "use": "非空 = 图卡在这里；中断时就是『卡在哪个节点』",
    },
    {
        "field": "tasks",
        "what": "待办任务，含中断信息 interrupts",
        "use": "动态中断的载荷在这里：snap.tasks[0].interrupts[0].value",
    },
    {
        "field": "created_at",
        "what": "这一帧的时间戳（ISO 字符串）",
        "use": "审批超时扫描、审计与时间旅行定位",
    },
]

# 课程 1.7 的两套暂停机制
PAUSE_MECHANISMS: list[dict[str, str]] = [
    {
        "dim": "写在哪",
        "dynamic": "节点函数内部：interrupt(...)",
        "static": "compile(interrupt_before=[...])",
    },
    {
        "dim": "何时触发",
        "dynamic": "运行到这行且条件满足才停",
        "static": "每次经过该节点前都停",
    },
    {
        "dim": "能否带问题数据",
        "dynamic": "能——interrupt 的参数就是发给人的载荷",
        "static": "不能，只能事后 get_state 看",
    },
    {
        "dim": "恢复方式",
        "dynamic": "invoke(Command(resume=值), config)",
        "static": "invoke(None, config)",
    },
    {
        "dim": "典型用途",
        "dynamic": "审批门、确认敏感操作",
        "static": "调试时逐步单步执行",
    },
]

# 课程 1.9 的三档存档器（接口一致，换存档器不改节点代码）
SAVERS: list[dict[str, str]] = [
    {
        "saver": "MemorySaver",
        "package": "内置 langgraph",
        "life": "进程内，重启即丢",
        "use": "本地开发、单元测试（本页 ①②④ 用它）",
    },
    {
        "saver": "SqliteSaver / AsyncSqliteSaver",
        "package": "langgraph-checkpoint-sqlite",
        "life": "落盘文件",
        "use": "单机、小流量（本页 ⑤ 用它演示重启续跑）",
    },
    {
        "saver": "PostgresSaver",
        "package": "langgraph-checkpoint-postgres",
        "life": "数据库",
        "use": "生产多副本——会话不绑在某台机器上",
    },
]

# 课程第 4 节的坑（第 1 个会造成真实资损）
PITFALLS: list[dict[str, str]] = [
    {
        "no": "1",
        "title": "interrupt() 前有副作用，恢复时重复执行",
        "symptom": "审批通过后用户被扣了两次款；或审批前就已经发了通知",
        "cause": "恢复时节点从头重放，interrupt 之前的代码原样再跑一遍",
        "fix": "interrupt 之前只许纯读；副作用放到拿到 resume 值之后，或拆成决策 / 审批 / 执行三个节点",
    },
    {
        "no": "2",
        "title": "invoke 忘带 config，或两次调用 thread_id 不一致",
        "symptom": "报 Checkpointer requires thread_id；或明明续聊，模型却说不记得",
        "cause": "每次请求都 uuid4() 新建 thread_id，等于每次开新会话",
        "fix": "thread_id 从业务键生成（user-{uid}-conv-{cid}），一个会话生命周期内稳定不变",
    },
    {
        "no": "3",
        "title": "以为 MemorySaver 是持久化",
        "symptom": "上线重启服务后所有会话失忆，本地单进程测试却一切正常",
        "cause": "MemorySaver 存在进程内存里，重启即蒸发",
        "fix": "开发用 MemorySaver，上生产前换成 SqliteSaver / PostgresSaver",
    },
    {
        "no": "4",
        "title": "拿静态断点的恢复方式恢复动态中断（或反过来）",
        "symptom": "interrupt() 停住后用 invoke(None) 恢复，图毫无反应或报错",
        "cause": "两套机制恢复方式不同：动态用 Command(resume=值)，静态用 invoke(None)",
        "fix": "不确定就 get_state：snap.tasks 里有 interrupts 就是动态中断",
    },
    {
        "no": "5",
        "title": "State 里放了不可序列化的对象",
        "symptom": "挂上 SqliteSaver 后报 cannot pickle / 序列化失败",
        "cause": "checkpoint 要把整个 State 落盘：数据库连接、文件句柄、自定义实例都存不进去",
        "fix": "State 只放 dict / list / str / int / 消息对象；活对象存 key，节点里现查现用",
    },
    {
        "no": "6",
        "title": "同一个 thread_id 被并发请求共用",
        "symptom": "两个用户消息交错时 Agent 回答串台",
        "cause": "thread_id 相同 → 读写同一份存档",
        "fix": "thread_id 的粒度是『一次对话』而不是『一个用户』：带上 conv 维度",
    },
]

# 课程 1.8 的时间旅行三步
TIME_TRAVEL_STEPS: list[str] = [
    "① get_state_history(config) 翻历史（新的在前），挑一帧当分叉点",
    "② update_state(带 checkpoint_id 的 config, 新值, as_node=...) 从该帧改写，分叉出新链（不动原历史）",
    "③ invoke(None, old_cfg) 不给新输入，从这一帧重新往后跑",
]

# 人机协同三条铁律
RULES: list[str] = [
    "interrupt() 的返回值 = 恢复时 Command(resume=...) 传入的值",
    "图停在超步边界且 state 已存档：进程可以死，checkpoint 活着（所以能隔一天再批）",
    "恢复时被 interrupt 的节点从头重放：interrupt 之前必须零副作用",
]

# 各实验的预设（前端一键填入）
PRESETS: dict[str, list[str]] = {
    "chat": ["我叫小胡，最喜欢的数字是 7", "我叫什么？最喜欢的数字是多少？", "刚才我们聊了什么？"],
    "approval": [
        "订单 A1001 不想要了，请直接帮我退款 299 元",
        "订单 A1001 的退款到账了吗？顺便退 299 元",
        "查一下订单 A1001 的状态",
    ],
    "time_travel": ["RAG", "MCP 协议", "Function Calling"],
}

# 确认门图的系统提示词（Day 02 的四段式）
SYSTEM_PROMPT = """你是售后 Agent。
规则：
1. 退款是敏感资金操作，必须调用 refund_order 工具，不得只在回复里承诺；
2. 调用前先用 get_order 查清订单金额与状态；
3. 工具返回【已拦截】时如实告知用户审批未通过。
输出格式：先用一句话给结论，再补充依据。"""

# 进程级存档器：整个进程共用一份，thread_id 区分会话。
# 这等价于生产里的一台存档服务——图可以每次重新编译，存档不会丢。
_MEMORY_SAVER: Any | None = None

# 待审批队列（练习 3 的定时任务扫描对象）：thread_id → 审批现场
PENDING_APPROVALS: dict[str, dict[str, Any]] = {}


# ============================================================
# 0. 公共设施：依赖探测 / 存档器 / 配置 / 快照序列化
# ============================================================


def _require_langgraph() -> None:
    """依赖探测：没装 langgraph 时给安装命令，而不是 ImportError 堆栈。"""
    try:
        import langgraph.graph  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 只在缺依赖的环境触发
        raise ValueError(f"未安装 langgraph，请先执行：{INSTALL_HINT}") from exc


def _memory_saver() -> Any:
    """进程级 MemorySaver 单例：多个请求共享，才能演示跨请求续跑。"""
    global _MEMORY_SAVER
    if _MEMORY_SAVER is None:
        from langgraph.checkpoint.memory import MemorySaver

        _MEMORY_SAVER = MemorySaver()
    return _MEMORY_SAVER


def _config(thread_id: str, checkpoint_id: str | None = None) -> dict[str, Any]:
    """构造 invoke 配置：thread_id 是会话身份，checkpoint_id 用于时间旅行定位某一帧。"""
    configurable: dict[str, Any] = {"thread_id": thread_id}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _thread_id(prefix: str) -> str:
    """生成一次性的演示会话 ID（真实业务应从用户/会话键派生，见坑 2）。"""
    return f"{prefix}-{uuid4().hex[:6]}"


def _checkpoint_id_of(snap: Any) -> str:
    """从 StateSnapshot 里取 checkpoint_id：它就是历史链上的帧主键。"""
    configurable = (getattr(snap, "config", None) or {}).get("configurable", {})
    return str(configurable.get("checkpoint_id") or "")


def _snapshot_view(snap: Any) -> dict[str, Any]:
    """把 StateSnapshot 转成前端好渲染的 dict（四要素 + 消息摘要）。"""
    values = getattr(snap, "values", None) or {}
    messages = values.get("messages", []) or []
    tasks = []
    for task in getattr(snap, "tasks", ()) or ():
        tasks.append(
            {
                "name": getattr(task, "name", ""),
                "interrupts": [_jsonable(item.value) for item in (getattr(task, "interrupts", ()) or ())],
            }
        )
    return {
        "values": {key: _jsonable(value) for key, value in values.items() if key != "messages"},
        "messages": _jsonable(messages),
        "messages_count": len(messages),
        "next": list(getattr(snap, "next", ()) or ()),
        "tasks": tasks,
        "created_at": getattr(snap, "created_at", None),
        "checkpoint_id": _checkpoint_id_of(snap),
        "has_interrupt": any(item["interrupts"] for item in tasks),
    }


async def _frames(app: Any, config: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    """翻出这条 thread 的历史帧（新的在前）：时间旅行的"存档列表"。"""
    rows: list[dict[str, Any]] = []
    index = 0
    async for snap in app.aget_state_history(config, limit=limit):
        view = _snapshot_view(snap)
        rows.append(
            {
                "index": index,
                "next": view["next"],
                "checkpoint_id": view["checkpoint_id"],
                "created_at": view["created_at"],
                "messages_count": view["messages_count"],
                "values": view["values"],
            }
        )
        index += 1
    return rows


def _tool_results(messages: list[Any]) -> list[dict[str, Any]]:
    """把 state 里的 ToolMessage 摘成"工具执行结果"列表（含是否被审批拦截）。"""
    names: dict[str, str] = {}
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            names[str(call.get("id", ""))] = str(call.get("name", ""))
    rows: list[dict[str, Any]] = []
    for message in messages:
        if getattr(message, "type", "") != "tool":
            continue
        content = _text(message)
        rows.append(
            {
                "tool": names.get(str(getattr(message, "tool_call_id", "")), "?"),
                "content": content,
                "blocked": content.startswith("【已拦截】"),
            }
        )
    return rows


def _last_answer(messages: list[Any]) -> str:
    """最后一条不带 tool_calls 的 AI 消息 = 可以直接回给用户的回答。"""
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", None):
            return _text(message)
    return ""


def _version(name: str) -> str:
    """读已安装包版本：框架篇的坑大半和版本有关。"""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001  没装就是没装，前端照常展示
        return "未安装"


# ============================================================
# 1. 存档 / 续跑 / 隔离：课程 01 的三个实验
# ============================================================


def _build_chat_graph(llm: Any, checkpointer: Any) -> Any:
    """最小的"记忆型"图：START → chat → END，checkpointer 是唯一的魔法。

    `turns` 用覆盖语义（节点里自己 +1）；`messages` 用 add_messages——
    续跑时新消息就是靠它并进旧历史的（Day 10 的 reducer 在今天的第二重身份）。
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class ChatState(TypedDict):
        messages: Annotated[list, add_messages]
        turns: int

    async def chat_node(state: ChatState) -> dict[str, Any]:
        reply = await llm.ainvoke(state["messages"])
        return {"messages": [reply], "turns": int(state.get("turns") or 0) + 1}

    builder = StateGraph(ChatState)
    builder.add_node("chat", chat_node)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    return builder.compile(checkpointer=checkpointer)


async def run_basic(db: AsyncSession, payload: Day11BasicRequest) -> dict[str, Any]:
    """跑课程 01 的三个实验：同 thread 续跑 / 跨 thread 隔离 / get_state 读最近一帧。

    每次都用新的 thread 对（`user-xxxx` 与 `user-xxxx-other`），保证实验可复现：
    实验 1 第二轮只传一条新消息、不传 turns，若最终 turns=2 且模型记得名字，
    就证明历史与计数都来自存档帧，而不是 Python 变量。
    """
    _require_langgraph()
    from langchain_core.messages import HumanMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day11-basic")
    app = _build_chat_graph(llm, _memory_saver())

    thread_a = _thread_id("user")
    thread_b = f"{thread_a}-other"
    config_a = _config(thread_a)
    config_b = _config(thread_b)

    started = time.perf_counter()
    first = await app.ainvoke(
        {"messages": [HumanMessage(content=payload.first_message)], "turns": 0}, config_a
    )
    # 实验 1：不传 turns、不传历史——新消息靠 reducer 并入，计数/历史靠存档帧带回来
    second = await app.ainvoke(
        {"messages": [HumanMessage(content=payload.second_message)]}, config_a
    )
    # 实验 2：换个 thread_id，对之前的对话一无所知
    isolated = await app.ainvoke(
        {"messages": [HumanMessage(content=payload.isolation_question)], "turns": 0}, config_b
    )
    # 实验 3：读最近一帧
    snap_a = await app.aget_state(config_a)
    snap_b = await app.aget_state(config_b)
    frames = await _frames(app, config_a, limit=10)

    return {
        "thread_id": thread_a,
        "other_thread_id": thread_b,
        "first_reply": _text(first["messages"][-1]),
        "second_reply": _text(second["messages"][-1]),
        "isolated_reply": _text(isolated["messages"][-1]),
        "turns": int(second.get("turns") or 0),
        "messages_count": len(second.get("messages") or []),
        "snapshot": _snapshot_view(snap_a),
        "other_snapshot": _snapshot_view(snap_b),
        "frames": frames,
        "graph": _graph_structure(
            app,
            labels={
                ("__start__", "chat"): "入口",
                ("chat", "__end__"): "跑完即存档（超步边界）",
            },
        ),
        "notes": [
            f"两个会话：{thread_a}（两轮）与 {thread_b}（隔离）——快照互不可见。",
            "实验 1 的第二次 invoke 只带了新消息：turns=2 与历史都来自 checkpointer，不是 Python 变量。",
            "实验 3：next=() 表示图已跑完；中断时 next 才会非空（见 ② 确认门）。",
        ],
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


async def chat(db: AsyncSession, payload: Day11ChatRequest) -> dict[str, Any]:
    """交互版续聊：同一个 thread_id 的多次请求共享存档（跨请求续跑）。

    与 Day 06 的 SessionState 对比：那边是"自己攒 messages"，
    这边只把新 HumanMessage 和同一个 thread_id 交给图，历史由 checkpointer 读出来。
    """
    _require_langgraph()
    from langchain_core.messages import HumanMessage

    saver = _memory_saver()
    thread_id = payload.thread_id or _thread_id("user")
    config_a = _config(thread_id)
    if payload.reset:
        await saver.adelete_thread(thread_id)

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day11-chat")
    app = _build_chat_graph(llm, saver)

    started = time.perf_counter()
    snap = await app.aget_state(config_a)
    is_new = not (getattr(snap, "values", None) or {})
    initial: dict[str, Any] = {"messages": [HumanMessage(content=payload.message)]}
    if is_new:
        initial["turns"] = 0  # 新会话要初始化普通字段；续跑时不能重置

    state = await app.ainvoke(initial, config_a)
    snapshot = _snapshot_view(await app.aget_state(config_a))
    frames = await _frames(app, config_a, limit=20)

    return {
        "thread_id": thread_id,
        "reply": _text(state["messages"][-1]),
        "turns": int(state.get("turns") or 0),
        "messages_count": len(state.get("messages") or []),
        "is_new_thread": is_new,
        "snapshot": snapshot,
        "frames": frames,
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ============================================================
# 2. 图级确认门：interrupt() + Command(resume=)
# ============================================================


def _build_approval_tools() -> list[Any]:
    """确认门用的两个工具：一个只读、一个敏感（write）。

    敏感工具才是真正执行退款的地方——它只在审批通过后才可能被调用，
    这就是"副作用放在 interrupt 之后"的落地方式。
    """
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def get_order(order_id: str) -> str:
        """查询订单金额与状态（只读）。

        何时用：用户问某个订单的金额、状态、是否已支付时。
        何时不用：用户只是问退款政策（不涉及具体订单）。

        Args:
            order_id: 订单号，形如 A1001。
        """
        orders = {
            "A1001": {"order_id": "A1001", "amount": 299, "status": "已支付"},
            "A1002": {"order_id": "A1002", "amount": 88, "status": "已支付"},
        }
        data = orders.get(order_id, {"order_id": order_id, "error": "订单不存在"})
        return json.dumps(data, ensure_ascii=False)

    @tool(parse_docstring=True)
    def refund_order(order_id: str, amount: float) -> str:
        """为订单执行退款（敏感资金操作，需要人工审批）。

        何时用：用户明确要求退款，且已确认订单号与金额时。
        何时不用：用户只是在咨询退款政策，或订单尚未确认。

        Args:
            order_id: 订单号，形如 A1001。
            amount: 退款金额（元）。
        """
        return f"已为订单 {order_id} 退款 {amount:g} 元"

    return [get_order, refund_order]


def _build_approval_graph(llm: Any, tools: list[Any], checkpointer: Any) -> Any:
    """agent（决策）→ approval（只有 interrupt，零副作用）→ tools（真正执行）。

    注意 tools 节点是手写的，不用 prebuilt `ToolNode`：只有手写才能按 `approved`
    拦截敏感调用，并为被拒的调用补一条"已拦截"的 ToolMessage 保持配对。
    """
    from langchain_core.messages import ToolMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.types import interrupt

    tool_map = {item.name: item for item in tools}
    llm_with_tools = llm.bind_tools(tools)

    class ApprovalState(TypedDict):
        messages: Annotated[list, add_messages]
        approved: bool
        patched_args: dict[str, Any]
        path: Annotated[list[str], operator.add]

    async def agent_node(state: ApprovalState) -> dict[str, Any]:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response], "path": ["agent"]}

    def approval_node(state: ApprovalState) -> dict[str, Any]:
        """这个节点里只有一个 interrupt：被重放 100 遍也不会产生任何副作用。"""
        last = state["messages"][-1]
        call = next(
            (item for item in getattr(last, "tool_calls", None) or [] if item.get("name") in DANGEROUS),
            None,
        )
        if call is None:  # 兜底：没有敏感调用就不拦
            return {"approved": True, "patched_args": {}, "path": ["approval:skip"]}

        decision = interrupt(
            {
                "type": "confirm",
                "tool": call.get("name"),
                "args": call.get("args") or {},
                "prompt": "敏感操作：确认执行退款吗？回复 true / false，或传改过的参数对象",
            }
        )
        # resume 值在这里才拿到：bool 表示批/驳，dict 表示"改完参数后放行"（练习 2）
        if isinstance(decision, dict):
            return {
                "approved": True,
                "patched_args": {str(key): value for key, value in decision.items()},
                "path": ["approval:patched"],
            }
        return {
            "approved": bool(decision),
            "patched_args": {},
            "path": ["approval:approved" if decision else "approval:rejected"],
        }

    async def tools_node(state: ApprovalState) -> dict[str, Any]:
        """执行最后一条 AIMessage 里的所有 tool_calls；敏感调用先看审批结果。"""
        last = state["messages"][-1]
        approved = bool(state.get("approved"))
        patched = dict(state.get("patched_args") or {})
        outputs: list[Any] = []
        for call in list(getattr(last, "tool_calls", None) or []):
            name = str(call.get("name", ""))
            call_id = str(call.get("id", ""))
            if name in DANGEROUS and not approved:
                outputs.append(
                    ToolMessage(
                        content="【已拦截】人工审批未通过，该操作未执行。",
                        tool_call_id=call_id,
                    )
                )
                continue
            args = dict(call.get("args") or {})
            if name in DANGEROUS and patched:
                args.update(patched)  # 人工改过的参数覆盖模型原始参数
            fn = tool_map.get(name)
            if fn is None:
                content = f"未知工具 {name}，可用工具：{list(tool_map)}"
            else:
                try:
                    content = str(await fn.ainvoke(args))
                except Exception as exc:  # noqa: BLE001  异常也包成 ToolMessage，配对不能断
                    content = f"工具执行失败：{type(exc).__name__}: {exc}"
            if name in DANGEROUS and patched:
                content = f"{content}（参数已按人工审批修改：{_jsonable(patched)}）"
            outputs.append(ToolMessage(content=content, tool_call_id=call_id))
        return {"messages": outputs, "path": ["tools"]}

    def after_agent(state: ApprovalState) -> str:
        """无 tool_calls → 收工；命中敏感工具 → 审批；其余 → 直接执行。"""
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        names = {str(item.get("name")) for item in last.tool_calls}
        return "approval" if names & DANGEROUS else "tools"

    builder = StateGraph(ApprovalState)
    builder.add_node("agent", agent_node)
    builder.add_node("approval", approval_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", after_agent, {"approval": "approval", "tools": "tools", END: END}
    )
    builder.add_edge("approval", "tools")
    builder.add_edge("tools", "agent")  # Day 10 那条回头边：工具结果回给模型组织语言
    return builder.compile(checkpointer=checkpointer)


async def start_approval(db: AsyncSession, payload: Day11ApprovalStartRequest) -> dict[str, Any]:
    """第一次调用：图走到 approval 节点冻结，state 已存档，返回待审批载荷。"""
    _require_langgraph()
    from langchain_core.messages import HumanMessage, SystemMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day11-approval")
    tools = _build_approval_tools()
    app = _build_approval_graph(llm, tools, _memory_saver())

    thread_id = payload.thread_id or _thread_id("ticket")
    config_a = _config(thread_id)

    started = time.perf_counter()
    state = await app.ainvoke(
        {
            "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=payload.question)],
            "approved": False,
            "patched_args": {},
            "path": [],
        },
        config_a,
    )
    snap = await app.aget_state(config_a)
    view = _snapshot_view(snap)
    paused = bool(view["next"])

    interrupt_payload = None
    for task in view["tasks"]:
        if task["interrupts"]:
            interrupt_payload = task["interrupts"][0]
            break

    if paused and interrupt_payload:
        PENDING_APPROVALS[thread_id] = {
            "thread_id": thread_id,
            "question": payload.question,
            "tool": str(interrupt_payload.get("tool", "")),
            "args": interrupt_payload.get("args") or {},
            "created_ts": time.time(),
            "created_at": view["created_at"],
        }
    else:
        PENDING_APPROVALS.pop(thread_id, None)

    return {
        "thread_id": thread_id,
        "paused": paused,
        "next": view["next"],
        "interrupt": interrupt_payload,
        "answer": _last_answer(state.get("messages") or []),
        "snapshot": view,
        "graph": _graph_structure(
            app,
            labels={
                ("__start__", "agent"): "入口",
                ("agent", "approval"): "命中敏感工具（refund_order）",
                ("agent", "tools"): "只读工具直接执行",
                ("agent", "__end__"): "无 tool_calls",
                ("approval", "tools"): "resume 后继续",
                ("tools", "agent"): "★ 回头边：工具结果回给模型",
            },
        ),
        "note": "暂停点已存档：进程可以死，thread_id 不变就能从这一帧继续（见 ② 的恢复按钮）。",
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


async def resume_approval(
    db: AsyncSession, payload: Day11ApprovalResumeRequest
) -> dict[str, Any]:
    """恢复：同一个 thread_id + Command(resume=决定)，图从 approval 帧继续跑。

    ``decision`` 三种形态（练习 2）：
    - ``True`` 放行；``False`` 拦截；``{"amount": 199}`` 改参数后放行。
    """
    _require_langgraph()
    from langgraph.types import Command

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day11-approval-resume")
    app = _build_approval_graph(llm, _build_approval_tools(), _memory_saver())
    config_a = _config(payload.thread_id)

    started = time.perf_counter()
    state = await app.ainvoke(Command(resume=payload.decision), config_a)
    snap = await app.aget_state(config_a)
    view = _snapshot_view(snap)
    PENDING_APPROVALS.pop(payload.thread_id, None)

    patched = dict(state.get("patched_args") or {})
    return {
        "thread_id": payload.thread_id,
        "approved": bool(state.get("approved")),
        "patched_args": _jsonable(patched),
        "final_answer": _last_answer(state.get("messages") or []) or "（模型没有产出最终回答）",
        "tool_results": _tool_results(state.get("messages") or []),
        "next": view["next"],
        "messages_count": len(state.get("messages") or []),
        "note": (
            "approval 节点被重放过，但它里面只有 interrupt，所以没有重复副作用；"
            "退款只在 tools 节点、approved=True 时执行一次。"
        ),
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


def list_pending() -> list[dict[str, Any]]:
    """列出等待人工审批的会话（练习 3 的"审批队列"）。"""
    now = time.time()
    rows = []
    for item in list(PENDING_APPROVALS.values()):
        rows.append(
            {
                "thread_id": item["thread_id"],
                "question": item["question"],
                "tool": item["tool"],
                "args": _jsonable(item["args"]),
                "wait_seconds": round(now - item["created_ts"], 1),
                "created_at": item.get("created_at"),
            }
        )
    rows.sort(key=lambda row: row["wait_seconds"], reverse=True)
    return rows


async def expire_approvals(db: AsyncSession, payload: Day11ExpireRequest) -> dict[str, Any]:
    """练习 3：模拟定时任务扫描超时审批，用 Command(resume=False) 自动驳回。

    ``max_age_seconds=0`` 表示"把队列里的审批一律视为已超时"——课堂演示用；
    生产里由 cron 传入 86400（24 小时）。
    """
    _require_langgraph()
    from langgraph.types import Command

    now = time.time()
    targets = [
        item
        for item in list(PENDING_APPROVALS.values())
        if (payload.thread_id is None or item["thread_id"] == payload.thread_id)
        and now - item["created_ts"] >= payload.max_age_seconds
    ]
    if not targets:
        return {"expired": [], "remaining": len(PENDING_APPROVALS), "note": "没有符合条件的待审批会话"}

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day11-approval-expire")
    app = _build_approval_graph(llm, _build_approval_tools(), _memory_saver())

    expired: list[dict[str, Any]] = []
    for item in targets:
        thread_id = item["thread_id"]
        waited = round(now - item["created_ts"], 1)
        state = await app.ainvoke(Command(resume=False), _config(thread_id))
        PENDING_APPROVALS.pop(thread_id, None)
        expired.append(
            {
                "thread_id": thread_id,
                "waited_seconds": waited,
                "answer": _last_answer(state.get("messages") or []) or "（模型没有产出最终回答）",
                "tool_results": _tool_results(state.get("messages") or []),
            }
        )

    return {
        "expired": expired,
        "remaining": len(PENDING_APPROVALS),
        "note": "自动驳回同样走 Command(resume=False)：模型收到【已拦截】的 ToolMessage 后如实转达用户。",
    }


# ============================================================
# 3. 节点重放与副作用（纯本地，不花 token）
# ============================================================


async def replay_demo() -> dict[str, Any]:
    """对照实验：interrupt 前有副作用 vs 只有 interrupt。

    两个图都跑"暂停 → 恢复"一遍，统计副作用真实执行次数：
    - 反面：副作用写在 interrupt 之前 → 恢复时节点重放，执行 2 次；
    - 正面：决策 / 审批 / 执行三节点拆开 → 执行 1 次。
    """
    _require_langgraph()
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    effects_bad: list[str] = []
    effects_good: list[str] = []

    class BadState(TypedDict):
        approved: bool
        runs: int

    def bad_node(state: BadState) -> dict[str, Any]:
        effects_bad.append("扣款")  # ⚠️ 副作用在 interrupt 之前
        decision = interrupt({"type": "confirm", "prompt": "确认扣款？"})
        return {"approved": bool(decision), "runs": state["runs"] + 1}

    bad_builder = StateGraph(BadState)
    bad_builder.add_node("bad_approval", bad_node)
    bad_builder.add_edge(START, "bad_approval")
    bad_builder.add_edge("bad_approval", END)
    bad_app = bad_builder.compile(checkpointer=MemorySaver())
    bad_cfg = _config(_thread_id("replay-bad"))

    await bad_app.ainvoke({"approved": False, "runs": 0}, bad_cfg)
    bad_before = len(effects_bad)
    await bad_app.ainvoke(Command(resume=True), bad_cfg)
    bad_after = len(effects_bad)

    class GoodState(TypedDict):
        approved: bool
        cycles: int

    def decide_node(state: GoodState) -> dict[str, Any]:
        # 决策节点：纯读 state、写 state，不做任何外部动作
        return {"cycles": state["cycles"] + 1}

    def approval_node(state: GoodState) -> dict[str, Any]:
        decision = interrupt({"type": "confirm", "prompt": "确认扣款？"})
        return {"approved": bool(decision)}

    def execute_node(state: GoodState) -> dict[str, Any]:
        if state["approved"]:
            effects_good.append("扣款")  # ✅ 副作用在拿到 resume 值之后
        return {}

    good_builder = StateGraph(GoodState)
    good_builder.add_node("decide", decide_node)
    good_builder.add_node("approval", approval_node)
    good_builder.add_node("execute", execute_node)
    good_builder.add_edge(START, "decide")
    good_builder.add_edge("decide", "approval")
    good_builder.add_edge("approval", "execute")
    good_builder.add_edge("execute", END)
    good_app = good_builder.compile(checkpointer=MemorySaver())
    good_cfg = _config(_thread_id("replay-good"))

    await good_app.ainvoke({"approved": False, "cycles": 0}, good_cfg)
    good_before = len(effects_good)
    await good_app.ainvoke(Command(resume=True), good_cfg)
    good_after = len(effects_good)

    return {
        "experiments": [
            {
                "key": "bad",
                "title": "❌ 反面：副作用写在 interrupt() 之前",
                "before_resume": bad_before,
                "after_resume": bad_after,
                "verdict": "暂停时已经扣了一次（恢复前就发生），恢复时节点从头重放又扣一次——真实资损。",
            },
            {
                "key": "good",
                "title": "✅ 正面：决策 / 审批 / 执行拆成三个节点",
                "before_resume": good_before,
                "after_resume": good_after,
                "verdict": "审批节点只有 interrupt，重放安全；执行节点在 resume 之后才跑，只扣一次。",
            },
        ],
        "rules": RULES,
        "note": "写完节点自问一句：这个节点被重放两遍，会不会扣两次钱？",
    }


# ============================================================
# 4. 时间旅行：翻历史 → 改参数 → 无输入重跑
# ============================================================


def _build_time_travel_graph(llm: Any, checkpointer: Any) -> Any:
    """set_topic（早期定参数）→ answer（烧 token 的那一步），两帧之间可回退。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class TravelState(TypedDict):
        messages: Annotated[list, add_messages]
        topic: str

    def set_topic(state: TravelState) -> dict[str, Any]:
        # 第一帧：把主题写进 state（模拟"参数在早期就定了"的场景）
        humans = [item for item in state["messages"] if getattr(item, "type", "") == "human"]
        return {"topic": _text(humans[-1]) if humans else ""}

    async def answer_node(state: TravelState) -> dict[str, Any]:
        reply = await llm.ainvoke(
            [
                SystemMessage(content="你是技术科普助手，回答控制在一句话以内。"),
                HumanMessage(content=f"请用一句话介绍「{state['topic']}」的核心概念"),
            ]
        )
        return {"messages": [reply]}

    builder = StateGraph(TravelState)
    builder.add_node("set_topic", set_topic)
    builder.add_node("answer", answer_node)
    builder.add_edge(START, "set_topic")
    builder.add_edge("set_topic", "answer")
    builder.add_edge("answer", END)
    return builder.compile(checkpointer=checkpointer)


async def run_time_travel(
    db: AsyncSession, payload: Day11TimeTravelRequest
) -> dict[str, Any]:
    """三步走：get_state_history 选帧 → update_state 分叉改写 → invoke(None) 重跑。"""
    _require_langgraph()
    from langchain_core.messages import HumanMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day11-time-travel")
    app = _build_time_travel_graph(llm, _memory_saver())

    thread_id = _thread_id("research")
    config_a = _config(thread_id)

    started = time.perf_counter()
    first = await app.ainvoke(
        {"messages": [HumanMessage(content=payload.topic)], "topic": ""}, config_a
    )
    original_answer = _text(first["messages"][-1])
    frames_before = await _frames(app, config_a, limit=20)

    target = None
    async for snap in app.aget_state_history(config_a):
        if tuple(snap.next or ()) == ("answer",):
            target = snap
            break
    if target is None:  # pragma: no cover - 图结构固定时不会发生
        raise ValueError("没有找到 answer 之前的存档帧，无法时间旅行")

    fork_id = _checkpoint_id_of(target)
    # 直接沿用历史帧的完整 config：里面除了 thread_id / checkpoint_id，还有 checkpoint_ns，
    # 自己拼一个缺字段的 config 会在 update_state 时被 checkpointer 拒绝。
    old_cfg = {"configurable": dict((target.config or {}).get("configurable") or {})}
    # ② 从选定帧改写 state：会分叉出新链（原历史不动）。
    #    ⚠️ 必须用 update_state 返回的新 config 继续跑——它指向刚写出的分叉帧；
    #    继续用 old_cfg（指向分叉前的旧帧）会读到改之前的 topic。
    fork_cfg = await app.aupdate_state(old_cfg, {"topic": payload.new_topic}, as_node="set_topic")
    # ③ 不给新输入，从这一帧重新往后跑
    new_state = await app.ainvoke(None, fork_cfg)
    frames_after = await _frames(app, config_a, limit=30)

    original_last_id = frames_before[0]["checkpoint_id"] if frames_before else ""
    original_preserved = any(row["checkpoint_id"] == original_last_id for row in frames_after)

    return {
        "thread_id": thread_id,
        "original_answer": original_answer,
        "new_answer": _text(new_state["messages"][-1]),
        "frames_before": frames_before,
        "frames_after": frames_after,
        "fork_checkpoint_id": fork_id,
        "original_preserved": original_preserved,
        "note": (
            f"从 checkpoint {fork_id[:8]}… 分叉改写 topic={payload.new_topic!r} 后重跑；"
            "原历史帧仍在（分叉不是覆盖）。调试时不必重跑前面烧 token 的步骤。"
        ),
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ============================================================
# 5. SQLite 存档：换连接 = 换进程，会话仍在
# ============================================================


async def run_sqlite(db: AsyncSession, payload: Day11SqliteRequest) -> dict[str, Any]:
    """用 AsyncSqliteSaver 落盘：每个请求开一次连接，等价于"进程重启后读档"。

    mode=first 写入一轮；mode=resume 只带新消息——历史从 day11_checkpoints.db 里读。
    真正重启后端后再点 resume 也能接上，这就是 MemorySaver 做不到的事。
    """
    _require_langgraph()
    try:
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError as exc:  # pragma: no cover - 只在缺依赖的环境触发
        raise ValueError(f"未安装 langgraph-checkpoint-sqlite，请先执行：{INSTALL_HINT}") from exc

    from langchain_core.messages import HumanMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day11-sqlite")
    thread_id = payload.thread_id or "sqlite-user-42"
    config_a = _config(thread_id)

    default_message = (
        "记住：我最喜欢的编程语言是 Python"
        if payload.mode == "first"
        else "我最喜欢的编程语言是什么？"
    )
    message = payload.message or default_message

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    async with AsyncSqliteSaver.from_conn_string(str(SQLITE_FILE)) as saver:
        # 建表（CREATE TABLE IF NOT EXISTS）：只读 / 删除也必须先 setup，
        # 否则第一次打开空文件时会报 no such table: checkpoints。
        await saver.setup()
        if payload.reset:
            await saver.adelete_thread(thread_id)
        app = _build_chat_graph(llm, saver)
        snap = await app.aget_state(config_a)
        is_new = not (getattr(snap, "values", None) or {})
        initial: dict[str, Any] = {"messages": [HumanMessage(content=message)]}
        if is_new:
            initial["turns"] = 0
        state = await app.ainvoke(initial, config_a)
        frames = await _frames(app, config_a, limit=20)
        view = _snapshot_view(await app.aget_state(config_a))

    size_kb = round(SQLITE_FILE.stat().st_size / 1024, 1) if SQLITE_FILE.exists() else 0.0
    return {
        "mode": payload.mode,
        "thread_id": thread_id,
        "db_path": str(SQLITE_FILE),
        "db_size_kb": size_kb,
        "reply": _text(state["messages"][-1]),
        "turns": int(state.get("turns") or 0),
        "messages_count": len(state.get("messages") or []),
        "snapshot": view,
        "frames": frames,
        "note": "每次请求都重新打开 SQLite 连接并重新编译图——会话仍在，说明 state 活在文件里而不是进程里。",
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ============================================================
# 6. 静态概念数据（供前端渲染）
# ============================================================


async def mapping() -> dict[str, Any]:
    """Day 11 的概念表：快照四要素 / 两套暂停机制 / 三档存档器 / 六个坑 / 时间旅行三步。"""
    return {
        "snapshot_fields": SNAPSHOT_FIELDS,
        "pause_mechanisms": PAUSE_MECHANISMS,
        "savers": SAVERS,
        "pitfalls": PITFALLS,
        "time_travel_steps": TIME_TRAVEL_STEPS,
        "rules": RULES,
        "presets": PRESETS,
        "install_hint": INSTALL_HINT,
        "langgraph_version": await asyncio.to_thread(_version, "langgraph"),
        "checkpoint_sqlite_version": await asyncio.to_thread(
            _version, "langgraph-checkpoint-sqlite"
        ),
    }


__all__ = [
    "INSTALL_HINT",
    "PAUSE_MECHANISMS",
    "PENDING_APPROVALS",
    "PITFALLS",
    "PRESETS",
    "RULES",
    "SAVERS",
    "SNAPSHOT_FIELDS",
    "SQLITE_FILE",
    "TIME_TRAVEL_STEPS",
    "chat",
    "expire_approvals",
    "list_pending",
    "mapping",
    "replay_demo",
    "resume_approval",
    "run_basic",
    "run_sqlite",
    "run_time_travel",
    "start_approval",
]
