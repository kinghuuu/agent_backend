"""Day 10 LangGraph 状态图：把 while 循环画成一张可中断的图。

课程 day10 的五个脚本在这里收成一个模块，每个函数对应一个脚本：

    状态与 reducer → ``state_reducer_demo``  （01：同一个图只改一行 Annotated，行为完全不同）
    最小线性图     → ``run_min_graph``       （02：retrieve → generate + 逐节点观察补丁）
    条件边路由     → ``run_conditional``     （03：classify + route + 四条分支，含练习 1 转人工）
    完整 ReAct 环  → ``run_react``           （04：agent ⇄ tools + 双刹车 + 降级交付，含练习 2/3）
    并行扇出       → ``run_parallel``        （05：静态扇出 / Send 动态扇出 / 子图）

四个贯穿全篇的结论：
1. LangChain 管"一步怎么算"，LangGraph 管"多步怎么流转"——两者不同层，节点内部照样用 LangChain 组件；
2. 节点返回的是**补丁**不是完整 state，合并语义由字段声明的 reducer 决定（不写 reducer 的 list 会被覆盖）；
3. `break` 变成一条边、`continue` 变成另一条边、`while` 变成一条回头边——控制流从此可视化；
4. `recursion_limit` 是保险丝不是刹车，捕获 ``GraphRecursionError`` 之后必须降级交付（Day 05）。

langgraph 与 day09 的 langchain 一样全部延迟导入：没装依赖时接口返回可读的 400，
而不是让整个应用起不来。
"""

import asyncio
import operator
import time
from typing import Annotated, Any, TypedDict

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import (
    Day10ConditionalRequest,
    Day10MinGraphRequest,
    Day10ParallelRequest,
    Day10ReactRequest,
)
from app.modules.study.services.day09_langchain import build_chat_model
from app.modules.study.services.llm import resolve_model

# 安装提示：未装依赖时所有 Day 10 接口都返回这句话
INSTALL_HINT = "pip install -U langgraph"

# 课程 1.2 的三件套（State / Node / Edge）
TRIPLE_PIECES: list[dict[str, str]] = [
    {
        "piece": "State",
        "what": "一个 TypedDict，全图共享的唯一状态",
        "handwritten": "Day 01 里的 messages 局部变量",
        "key_point": "节点不持有状态，只读写这份；字段用 Annotated[..., reducer] 声明合并语义",
    },
    {
        "piece": "Node",
        "what": "普通函数 state -> 增量补丁 dict",
        "handwritten": "llm.invoke 那一段 / for call in tool_calls 那一段",
        "key_point": "返回值是补丁不是完整 state：没返回的字段保持原值，千万别原地改 state",
    },
    {
        "piece": "Edge",
        "what": "普通边（固定去哪）+ 条件边（按 state 算去哪）",
        "handwritten": "if not tool_calls: break 这段控制流",
        "key_point": "有环就是靠条件边指回上游节点——环 = 一条回头边",
    },
]

# 课程 1.3 的 reducer 对照表
REDUCER_TABLE: list[dict[str, str]] = [
    {
        "declare": "log: list[str]",
        "merge": "覆盖（后写的赢）",
        "use": "只有一个节点写的字段，如 answer",
    },
    {
        "declare": "log: Annotated[list, operator.add]",
        "merge": "追加",
        "use": "累积日志、累积检索结果（多节点并发写）",
    },
    {
        "declare": "messages: Annotated[list, add_messages]",
        "merge": "按 id 追加 + 同 id 更新",
        "use": "对话历史（LangGraph 内置）",
    },
    {
        "declare": "steps: int",
        "merge": "覆盖（节点里要写 state['steps'] + 1）",
        "use": "计数器",
    },
]

# 课程 1.6 的手写 while ↔ 图 对照表
WHILE_VS_GRAPH: list[dict[str, str]] = [
    {"handwritten": "messages = [] 局部变量", "graph": "class State(TypedDict): messages: Annotated[list, add_messages]", "gap": "状态从局部变量变成图的显式契约，可持久化"},
    {"handwritten": "llm.invoke(messages) 那一段", "graph": "agent 节点", "gap": "逻辑一样，但可被单独测试/替换"},
    {"handwritten": "for call in ai.tool_calls: ...", "graph": "tools 节点", "gap": "一样，但要一条不落地回 ToolMessage"},
    {"handwritten": "if not ai.tool_calls: break", "graph": "should_continue() 条件边", "gap": "break 变成一条边，continue 变成另一条边"},
    {"handwritten": "while step < MAX_STEPS", "graph": "recursion_limit + 条件边里的 steps 判断", "gap": "刹车从代码里挪到图配置里"},
    {"handwritten": "while True: 本身", "graph": "add_edge('tools', 'agent') 这条回头边", "gap": "这条边就是『环』"},
    {"handwritten": "想看跑了几步只能 print", "graph": "draw_mermaid() / draw_ascii()", "gap": "图是可画出来的——这是最大的收益"},
]

# 课程 2.2 的 stream_mode 取值表
STREAM_MODES: list[dict[str, str]] = [
    {"mode": "updates", "gives": "每个节点提交的补丁", "when": "调试、日志（本项目 02/04 都用它）"},
    {"mode": "values", "gives": "每个超步后的完整 state", "when": "想看状态怎么长"},
    {"mode": "messages", "gives": "LLM 的逐 token 流", "when": "前台打字机效果（Day 13）"},
    {"mode": "debug", "gives": "极详细的内部事件", "when": "疑难排查"},
]

# 课程第 4 节的六个坑
PITFALLS: list[dict[str, str]] = [
    {
        "no": "1",
        "title": "messages 忘了 add_messages，Agent 失忆",
        "symptom": "第一轮正常，第二轮模型说『你刚才没告诉我订单号』",
        "cause": "第二次写入把 messages 覆盖成一条",
        "fix": "messages: Annotated[list, add_messages]；所有会被多个节点/多轮写入的 list 字段一律要 reducer",
    },
    {
        "no": "2",
        "title": "条件边返回了映射表里没有的字符串",
        "symptom": "KeyError: 'tool' 或莫名其妙直接结束",
        "cause": "route() 返回 'tool'，映射表写的是 'tools'",
        "fix": "返回值用 Literal 约束；route() 加兜底分支；写完先 draw_mermaid() 看一眼",
    },
    {
        "no": "3",
        "title": "环没有语义出口，靠 recursion_limit 硬停",
        "symptom": "跑满 25 步才报错，中间烧了一堆 token",
        "cause": "should_continue 只看 tool_calls，模型反复调同一个工具（Day 05 的变体自旋）",
        "fix": "条件边里同时判 steps（>= N 就去 degrade 节点）；工具层拦截同参重复调用",
    },
    {
        "no": "4",
        "title": "在节点里直接改 state（mutation）",
        "symptom": "有时对有时错，stream 里看到的补丁和最终 state 不一致",
        "cause": "state['messages'].append(x) 原地改了对象，却没有返回补丁",
        "fix": "节点只读 state、只返回新 dict：要加消息就 return {'messages': [x]}",
    },
    {
        "no": "5",
        "title": "把路由决策和业务执行写在同一个节点",
        "symptom": "图上只有两个节点，draw 出来毫无信息量；单测一个分支要跑全流程",
        "cause": "if/elif 塞在节点里，图退化成一条线",
        "fix": "拆成 classify（写标签）+ route（纯函数选路）+ 各业务节点",
    },
    {
        "no": "6",
        "title": "一个 AIMessage 带 N 个 tool_calls，只回了 1 条 ToolMessage",
        "symptom": "第二轮报 400 An issue with your tool_calls（各厂商措辞不同）",
        "cause": "Day 03 的配对铁律在图里同样成立，错误被节点边界掩盖了",
        "fix": "tools_node 里 for call in last.tool_calls 一条不落；异常也包成 ToolMessage，不能 raise 跳过",
    },
]

# 课程 2.5 的 Send 三条规矩
SEND_RULES: list[str] = [
    "目标节点收到的是 Send 的 payload，不是全局 state",
    "写回仍走全局 state 的 channel，所以被并发写的字段必须有 reducer",
    "分支数在运行时决定，适合『按文档片段逐个摘要』这类 map-reduce",
]

# 各实验的预设问题（前端一键填入）
PRESETS: dict[str, list[str]] = {
    "min_graph": ["XR-500 Pro 保修多久？", "能退款吗？", "今天天气怎么样？"],
    "conditional": ["SO20261101001 到哪了？", "能退款吗？", "你们几点上班？"],
    "react": [
        "帮我查下 SO20261101001 到哪了，顺便说下 XR-500 Pro 的退款政策",
        "SO9999 什么状态？",
    ],
    "parallel": ["退款；订单物流；发票政策"],
}

# ReAct 图的系统提示词（Day 02 的四段式：角色 / 规则 / 输出格式 / 示例）
SYSTEM_PROMPT = """你是售后 Agent。
规则：
1. 涉及订单物流、退款政策等实时或专有信息，必须调用工具，不得凭记忆回答；
2. 工具返回错误时先自查参数，最多重试一次；仍失败则如实告知用户；
3. 不知道就说不知道，并说明需要转人工。
输出格式：先用一句话给结论，再补充依据；引用工具结果时末尾标注来源。"""

# 工具持续失败时用的提示（练习 2：观察语义刹车后是否优雅收尾）
DEGRADE_ADVICE = "工具已不可用，请基于已知信息作答，并明确说明信息可能不完整"
DEGRADE_NOTE = f"（系统提示：{DEGRADE_ADVICE}）"


# ============================================================
# 0. 公共设施：依赖探测 / 序列化 / 图结构
# ============================================================


def _require_langgraph() -> None:
    """依赖探测：没装 langgraph 时给安装命令，而不是 ImportError 堆栈。"""
    try:
        import langgraph.graph  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 只在缺依赖的环境触发
        raise ValueError(f"未安装 langgraph，请先执行：{INSTALL_HINT}") from exc


def _text(message: Any) -> str:
    """取消息正文：兼容 str 与「内容块 list」两种返回形态（与 Day 09 同款）。

    部分 OpenAI 兼容网关会把 content 返回成 ``[{"type": "text", "text": ...}]``，
    直接 str() 会得到一坨带花括号的字符串，所以这里统一抽文本。
    """
    content = getattr(message, "content", message)
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


def _jsonable(value: Any) -> Any:
    """把 state 里的值转成可 JSON 序列化的形态（消息对象 → 摘要 dict）。

    不直接用 ``model_dump()``：消息里可能带 ``reasoning_content`` 等供应商私有字段，
    学习页只需要"谁说了什么、有没有 tool_calls"。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if hasattr(value, "type") and hasattr(value, "content"):
        row: dict[str, Any] = {"type": getattr(value, "type", "")}
        message_id = getattr(value, "id", None)
        if message_id:
            row["id"] = message_id
        row["content"] = _text(value)
        tool_calls = getattr(value, "tool_calls", None)
        if tool_calls:
            row["tool_calls"] = [
                {"id": call.get("id", ""), "name": call.get("name", ""), "args": call.get("args", {})}
                for call in tool_calls
            ]
        tool_call_id = getattr(value, "tool_call_id", None)
        if tool_call_id:
            row["tool_call_id"] = tool_call_id
        return row
    return str(value)


def _graph_structure(app: Any, labels: dict[tuple[str, str], str] | None = None) -> dict[str, Any]:
    """从编译好的图里抽出可渲染的结构：节点、边（含条件边标注）、mermaid 源码、ascii。

    ``draw_ascii()`` 需要额外装 grandalf，没装时返回空串 + 提示，
    而 mermaid 源码不依赖任何可选包，所以前端始终有图可看。
    """
    graph = app.get_graph()
    labels = labels or {}
    edges: list[dict[str, Any]] = []
    for edge in graph.edges:
        label = labels.get((edge.source, edge.target), "")
        if not label and getattr(edge, "conditional", False):
            label = "条件边"
        edges.append(
            {
                "source": edge.source,
                "target": edge.target,
                "conditional": bool(getattr(edge, "conditional", False)),
                "label": label,
            }
        )

    ascii_text = ""
    ascii_hint = ""
    try:
        ascii_text = graph.draw_ascii()
    except ImportError:
        ascii_hint = "draw_ascii() 需要额外依赖：pip install grandalf（mermaid 源码不依赖它）"
    except Exception as exc:  # noqa: BLE001  绘图失败不该影响实验本身
        ascii_hint = f"draw_ascii() 失败：{type(exc).__name__}: {exc}"

    return {
        "nodes": list(graph.nodes.keys()),
        "edges": edges,
        "mermaid": graph.draw_mermaid(),
        "ascii": ascii_text,
        "ascii_hint": ascii_hint,
    }


# ============================================================
# 1. 状态与 reducer：同一个图只改一行 Annotated
# ============================================================


async def state_reducer_demo() -> dict[str, Any]:
    """跑四个对照实验，证明"补丁怎么合并"决定 Agent 会不会失忆。

    实验 A：``log: list[str]``             → 后写的覆盖前写的（失忆的根源）
    实验 B：``log: Annotated[list, +]``    → 追加，正确
    实验 C：``messages: Annotated[, add_messages]`` → 按 id 追加 + 同 id 更新
    实验 D：坑——节点图省事返回全量再追加，长度指数级增长

    每个实验都用 ``stream_mode=["updates", "values"]`` 同时拿到"补丁"与"补丁后的状态"，
    所以能同时展示"node_two 读到的是什么"和"写入之后变成什么"。
    """
    _require_langgraph()
    from langgraph.graph import END, START, StateGraph

    async def run_case(state_cls: type, nodes: list[tuple[str, Any]], initial: dict[str, Any]) -> dict[str, Any]:
        """建一张线性图跑一遍，返回逐节点（收到 / 补丁）与最终状态。"""
        builder = StateGraph(state_cls)
        for name, fn in nodes:
            builder.add_node(name, fn)
        builder.add_edge(START, nodes[0][0])
        for (current, _), (nxt, _) in zip(nodes, nodes[1:]):
            builder.add_edge(current, nxt)
        builder.add_edge(nodes[-1][0], END)
        app = builder.compile()

        rows: list[dict[str, Any]] = []
        latest = _jsonable(initial)
        async for mode, chunk in app.astream(initial, stream_mode=["updates", "values"]):
            if mode == "updates":
                for node, patch in chunk.items():
                    rows.append(
                        {"node": node, "received": latest, "patch": _jsonable(patch)}
                    )
            else:
                latest = _jsonable(chunk)
        return {"patches": rows, "result": latest}

    def no_reducer_node(name: str) -> Any:
        def node(state: Any) -> dict[str, Any]:
            return {"log": [name]}

        return node

    def add_reducer_node(name: str) -> Any:
        def node(state: Any) -> dict[str, Any]:
            return {"log": [name]}

        return node

    def total_copy_node(name: str) -> Any:
        """反面教材：把全量再拼一遍交给 reducer——reducer 又加了一次。"""

        def node(state: Any) -> dict[str, Any]:
            return {"log": state["log"] + [name]}

        return node

    # ---- 实验 A：无 reducer ----
    class NoReducerState(TypedDict):
        log: list[str]

    case_a = await run_case(
        NoReducerState,
        [("one", no_reducer_node("one")), ("two", no_reducer_node("two"))],
        {"log": []},
    )

    # ---- 实验 B：operator.add ----
    class AddState(TypedDict):
        log: Annotated[list[str], operator.add]

    case_b = await run_case(
        AddState,
        [("one", add_reducer_node("one")), ("two", add_reducer_node("two"))],
        {"log": []},
    )

    # ---- 实验 C：add_messages（同 id 更新） ----
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph.message import add_messages

    class MessageState(TypedDict):
        messages: Annotated[list, add_messages]

    def say_human(state: Any) -> dict[str, Any]:
        return {"messages": [HumanMessage(content="XR-500 Pro 保修多久？", id="m1")]}

    def say_ai(state: Any) -> dict[str, Any]:
        return {"messages": [AIMessage(content="整机保修两年。", id="m2")]}

    def edit_ai(state: Any) -> dict[str, Any]:
        """同 id 再写一次 = 修订，而不是追加一条。"""
        return {"messages": [AIMessage(content="整机保修 24 个月，电池 12 个月。", id="m2")]}

    case_c = await run_case(
        MessageState,
        [("human", say_human), ("ai", say_ai), ("edit", edit_ai)],
        {"messages": []},
    )

    # ---- 实验 D：reducer 之上再拼全量的坑 ----
    case_d = await run_case(
        AddState,
        [("one", total_copy_node("one")), ("two", total_copy_node("two"))],
        {"log": ["seed"]},
    )

    experiments = [
        {
            "key": "no_reducer",
            "title": "A · 不给 reducer（默认覆盖）",
            "state_def": "log: list[str]",
            "patches": case_a["patches"],
            "result": case_a["result"],
            "verdict": "后写的覆盖前写的：最终只剩 ['two']。注意 node_two 确实『读到』了 ['one']——"
            "写语义错、读语义对，这正是这个 bug 难查的原因。",
        },
        {
            "key": "operator_add",
            "title": "B · operator.add（追加）",
            "state_def": "log: Annotated[list[str], operator.add]",
            "patches": case_b["patches"],
            "result": case_b["result"],
            "verdict": "追加正确：['one', 'two']。节点只返回**新增的那一条**，合并交给 reducer。",
        },
        {
            "key": "add_messages",
            "title": "C · add_messages（按 id 追加 + 同 id 更新）",
            "state_def": "messages: Annotated[list, add_messages]",
            "patches": case_c["patches"],
            "result": case_c["result"],
            "verdict": "第 3 个节点用同一个 id=m2 再写一次，结果不是 3 条而是 2 条——"
            "同 id 是『修订』。Day 09 说的消息编辑就靠这个。",
        },
        {
            "key": "total_copy",
            "title": "D · 坑：节点返回全量再让 reducer 追加",
            "state_def": "log: Annotated[list[str], operator.add]（节点返回 state['log'] + ['x']）",
            "patches": case_d["patches"],
            "result": case_d["result"],
            "verdict": "['seed'] 被复制了两遍：节点返回全量 → reducer 又加一次。"
            "多跑几轮长度会指数级增长——节点永远只返回新增项。",
        },
    ]

    return {
        "experiments": experiments,
        "reducer_table": REDUCER_TABLE,
        "triple_pieces": TRIPLE_PIECES,
        "takeaway": "90% 的新手第一个 bug 都出在这里：不给 messages 加 reducer，第二个节点一写就把它覆盖成一条，"
        "Agent 瞬间失忆。判断标准很简单——**这个 list 字段会不会被多个节点/多轮写入**，会就要 reducer。",
    }


# ============================================================
# 2. 最小线性图：检索 → 生成，并看见每一步
# ============================================================

# 假知识库：真实项目里换成 Day 07 的 rag.search
KB = {
    "保修": "XR-500 Pro 整机保修 24 个月，电池 12 个月。[来源: 产品手册 v3.2]",
    "退款": "签收 7 天内可无理由退款，需保留原包装。[来源: 售后政策 2026]",
}


async def run_min_graph(db: AsyncSession, payload: Day10MinGraphRequest) -> dict[str, Any]:
    """跑一张 START → retrieve → generate → END 的线性图。

    重点不是回答，而是 ``stream_mode=['updates', 'values']`` 让人看见两件事：
    每个节点交了什么**补丁**，以及**完整状态**是怎么一步步长出来的。
    """
    _require_langgraph()
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import END, START, StateGraph

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day10-min-graph")

    class State(TypedDict):
        question: str
        docs: list[str]  # 只被 retrieve 写 → 不需要 reducer
        answer: str  # 只被 generate 写 → 覆盖语义正好
        trace: Annotated[list[str], operator.add]  # 两个节点都写 → 必须追加

    def retrieve(state: State) -> dict[str, Any]:
        """假检索：命中关键词就取知识库条目（真实项目换 Day 07 的混合检索）。"""
        hits = [value for key, value in KB.items() if key in state["question"]]
        return {"docs": hits or ["（知识库无命中）"], "trace": ["retrieve"]}

    async def generate(state: State) -> dict[str, Any]:
        """把检索结果塞进【资料】段，要求"只依据资料回答、末尾标来源"。"""
        context = "\n".join(f"- {doc}" for doc in state["docs"])
        system = SystemMessage(
            content="你是客服 Agent。只能依据【资料】回答；资料里没有就说不知道，"
            "并说明需要转人工。回答末尾标注来源。"
        )
        human = HumanMessage(content=f"【资料】\n{context}\n\n【问题】{state['question']}")
        response = await llm.ainvoke([system, human])
        return {"answer": _text(response), "trace": ["generate"]}

    builder = StateGraph(State)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate", generate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    app = builder.compile()

    events: list[dict[str, Any]] = []
    latest: dict[str, Any] = {}
    started = time.perf_counter()
    async for mode, chunk in app.astream(
        {"question": payload.question, "docs": [], "answer": "", "trace": []},
        stream_mode=["updates", "values"],
    ):
        if mode == "updates":
            for node, patch in chunk.items():
                events.append(
                    {
                        "mode": "updates",
                        "node": node,
                        "fields": list(patch.keys()),
                        "detail": _jsonable(patch),
                    }
                )
        else:
            latest = chunk
            events.append(
                {
                    "mode": "values",
                    "node": "",
                    "fields": list(chunk.keys()),
                    "detail": {
                        "trace": _jsonable(chunk.get("trace", [])),
                        "docs": len(chunk.get("docs", []) or []),
                        "answer_len": len(chunk.get("answer", "") or ""),
                    },
                }
            )

    return {
        "graph": _graph_structure(
            app,
            labels={
                ("__start__", "retrieve"): "固定边",
                ("retrieve", "generate"): "固定边",
                ("generate", "__end__"): "固定边",
            },
        ),
        "events": events,
        "trace": _jsonable(latest.get("trace", [])),
        "docs": _jsonable(latest.get("docs", [])),
        "answer": latest.get("answer", ""),
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "note": "线性图只是流水线：answer 用覆盖语义（只有一个节点写），trace 用 operator.add（两个节点都写）。"
        "想验证覆盖的后果，把 trace 的 Annotated 去掉再跑一次。",
        "while_mapping": WHILE_VS_GRAPH,
    }


# ============================================================
# 3. 条件边：把 if/elif 变成一张路由表
# ============================================================


def route_intent(intent: str, confidence: float, threshold: float) -> str:
    """纯函数路由：只读 state 算下一步去哪，不碰业务（可单测、可插桩）。

    返回值必须是 ``add_conditional_edges`` 映射表里的 key。
    练习 1：置信度低于阈值 → 转人工，而不是硬答一个可能错的分支。
    """
    if confidence < threshold:
        return "human"
    if intent == "order":
        return "order"
    if intent == "policy":
        return "policy"
    return "chat"


def _intent_messages(question: str, method: str) -> list[Any]:
    """按下发方式组装分类消息。

    json_mode 有硬性要求：prompt 里必须出现 "json" 字样，否则供应商直接 400
    （``Prompt must contain the word 'json'``）。这条要求藏在供应商文档里，
    封装层不会替你加——Day 09 踩过同一个坑，这里原样复用结论。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages: list[Any] = [
        SystemMessage(content="判断用户意图，只输出结构化结果。拿不准就把 confidence 调低。")
    ]
    if method == "json_mode":
        messages.append(
            SystemMessage(
                content=(
                    "以 JSON 对象输出，字段：intent（order/policy/chat）、"
                    "confidence（0~1 的浮点数）、order_id（没有就给空串）。只输出 JSON。"
                )
            )
        )
    messages.append(HumanMessage(content=question))
    return messages


def _keyword_intent(question: str) -> dict[str, Any]:
    """兜底分类器：结构化输出整体不可用时，至少让图能跑完（关键词 + 低置信度）。

    真实项目不要这么干——这里只是保证"模型不支持结构化输出"时实验台仍然可用，
    并且用低 confidence 主动触发转人工分支，把局限性暴露出来。
    """
    if "SO" in question.upper():
        tail = question.upper().split("SO", 1)[-1].split(" ")[0].strip()
        return {"intent": "order", "confidence": 0.55, "order_id": f"SO{tail}", "method": "keyword"}
    if any(word in question for word in ("退", "保修", "政策", "运费")):
        return {"intent": "policy", "confidence": 0.75, "order_id": "", "method": "keyword"}
    return {"intent": "chat", "confidence": 0.4, "order_id": "", "method": "keyword"}


async def run_conditional(db: AsyncSession, payload: Day10ConditionalRequest) -> dict[str, Any]:
    """classify 写标签 → route 纯函数选路 → 四条分支各自收口。

    决策与执行分离（Day 04「工具三层契约」在控制流上的复用）：
    ``classify`` 只管打标签，``route`` 只负责选路，具体活儿交给下游节点——
    这样每个分支都能单测、能插桩、能在图上看见。
    """
    _require_langgraph()
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    from app.modules.study.schemas.study import Day10Intent

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day10-conditional")

    ORDERS = {"SO20261101001": "已发货，顺丰 SF1234567890，预计 11-09 送达"}
    THRESHOLD = payload.threshold

    class State(TypedDict):
        messages: Annotated[list, add_messages]
        intent: str
        confidence: float
        order_id: str
        answer: str
        path: Annotated[list[str], operator.add]  # 审计日志：走了哪些节点

    async def classify(state: State) -> dict[str, Any]:
        """只做一件事：把当前问题打上 (intent, confidence, order_id) 标签。

        下发方式按 default → function_calling → json_mode 依次降级（thinking 模型三种
        都可能踩坑：response_format 不支持 / tool_choice 不支持 / json_mode 要求 prompt 含
        "json"）；全失败才退到关键词兜底，并用低 confidence 主动转人工。
        """
        question = _text(state["messages"][-1])
        parsed: dict[str, Any] | None = None
        method = ""
        for candidate in ("default", "function_calling", "json_mode"):
            try:
                runner = (
                    llm.with_structured_output(Day10Intent)
                    if candidate == "default"
                    else llm.with_structured_output(Day10Intent, method=candidate)
                )
                result = await runner.ainvoke(_intent_messages(question, candidate))
                parsed = result.model_dump() if hasattr(result, "model_dump") else dict(result)
                method = candidate
                break
            except Exception:  # noqa: BLE001  降级失败也是课堂素材，继续试下一种
                continue
        if parsed is None:
            parsed = _keyword_intent(question)
            method = str(parsed.pop("method", "keyword"))
        else:
            parsed.pop("method", None)
        return {
            "intent": str(parsed.get("intent", "chat")),
            "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            "order_id": str(parsed.get("order_id", "") or ""),
            "path": [f"classify({method})"],
            "messages": [],
        }

    def route(state: State) -> str:
        """条件边本体：只做决策。"""
        return route_intent(state["intent"], state["confidence"], THRESHOLD)

    async def order_node(state: State) -> dict[str, Any]:
        order_id = state["order_id"]
        if order_id not in ORDERS:
            out = f"没查到订单 {order_id or '（未提取到订单号）'}，请确认订单号，或回复「转人工」。"
        else:
            out = f"订单 {order_id}：{ORDERS[order_id]}"
        return {"answer": out, "path": ["order"]}

    async def policy_node(state: State) -> dict[str, Any]:
        return {
            "answer": "签收 7 天内可无理由退款，需保留原包装。[来源: 售后政策 2026]",
            "path": ["policy"],
        }

    async def chat_node(state: State) -> dict[str, Any]:
        response = await llm.ainvoke(
            [SystemMessage(content="你是客服，简短友好，不承诺任何政策。")] + state["messages"]
        )
        return {"answer": _text(response), "path": ["chat"]}

    async def human_node(state: State) -> dict[str, Any]:
        """练习 1：低置信度 → 转人工。宁可不答，也不要答错一个政策。"""
        return {
            "answer": (
                f"这个问题我拿不准（意图={state['intent']}，置信度={state['confidence']:.2f} "
                f"< {THRESHOLD:.2f}），已为您转接人工客服，请稍候。"
            ),
            "path": ["handoff"],
        }

    builder = StateGraph(State)
    builder.add_node("classify", classify)
    builder.add_node("order", order_node)
    builder.add_node("policy", policy_node)
    builder.add_node("chat", chat_node)
    builder.add_node("human", human_node)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route,
        {"order": "order", "policy": "policy", "chat": "chat", "human": "human"},
    )
    for branch in ("order", "policy", "chat", "human"):
        builder.add_edge(branch, END)  # 分支各自收口，不汇流（汇流会被多条边重复触发）
    app = builder.compile()

    runs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for question in payload.questions:
        begin = time.perf_counter()
        result = await app.ainvoke({"messages": [HumanMessage(content=question)]})
        runs.append(
            {
                "question": question,
                "intent": result.get("intent", ""),
                "confidence": round(float(result.get("confidence", 0.0) or 0.0), 3),
                "order_id": result.get("order_id", ""),
                "path": _jsonable(result.get("path", [])),
                "answer": result.get("answer", ""),
                "elapsed_ms": int((time.perf_counter() - begin) * 1000),
            }
        )

    return {
        "graph": _graph_structure(
            app,
            labels={
                ("__start__", "classify"): "固定边",
                ("classify", "order"): "intent=order",
                ("classify", "policy"): "intent=policy",
                ("classify", "chat"): "intent=chat",
                ("classify", "human"): f"confidence < {THRESHOLD:.2f}",
            },
        ),
        "runs": runs,
        "threshold": THRESHOLD,
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "note": "条件边也可以不写映射表（那时返回值必须恰好等于节点名），但显式映射表更好："
        "改节点名只改一处，路由语义与节点名解耦。四个分支各写一条到 END 的边，"
        "而不是四条边都指向同一个汇总节点——后者会让汇总节点被多个上游重复触发。"
        "置信度阈值是练习 1 的开关：调高它就能看到转人工分支。",
    }


# ============================================================
# 4. 完整 ReAct 环：agent ⇄ tools + 双刹车 + 降级交付
# ============================================================


def _build_react_tools(broken: bool = False) -> list[Any]:
    """Day 10 的工具集：契约沿用 Day 03/04，数据换成课程里的订单号格式。

    ``broken=True`` 让工具永远返回错误——练习 2 用它验证"6 步后是否优雅收尾"。
    """
    from langchain_core.tools import tool
    from pydantic import Field

    orders = {
        "SO20261101001": "已发货，顺丰 SF1234567890，预计 11-09 送达",
        "SO20261101002": "待发货，仓库已拣货",
    }

    @tool(parse_docstring=True)
    def get_order_status(
        order_id: Annotated[str, Field(description="订单号，形如 SO20261101001")],
    ) -> str:
        """查询订单的物流与发货状态。

        何时用：用户问"发货了吗""到哪了""物流状态"这类实时事实时。
        何时不用：用户问退款政策、保修期限等静态规则时，改用 query_refund_policy。
        返回：订单号 + 当前状态；订单不存在时明确说"不存在"，不要编造。

        Args:
            order_id: 订单号，形如 SO20261101001。
        """
        if broken:
            return f"错误: 订单服务暂时不可用（无法查询 {order_id}），请稍后重试"
        return orders.get(order_id, f"订单 {order_id} 不存在")

    @tool(parse_docstring=True)
    def query_refund_policy(
        product: Annotated[str, Field(description="产品名称，如 XR-500 Pro")],
    ) -> str:
        """查询某产品的退款政策（静态规则，不需要订单号）。

        何时用：用户问"能退吗""退款要运费吗""几天内可退"这类规则问题时。
        何时不用：用户问某个订单的具体状态时，改用 get_order_status。
        返回：退款期限 + 条件 + 来源。

        Args:
            product: 产品名称，如 XR-500 Pro。
        """
        if broken:
            return f"错误: 政策库暂时不可用（无法查询 {product}）"
        return f"{product}：签收 7 天内无理由退款，需保留原包装。[来源: 售后政策 2026]"

    return [get_order_status, query_refund_policy]


def _sanitize_for_degrade(messages: list[Any]) -> list[Any]:
    """降级前清理历史：丢掉未配对的 tool_calls 与孤儿 ToolMessage。

    为什么必须清：API 要求 ``assistant.tool_calls`` 后面必须紧跟**同数量**的 tool 消息，
    配对断裂直接 400。降级路径正是"工具停不下来"的时刻，历史里几乎一定有悬空的
    tool_calls——不清就从一个错误掉进另一个错误。
    工具结果改写成一段摘要，信息不丢。
    """
    from langchain_core.messages import HumanMessage, ToolMessage

    cleaned: list[Any] = []
    digest: list[str] = []
    for message in messages:
        if isinstance(message, ToolMessage):
            digest.append(_text(message)[:200])
            continue
        if getattr(message, "tool_calls", None):
            continue
        cleaned.append(message)
    if digest:
        cleaned.append(
            HumanMessage(
                content="（已执行过的工具结果摘要）\n" + "\n".join(f"- {item}" for item in digest)
            )
        )
    return cleaned


def _inner_step_router(step_limit: int) -> Any:
    """子图内的条件边：只在 tools / END 之间选（降级节点属于外层，子图里没有）。"""

    def route(state: Any) -> str:
        from langgraph.graph import END

        last = state["messages"][-1]
        if getattr(last, "tool_calls", None) and state["steps"] < step_limit:
            return "tools"
        return END

    return route


def _build_react_graph(llm: Any, tools: list[Any], step_limit: int, as_subgraph: bool) -> Any:
    """建 ReAct 图：agent ⇄ tools 的环 + 语义刹车 + 降级节点。

    ``as_subgraph=True``（练习 3）把 agent+tools 打包成子图，外层再加降级节点：
    子图内部没有语义出口，所以它在"步数到顶但仍有 tool_calls"时只能返回 END，
    由外层条件边判断"悬空的 tool_calls = 没收敛"再进降级节点。
    """
    from langchain_core.messages import HumanMessage, ToolMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    tool_map = {item.name: item for item in tools}
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]  # 必须 add_messages，否则历史被覆盖
        steps: int  # 覆盖语义，节点里自己 +1
        degraded: bool

    async def agent_node(state: AgentState) -> dict[str, Any]:
        """一次"想"：带工具的模型调用，返回值作为补丁追加进 messages。"""
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response], "steps": state["steps"] + 1}

    async def tools_node(state: AgentState) -> dict[str, Any]:
        """一次"做"：执行最后一条 AIMessage 里的**所有** tool_calls。

        ⚠️ Day 03 的配对铁律：有几个 tool_calls 就回几条 ToolMessage；
        异常也要包成 ToolMessage，不能 raise 跳过（跳过就是配对断裂）。
        """
        last = state["messages"][-1]
        outputs: list[Any] = []
        for call in list(getattr(last, "tool_calls", []) or []):
            fn = tool_map.get(call.get("name", ""))
            if fn is None:
                content = f"未知工具 {call.get('name')}，可用工具：{list(tool_map)}"
            else:
                try:
                    content = str(await fn.ainvoke(call))
                except Exception as exc:  # noqa: BLE001  参数校验失败也在这一层兜住
                    content = f"工具执行失败：{type(exc).__name__}: {exc}。请检查参数后重试一次。"
            outputs.append(ToolMessage(content=content, tool_call_id=call.get("id", "")))
        return {"messages": outputs}

    async def degrade_node(state: AgentState) -> dict[str, Any]:
        """语义刹车后的降级交付（Day 05）：剥掉工具，逼模型用现有信息作答。"""
        history = _sanitize_for_degrade(state["messages"])
        response = await llm.ainvoke([*history, HumanMessage(content=DEGRADE_NOTE)])
        return {"messages": [response], "degraded": True}

    def after_agent(state: AgentState) -> str:
        """break（END）/ continue（tools）/ 刹车（degrade）三条出口。"""
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        if state["steps"] >= step_limit:
            # recursion_limit 是保险丝，这里才是刹车：到顶就转降级，而不是烧到上限报错
            return "degrade"
        return "tools"

    def after_subgraph(state: AgentState) -> str:
        """子图出口只有一个：还有悬空的 tool_calls 就说明没收敛 → 降级。"""
        last = state["messages"][-1]
        return "degrade" if getattr(last, "tool_calls", None) else END

    if not as_subgraph:
        builder = StateGraph(AgentState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", tools_node)
        builder.add_node("degrade", degrade_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", after_agent, {"tools": "tools", "degrade": "degrade", END: END}
        )
        builder.add_edge("tools", "agent")  # ← 回头边 = 循环
        builder.add_edge("degrade", END)
        return builder.compile()

    inner_builder = StateGraph(AgentState)
    inner_builder.add_node("agent", agent_node)
    inner_builder.add_node("tools", tools_node)
    inner_builder.add_edge(START, "agent")
    inner_builder.add_conditional_edges(
        "agent", _inner_step_router(step_limit), {"tools": "tools", END: END}
    )
    inner_builder.add_edge("tools", "agent")
    inner = inner_builder.compile()

    outer = StateGraph(AgentState)
    outer.add_node("react_subgraph", inner)  # 一个编译好的图就是一个节点
    outer.add_node("degrade", degrade_node)
    outer.add_edge(START, "react_subgraph")
    outer.add_conditional_edges("react_subgraph", after_subgraph, {"degrade": "degrade", END: END})
    outer.add_edge("degrade", END)
    return outer.compile()


async def _degrade_outside(llm: Any, question: str, reason: str) -> str:
    """图外降级：``recursion_limit`` 抛错时 state 已经丢了（没有 checkpointer，Day 11 才讲）。

    所以只能拿原问题重新问一遍——这也正好说明"为什么明天要上存档"。
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    response = await llm.ainvoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
            HumanMessage(content=f"（系统提示：{reason}；{DEGRADE_ADVICE}）"),
        ]
    )
    return _text(response)


async def run_react(db: AsyncSession, payload: Day10ReactRequest) -> dict[str, Any]:
    """把 Day 01 的 while 循环变成 agent ⇄ tools 的环，并展示三条出口。

    三条出口分别是：语义收敛（无 tool_calls）、语义刹车（steps 到顶 → degrade 节点）、
    物理保险丝（recursion_limit → GraphRecursionError → 图外降级）。
    """
    _require_langgraph()

    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.errors import GraphRecursionError

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day10-react")
    tools = _build_react_tools(broken=payload.tool_mode == "broken")
    app = _build_react_graph(llm, tools, payload.step_limit, payload.as_subgraph)

    initial = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=payload.question)],
        "steps": 0,
        "degraded": False,
    }

    events: list[dict[str, Any]] = []
    collected: list[Any] = []
    outer_steps = 0
    stop_reason = "completed"
    final_answer = ""
    degraded = False
    started = time.perf_counter()

    try:
        stream_kwargs: dict[str, Any] = {"subgraphs": True} if payload.as_subgraph else {}
        async for chunk in app.astream(
            initial,
            {"recursion_limit": payload.recursion_limit},
            stream_mode="updates",
            **stream_kwargs,
        ):
            if payload.as_subgraph:
                namespace, updates = chunk
            else:
                namespace, updates = (), chunk
            scope = "/".join(name.split(":")[0] for name in namespace)
            if not scope:
                outer_steps += 1
            for node, patch in updates.items():
                messages = list(patch.get("messages", []) or [])
                collected.extend(messages)
                if node == "degrade":
                    degraded = True
                events.append(
                    {
                        "scope": scope or "外层",
                        "node": node,
                        "steps": patch.get("steps"),
                        "messages": _jsonable(messages),
                        "kind": _event_kind(messages),
                    }
                )
    except GraphRecursionError:
        stop_reason = "recursion_limit"
        degraded = True
        final_answer = await _degrade_outside(
            llm, payload.question, f"已达 recursion_limit={payload.recursion_limit}，工具调用未收敛"
        )

    if stop_reason != "recursion_limit":
        # 最终回答 = 最后一条不带 tool_calls 的 AI 消息；降级节点的回答也算
        for message in reversed(collected):
            if getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", None):
                final_answer = _text(message)
                break
        if degraded:
            stop_reason = "degraded"
        elif not final_answer:
            stop_reason = "unfinished"

    return {
        "graph": _graph_structure(
            app,
            labels=(
                {
                    ("__start__", "react_subgraph"): "入口",
                    ("react_subgraph", "degrade"): "子图退出时仍有 tool_calls",
                    ("react_subgraph", "__end__"): "子图语义收敛",
                    ("degrade", "__end__"): "降级交付",
                }
                if payload.as_subgraph
                else {
                    ("__start__", "agent"): "入口",
                    ("agent", "tools"): "有 tool_calls（continue）",
                    ("agent", "degrade"): f"steps ≥ {payload.step_limit}（刹车）",
                    ("agent", "__end__"): "无 tool_calls（break）",
                    ("tools", "agent"): "★ 回头边 = 循环",
                    ("degrade", "__end__"): "降级交付",
                }
            ),
        ),
        "events": events,
        "final_answer": final_answer,
        "stop_reason": stop_reason,
        "degraded": degraded,
        "recursion_limit": payload.recursion_limit,
        "step_limit": payload.step_limit,
        "outer_steps": outer_steps,
        "as_subgraph": payload.as_subgraph,
        "tool_mode": payload.tool_mode,
        "messages_count": len(initial["messages"]) + len(collected),
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "note": (
            "recursion_limit 数的是**超步**：一次 agent→tools→agent 吃掉 3 个，默认 25。"
            "它只是保险丝，语义刹车（steps ≥ N → degrade）才是日常出口；"
            "捕获 GraphRecursionError 后必须降级交付，而不是把异常抛给用户。"
        ),
        "while_mapping": WHILE_VS_GRAPH,
    }


def _event_kind(messages: list[Any]) -> str:
    """给一条节点更新打标签：决定调工具 / 工具返回 / 生成回答 / 空更新。"""
    if not messages:
        return "state"
    first = messages[0]
    if getattr(first, "tool_calls", None):
        return "tool_call"
    if getattr(first, "type", "") == "tool":
        return "tool_result"
    return "answer"


# ============================================================
# 5. 并行扇出 / 汇聚：静态 + Send 动态 + 子图
# ============================================================


def _split_subs(query: str) -> list[str]:
    """把一句话按分号拆成子查询；没有分号就整句当成一个子查询。"""
    text = query.replace(";", "；")
    subs = [item.strip() for item in text.split("；") if item.strip()]
    return subs or [query]


async def run_parallel(payload: Day10ParallelRequest) -> dict[str, Any]:
    """三个并行实验，全部本地假检索（不花 token，纯看调度）。

    ① 静态扇出：三条 START 出发的边 + 三条汇到 merge 的边，汇聚节点自动等全部上游；
    ② ``Send`` 动态扇出：分支数运行时才定（按；拆子查询）；
    ③ 子图：一个编译好的图直接当节点用，``subgraphs=True`` 能看到它内部事件。
    """
    _require_langgraph()

    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    async def fake_search(source: str, query: str, cost: float) -> str:
        """模拟一次 IO：异步 sleep 才能真正并发（同步 time.sleep 会卡住事件循环）。"""
        await asyncio.sleep(cost)
        return f"[{source}] 关于「{query}」的命中结果"

    def merge_docs(state: Any) -> dict[str, Any]:
        """汇聚节点：docs 已经由 reducer 合好，这里只做拼接。

        注意 docs 的顺序**不保证**（并行完成顺序不同）；要稳定顺序就在这里 sorted。
        """
        return {"merged": " / ".join(state["docs"])}

    # ---------- ① 静态扇出 ----------
    class FanState(TypedDict):
        query: str
        docs: Annotated[list[str], operator.add]  # 三个节点并发写 → 必须 reducer
        merged: str

    costs = {"kb": 0.3, "ticket": 0.5, "web": 0.4}

    def make_searcher(name: str, label: str) -> Any:
        async def node(state: FanState) -> dict[str, Any]:
            return {"docs": [await fake_search(label, state["query"], costs[name])]}

        return node

    merge = merge_docs

    fan = StateGraph(FanState)
    fan.add_node("merge", merge)  # ⚠️ 先加节点再加边：边引用未知节点会直接报错
    for name, label in (("kb", "KB"), ("ticket", "工单"), ("web", "Web")):
        fan.add_node(name, make_searcher(name, label))
        fan.add_edge(START, name)  # 三条从 START 出发的边 = 扇出
        fan.add_edge(name, "merge")  # 三条汇到同一个点 = 汇聚（自动等待）
    fan.add_edge("merge", END)
    fan_app = fan.compile()

    begin = time.perf_counter()
    fan_out = await fan_app.ainvoke({"query": payload.query, "docs": [], "merged": ""})
    fan_elapsed = time.perf_counter() - begin

    # ---------- ② Send 动态扇出 ----------
    class DynState(TypedDict):
        query: str
        docs: Annotated[list[str], operator.add]
        merged: str

    subs = _split_subs(payload.query)

    def plan(state: DynState) -> list[Any]:
        """把一个问题拆成 N 个子查询，N 运行时才知道——这就是 Send 的用途。"""
        branches = _split_subs(state["query"])
        return [Send("search_one", {"sub": item}) for item in branches]

    async def search_one(payload_item: dict[str, Any]) -> dict[str, Any]:
        """Send 目标节点收到的是 payload，不是全局 state。"""
        await asyncio.sleep(0.2)
        return {"docs": [f"[sub] {payload_item['sub']} 的结果"]}

    dyn = StateGraph(DynState)
    dyn.add_node("search_one", search_one)
    dyn.add_node("merge", merge)
    dyn.add_conditional_edges(START, plan, ["search_one"])
    dyn.add_edge("search_one", "merge")
    dyn.add_edge("merge", END)
    dyn_app = dyn.compile()

    begin = time.perf_counter()
    dyn_out = await dyn_app.ainvoke({"query": payload.query, "docs": [], "merged": ""})
    dyn_elapsed = time.perf_counter() - begin

    # ---------- ③ 子图当节点 ----------
    class SubState(TypedDict):
        query: str
        # ⚠️ 父图的 log 用**覆盖**语义：子图退出时提交的是整份子状态，
        # 父图若也写 operator.add，triage 那条会被再追加一次（已实测，见 reducer_trap）。
        log: list[str]
        summary: str

    class InnerState(TypedDict):
        query: str
        log: Annotated[list[str], operator.add]  # 子图内部用追加
        summary: str

    async def inner_one(state: InnerState) -> dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"log": ["inner-1 研究"]}

    async def inner_two(state: InnerState) -> dict[str, Any]:
        await asyncio.sleep(0.1)
        return {"log": ["inner-2 汇总"]}

    inner_builder = StateGraph(InnerState)
    inner_builder.add_node("inner_one", inner_one)
    inner_builder.add_node("inner_two", inner_two)
    inner_builder.add_edge(START, "inner_one")
    inner_builder.add_edge("inner_one", "inner_two")
    inner_builder.add_edge("inner_two", END)
    inner = inner_builder.compile()

    async def triage(state: SubState) -> dict[str, Any]:
        return {"log": ["triage 分流"]}

    async def finish(state: SubState) -> dict[str, Any]:
        return {"summary": " → ".join(state["log"])}

    outer_builder = StateGraph(SubState)
    outer_builder.add_node("triage", triage)
    outer_builder.add_node("research", inner)  # ★ 一个编译好的图就是一个节点
    outer_builder.add_node("finish", finish)
    outer_builder.add_edge(START, "triage")
    outer_builder.add_edge("triage", "research")
    outer_builder.add_edge("research", "finish")
    outer_builder.add_edge("finish", END)
    outer = outer_builder.compile()

    sub_events: list[dict[str, Any]] = []
    sub_state: dict[str, Any] = {}
    begin = time.perf_counter()
    async for chunk in outer.astream(
        {"query": payload.query, "log": [], "summary": ""},
        stream_mode="updates",
        subgraphs=True,
    ):
        namespace, updates = chunk
        scope = "/".join(name.split(":")[0] for name in namespace) or "外层"
        for node, patch in updates.items():
            sub_events.append({"scope": scope, "node": node, "patch": _jsonable(patch)})
            sub_state.update(patch)
    sub_elapsed = time.perf_counter() - begin

    return {
        "static": {
            "costs": costs,
            "sum_s": round(sum(costs.values()), 2),
            "elapsed_s": round(fan_elapsed, 2),
            "docs": _jsonable(fan_out.get("docs", [])),
            "merged": fan_out.get("merged", ""),
            "graph": _graph_structure(
                fan_app,
                labels={
                    ("__start__", "kb"): "扇出",
                    ("__start__", "ticket"): "扇出",
                    ("__start__", "web"): "扇出",
                    ("kb", "merge"): "汇聚（自动等待）",
                    ("ticket", "merge"): "汇聚（自动等待）",
                    ("web", "merge"): "汇聚（自动等待）",
                },
            ),
        },
        "dynamic": {
            "subs": subs,
            "branch_count": len(subs),
            "elapsed_s": round(dyn_elapsed, 2),
            "docs": _jsonable(dyn_out.get("docs", [])),
            "merged": dyn_out.get("merged", ""),
            "graph": _graph_structure(
                dyn_app,
                labels={
                    ("__start__", "search_one"): f"Send × {len(subs) or 1}（运行时决定）",
                    ("search_one", "merge"): "汇聚（自动等待）",
                },
            ),
        },
        "subgraph": {
            "elapsed_s": round(sub_elapsed, 2),
            "events": sub_events,
            "summary": sub_state.get("summary", ""),
            "graph": _graph_structure(
                outer,
                labels={
                    ("__start__", "triage"): "入口",
                    ("triage", "research"): "子图（内部节点在外层不可见）",
                    ("research", "finish"): "子图跑完才继续",
                },
            ),
            "reducer_trap": "子图当节点时，它退出时提交的是**整份子状态**——所以父图同一个 key 若也写 operator.add，"
            "父图已有的值会被再追加一次（实测：log 里『triage 分流』出现两遍）。"
            "这就是课程 2.6『子图 state 要与父图兼容』的真实含义：**key 要对得上，reducer 要想清楚由谁负责。**",
        },
        "note": "并行耗时取最大而不是求和（0.3 + 0.5 + 0.4 = 1.2s → ≈0.5s）；"
        "docs 的**顺序不保证**（完成顺序不同），要稳定顺序就在 merge 里显式排序。"
        "汇聚节点会自动等全部上游完成——这是超步模型的直接结果，不用写 join。",
        "send_rules": SEND_RULES,
    }


# ============================================================
# 6. 静态概念数据（供前端渲染）
# ============================================================


async def mapping() -> dict[str, Any]:
    """Day 10 的概念表：三件套 / reducer / while 对照 / stream_mode / 六个坑 / 版本。"""
    return {
        "triple_pieces": TRIPLE_PIECES,
        "reducer_table": REDUCER_TABLE,
        "while_vs_graph": WHILE_VS_GRAPH,
        "stream_modes": STREAM_MODES,
        "pitfalls": PITFALLS,
        "send_rules": SEND_RULES,
        "presets": PRESETS,
        "install_hint": INSTALL_HINT,
        "langgraph_version": await asyncio.to_thread(_version),
    }


def _version() -> str:
    """读已安装的 langgraph 版本：版本差异是框架篇最常见的坑。"""
    try:
        from importlib.metadata import version

        return version("langgraph")
    except Exception:  # noqa: BLE001  没装就是没装，前端照常展示
        return "未安装"


__all__ = [
    "PRESETS",
    "REDUCER_TABLE",
    "STREAM_MODES",
    "TRIPLE_PIECES",
    "WHILE_VS_GRAPH",
    "mapping",
    "route_intent",
    "run_conditional",
    "run_min_graph",
    "run_parallel",
    "run_react",
    "state_reducer_demo",
]
