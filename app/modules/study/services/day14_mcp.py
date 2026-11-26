"""Day 14 MCP 协议：把工具从 Agent 的进程里解放出来。

课程 day14 的五个脚本在这里收成一个模块：

    最小 Server / 多工具 Server   → ``discover``    （01/02：真实 spawn 子进程 + tools/list 发现）
    客户端接入 LangGraph          → ``run_agent``   （03：get_tools() 把 MCP 工具转成 LangChain Tool）
    远程共享服务                  → ``run_http``    （04：streamable-http，后台线程起 uvicorn 再连）
    MCP + 本地工具混装            → ``run_mixed``   （05：余额不足 → 本地 escalate_to_human，流式照常）
    stdout 污染协议通道           → ``run_noisy``   （坑 1 现场：三种污染形态的实测后果）

贯穿全篇的三条结论：

1. **FC 是能力层，MCP 是协议层**：MCP 只回答"工具怎么被发现、被传输、被调用"，
   模型怎么选工具仍然是 Function Calling 那一套（Day 03）——两者叠加，不互斥；
2. **stdio 的 Server 不是常驻服务**：客户端每开一个会话就 spawn 一个子进程，
   随会话生灭；要团队共享就得离开 stdio 换成 streamable-http；
3. **stdout 是协议通道**：任何纯文本混进去都会破坏 JSON-RPC 帧。本模块实测（mcp 1.24）
   把这条坑分成了两档：带换行的整行垃圾**可能**被客户端跳过（也可能报 Invalid JSON，取决于
   它到达的时刻），而不带换行的半行文本会和下一帧 JSON 粘在一起，**必然**挂死——
   ``run_noisy`` 就是这三种形态的对照现场。

三个工程决定（与课程示例的差异）：

- Server 脚本落成 ``mcp_servers/`` 下的真实文件，用 ``sys.executable`` 拉起，
  保证子进程与宿主同一个解释器（否则就是 Day 13 那种"宿主管用、子进程缺依赖"）；
- ``discover`` 同一次请求里既给**协议视角**（``session.list_tools()`` 的原始 JSON Schema），
  也给**LangChain 视角**（``get_tools()`` 转换后的 args_schema），把桥的两端摆在一起；
- ``run_http`` 在后台线程起一个临时 uvicorn（随机端口），用完立刻关掉，
  不往仓库里留常驻服务，也不占用用户的 15401。

mcp / langchain-mcp-adapters / langgraph 全部延迟导入：缺依赖时接口返回可读的 400，
而不是让整个应用起不来。
"""

import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import (
    Day14AgentRequest,
    Day14CallRequest,
    Day14HttpRequest,
    Day14MixedRequest,
    Day14NoisyRequest,
)
from app.modules.study.services.day09_langchain import build_chat_model
from app.modules.study.services.day10_langgraph import _jsonable, _text
from app.modules.study.services.llm import resolve_model

INSTALL_HINT = 'pip install "mcp[cli]" langchain-mcp-adapters'

# Server 脚本目录（真实文件，不是字符串拼出来的临时脚本）
SERVERS_DIR = Path(__file__).parent / "mcp_servers"

# 演示用的两个 Server：key → 脚本名（key 同时是客户端配置里的连接名）
SERVER_SCRIPTS: dict[str, str] = {
    "docs": "doc_server.py",
    "crm": "crm_server.py",
    "noisy": "noisy_server.py",
}

# ============================================================
# 静态概念数据（供前端渲染，全部来自课程原文）
# ============================================================

# 1.1 M×N → M+N
MXN: dict[str, Any] = {
    "problem": "M 个 Agent 应用 × N 个工具/数据源 = M×N 套自定义集成",
    "solution": "M 个 Agent 应用 ──统一协议──► MCP Server ──封装一次──► N 个数据源",
    "pain_points": [
        "每接一个系统就要写一遍胶水代码：Agent1 自定义 API 接数据库、SDK 胶水接 CRM",
        "同一个数据库/CRM，换一个 Agent 再写一遍",
        "换框架（LangGraph / 原生 FC / 某厂平台）就要按新格式重写同一套工具",
    ],
    "value": "工具方只实现一次 MCP Server，所有 MCP 客户端（不限 Python / 不限框架）都能用",
}

# 1.2 FC 与 MCP 的分工
FC_VS_MCP: list[dict[str, str]] = [
    {
        "dimension": "回答的问题",
        "fc": "模型『怎么决定调用工具』",
        "mcp": "工具『怎么被发现、被传输、被调用』",
    },
    {"dimension": "所在位置", "fc": "模型 ↔ 你的代码之间", "mcp": "你的代码 ↔ 工具进程/服务之间"},
    {
        "dimension": "关心什么",
        "fc": "intent → arguments JSON",
        "mcp": "工具清单（tools/list）、调用（tools/call）、传输",
    },
    {
        "dimension": "是否管执行",
        "fc": "❌ 执行永远在你本地（Day 03 铁律）",
        "mcp": "✅ 定义了客户端如何触发远端执行",
    },
    {
        "dimension": "在 LangGraph 里",
        "fc": "create_agent 内部走 tool_calls → ToolNode",
        "mcp": "langchain-mcp-adapters 把 Server 工具转成 @tool 列表",
    },
]

# 1.3 架构三角色
ROLES: list[dict[str, str]] = [
    {"role": "Host", "who": "你的 Agent 应用（今天就 LangGraph）", "duty": "决定要连哪些 Server、把工具交给模型"},
    {"role": "Client", "who": "Host 内部、与一个 Server 1:1 的中间层", "duty": "握手、发 tools/list、发 tools/call（adapters 帮你写了）"},
    {"role": "Server", "who": "独立进程或独立 HTTP 服务", "duty": "暴露工具 / 资源 / 提示词，自己访问真实系统"},
]

# 1.3 两种传输（外加被取代的 sse）
TRANSPORTS: list[dict[str, str]] = [
    {
        "transport": "stdio",
        "scene": "本地工具：和 Host 同机，Client 拉起子进程",
        "launch": "Client 负责 spawn（command + args）",
        "lifecycle": "Client 退出即结束，不是常驻服务",
        "config": '"transport": "stdio"',
    },
    {
        "transport": "streamable-http",
        "scene": "远程/共享工具：独立部署的 HTTP 服务",
        "launch": "你自己起服务（uvicorn 等）",
        "lifecycle": "独立运行，多 Host 共享",
        "config": '"transport": "http"',
    },
    {
        "transport": "sse（已弃用）",
        "scene": "早期远程传输，2024-11 规范修订后仅作向后兼容",
        "launch": "同上",
        "lifecycle": "同上",
        "config": '新代码不要写 "sse"',
    },
]

# 1.4 Server 三种原语
PRIMITIVES: list[dict[str, str]] = [
    {
        "primitive": "Tools（model-controlled）",
        "decorator": "@mcp.tool()",
        "job": "可被模型选择调用的函数",
        "maps_to": "Day 03 的 tools 注册表",
    },
    {
        "primitive": "Resources（app-controlled）",
        "decorator": '@mcp.resource("crm://stats")',
        "job": "供宿主直接读取的数据，不经过模型决策",
        "maps_to": "Day 07 的文档 RAG 源（程序决定怎么用）",
    },
    {
        "primitive": "Prompts（user-controlled）",
        "decorator": "@mcp.prompt()",
        "job": "预置提示词模板，由用户选择触发",
        "maps_to": "Day 02 的 system prompt 模板",
    },
]

# 2.1 迁移对照：Day 03 进程内 → Day 14 MCP Server
MIGRATION: list[dict[str, str]] = [
    {"day03": "@tool 装饰器", "day14": "@mcp.tool() 装饰器", "why": "装饰器换名字，逻辑不变"},
    {
        "day03": "docstring 即说明书（Day 04 四段式）",
        "day14": "docstring 仍即说明书，协议自动透传",
        "why": "跨进程后用户看不到源码，说明书质量更关键",
    },
    {
        "day03": "参数 schema 由 Pydantic 生成",
        "day14": "参数 schema 同样自动生成，走 JSON Schema",
        "why": "跨语言也能读懂同一份 inputSchema",
    },
    {
        "day03": "dispatch 注册表手动管理",
        "day14": "Server 自己就是注册表，tools/list 自动暴露",
        "why": "工具清单变成协议能力，不用手写枚举",
    },
    {
        "day03": "执行在 Host 进程",
        "day14": "执行在 Server 进程，结果跨进程传回",
        "why": "进程隔离：Server 崩了 Agent 不崩",
    },
]

# 第 4 节：五个常见坑
PITFALLS: list[dict[str, str]] = [
    {
        "no": "1",
        "title": "stdio Server 里 print 到 stdout",
        "symptom": "客户端报 Invalid JSON，或者干脆挂死等不到响应",
        "fix": "日志一律走 stderr（logging 默认就是 stderr）；尤其小心 C 扩展/子进程直接写 fd 1",
    },
    {
        "no": "2",
        "title": "get_tools() 忘了 await",
        "symptom": "拿到 coroutine 而不是列表，create_agent 报类型错",
        "fix": "整个客户端流程写在 async def 里，tools = await client.get_tools()",
    },
    {
        "no": "3",
        "title": "transport 写成 sse",
        "symptom": "兼容性警告或行为差异",
        "fix": '远程统一 "transport": "http"，URL 以 /mcp 结尾（FastMCP 默认挂载路径）',
    },
    {
        "no": "4",
        "title": "以为 stdio Server 是常驻服务",
        "symptom": "去『部署』一个 stdio Server，结果客户端各自 spawn，状态互不相通",
        "fix": "1 个 Host 且同机 → stdio；多个或跨机 → streamable-http",
    },
    {
        "no": "5",
        "title": "把 MCP 神化成 Agent 互操作协议",
        "symptom": "以为 MCP 能解决 Agent 之间的委托与协商",
        "fix": "MCP 只标准化 Host↔工具（单向供给）；Agent↔Agent 是另一类协议（A2A 等）",
    },
]

# 第 5 节：速查表
CHEATSHEET: list[dict[str, str]] = [
    {"topic": "MCP 定位", "content": "FC = 模型怎么选工具（能力层）；MCP = 工具怎么被发现/调用（协议层）；两者叠加使用"},
    {"topic": "架构三角色", "content": "Host（你的 Agent）─ Client（1:1 连接）─ MCP Server（独立进程/服务）"},
    {"topic": "stdio", "content": "command + args，Client spawn 子进程，本机单人，进程随会话生灭"},
    {"topic": "http", "content": "url = http://host:port/mcp，独立部署，团队共享，常驻"},
    {"topic": "三原语", "content": "@mcp.tool() 模型决策调用（90% 场景）/ @mcp.resource() 宿主直读 / @mcp.prompt() 用户选择"},
    {"topic": "客户端三步", "content": "MultiServerMCPClient({...}) → await get_tools() → create_agent(llm, tools=tools)"},
    {"topic": "铁律", "content": "stdio 的 stdout 是协议通道，日志只准写 stderr；Server 端错误变成错误文本回填，Agent 不会崩"},
]

CHECKLIST: list[str] = [
    "能说清 M×N 问题，以及 MCP 怎么把它降成 M+N",
    "能区分 FC 与 MCP：谁管选工具、谁管发现与传输",
    "能画出 Host / Client / Server 三角色，并说明 Client 是 1:1 的",
    "能选传输：1 个 Host 且同机用 stdio，多个或跨机用 http",
    "能写出最小 Server：FastMCP + @mcp.tool() + mcp.run(transport=\"stdio\")",
    "能接入 Host：await client.get_tools() 后与本地 @tool 混装进同一个 tools 列表",
    "能解释为什么 stdio Server 不是常驻服务、状态随子进程生灭",
    "知道 stdout 是协议通道，日志必须走 stderr",
]

PRESETS: dict[str, list[str]] = {
    "agent": [
        "查一下客户 C001 的信息，然后看看文档库里 Checkpointer 是什么",
        "客户 C002 现在余额多少？顺便查一下 MCP 协议是干什么的",
    ],
    "mixed": [
        "给客户 C001 扣款 5000 元",
        "给客户 C002 扣款 50 元，然后把结果告诉我",
    ],
}


# ============================================================
# 0. 公共设施：依赖探测 / 连接配置 / 工具视图
# ============================================================


def _require_mcp() -> None:
    """依赖探测：没装 mcp 时给安装命令，而不是 ImportError 堆栈。"""
    try:
        import mcp.server.fastmcp  # noqa: F401
        import langchain_mcp_adapters.client  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 只在缺依赖的环境触发
        raise ValueError(f"未安装 mcp / langchain-mcp-adapters，请先执行：{INSTALL_HINT}") from exc


def _version(name: str) -> str:
    """读已安装包版本：MCP 的坑一大半和 SDK 版本有关（坑 3 就是 sse → http）。"""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001  没装就是没装，前端照常展示
        return "未安装"


def _elapsed_ms(started: float) -> int:
    """从起点到现在的毫秒数（前端用它标"握手花了多久、卡在第几秒"）。"""
    return int((time.perf_counter() - started) * 1000)


def _server_path(key: str) -> str:
    """Server 脚本的绝对路径（``sys.executable`` 拉起时必须是绝对路径）。"""
    script = SERVER_SCRIPTS.get(key)
    if not script:
        raise ValueError(f"未知的 MCP Server：{key}（可选：{'、'.join(SERVER_SCRIPTS)}）")
    return str(SERVERS_DIR / script)


def _stdio_connection(key: str, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    """stdio 连接配置：command 用当前解释器，args 是真实脚本文件。

    用 ``sys.executable`` 而不是 ``"python"``：后者可能解析到系统 Python，
    子进程就会缺依赖（Day 13 踩过一模一样的坑）。
    """
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return {
        "command": sys.executable,
        "args": [_server_path(key)],
        "transport": "stdio",
        "env": env,
    }


def _schema_of(tool: Any) -> dict[str, Any]:
    """取工具的 JSON Schema：LangChain Tool 的 args_schema 可能是 pydantic 模型或 dict。"""
    schema = getattr(tool, "args_schema", None)
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_json_schema"):
        try:
            return schema.model_json_schema()
        except Exception:  # noqa: BLE001  个别 schema 生成失败不该拖垮实验
            return {}
    return {}


def _params_view(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """把 JSON Schema 的 properties 摊平成表格行：名字 / 类型 / 是否必填 / 说明。"""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    rows: list[dict[str, Any]] = []
    for name, spec in properties.items():
        spec = spec if isinstance(spec, dict) else {}
        rows.append(
            {
                "name": name,
                "type": spec.get("type") or "any",
                "required": name in required,
                "description": (spec.get("description") or "").strip(),
            }
        )
    return rows


def _tool_view(tool: Any, server: str = "") -> dict[str, Any]:
    """把一个 LangChain Tool 压成前端可渲染的一行（这就是"桥"的产物）。"""
    schema = _schema_of(tool)
    description = (getattr(tool, "description", "") or "").strip()
    return {
        "name": getattr(tool, "name", ""),
        "server": server,
        "description": description,
        "summary": description.splitlines()[0] if description else "",
        "params": _params_view(schema),
        "input_schema": _jsonable(schema),
    }


def _content_texts(result: Any) -> list[str]:
    """把 MCP 的 CallToolResult 摊成纯文本列表（text / image / resource 都见过）。"""
    texts: list[str] = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        texts.append(str(text) if text is not None else f"[{getattr(item, 'type', 'unknown')} 内容]")
    return texts


def _free_port() -> int:
    """要一个空闲端口（bind 0 让内核分配，立刻释放后用；有小概率被抢占，教学够用）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _trace_steps(messages: list[Any]) -> list[dict[str, Any]]:
    """把消息序列摊成"工具调用 → 工具结果 → 回答"的轨迹。

    MCP 工具在 LangGraph 眼里和本地工具没有区别：都是 tool_calls + ToolMessage，
    所以这条轨迹对两种来源完全通用——这正是解耦的价值。
    """
    steps: list[dict[str, Any]] = []
    for message in messages or []:
        kind = getattr(message, "type", "")
        if kind in {"ai", "AIMessage"}:
            for call in getattr(message, "tool_calls", None) or []:
                steps.append(
                    {
                        "kind": "tool_call",
                        "name": call.get("name", ""),
                        "args": _jsonable(call.get("args") or {}),
                        "channel": "mcp" if call.get("name") in {"search_docs", "get_customer", "deduct_balance"} else "local",
                    }
                )
            answer = _text(message).strip()
            if answer:
                steps.append({"kind": "answer", "name": "model", "content": answer})
        elif kind == "tool":
            steps.append(
                {
                    "kind": "tool_result",
                    "name": getattr(message, "name", "") or "",
                    "content": _text(message).strip(),
                }
            )
    return steps


def _group_bubbles(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按节点把 token 聚成气泡：节点一变就开新泡（与 Day 13 同一套数据形态）。"""
    bubbles: list[dict[str, Any]] = []
    for row in tokens:
        content = row.get("content") or ""
        if not content.strip():
            continue
        if not bubbles or bubbles[-1]["node"] != row["node"]:
            bubbles.append({"node": row["node"], "text": "", "tokens": 0})
        bubbles[-1]["text"] += content
        bubbles[-1]["tokens"] += 1
    return bubbles


# ============================================================
# 1. 概念表
# ============================================================


async def mapping() -> dict[str, Any]:
    """Day 14 概念表：M×N / FC vs MCP / 三角色 / 两传输 / 三原语 / 迁移 / 五个坑 / 速查表。"""
    return {
        "mxn": MXN,
        "fc_vs_mcp": FC_VS_MCP,
        "roles": ROLES,
        "transports": TRANSPORTS,
        "primitives": PRIMITIVES,
        "migration": MIGRATION,
        "pitfalls": PITFALLS,
        "cheatsheet": CHEATSHEET,
        "checklist": CHECKLIST,
        "presets": PRESETS,
        "servers": [
            {"key": key, "script": script, "transport": "stdio"} for key, script in SERVER_SCRIPTS.items()
        ],
        "install_hint": INSTALL_HINT,
        "versions": {
            "mcp": _version("mcp"),
            "langchain-mcp-adapters": _version("langchain-mcp-adapters"),
            "langgraph": _version("langgraph"),
        },
    }


# ============================================================
# 2. 课程 01/02：工具发现（真实 spawn + tools/list）
# ============================================================


async def discover() -> dict[str, Any]:
    """spawn 两个 stdio Server，把「协议视角」和「LangChain 视角」摆在一起。

    每个 Server 会开两次会话：一次用原始 ``ClientSession`` 调 ``list_tools`` /
    ``list_resources`` / ``list_prompts``（协议视角），一次用 ``get_tools()``
    走 adapter 的转换（LangChain 视角）。stdio 下**每开一次会话就是一个子进程**，
    所以本接口一共 spawn 4 个进程——这个数字本身就是课程 3.2 那条轨迹的证据。
    """
    _require_mcp()
    from langchain_mcp_adapters.client import MultiServerMCPClient

    started = time.perf_counter()
    keys = ["docs", "crm"]
    client = MultiServerMCPClient({key: _stdio_connection(key) for key in keys})

    servers: list[dict[str, Any]] = []
    spawn_count = 0
    total_tools = 0

    for key in keys:
        session_started = time.perf_counter()
        tools_raw: list[dict[str, Any]] = []
        resources: list[dict[str, Any]] = []
        prompts: list[dict[str, Any]] = []

        async with client.session(key) as session:
            spawn_count += 1
            listed = await session.list_tools()
            for tool in listed.tools:
                schema = getattr(tool, "inputSchema", None) or {}
                tools_raw.append(
                    {
                        "name": tool.name,
                        "description": (tool.description or "").strip(),
                        "params": _params_view(schema),
                        "input_schema": _jsonable(schema),
                    }
                )
            try:
                listed_resources = await session.list_resources()
                resources = [
                    {
                        "uri": str(item.uri),
                        "name": getattr(item, "name", "") or "",
                        "description": (getattr(item, "description", "") or "").strip(),
                    }
                    for item in listed_resources.resources
                ]
            except Exception as exc:  # noqa: BLE001  Server 没实现资源属正常情况
                resources = [{"uri": "-", "name": "（本 Server 没有资源）", "description": str(exc)[:80]}]
            try:
                listed_prompts = await session.list_prompts()
                prompts = [
                    {
                        "name": item.name,
                        "description": (item.description or "").strip(),
                        "arguments": [arg.name for arg in (item.arguments or [])],
                    }
                    for item in listed_prompts.prompts
                ]
            except Exception:  # noqa: BLE001  同上
                prompts = []
        session_ms = _elapsed_ms(session_started)

        adapted_started = time.perf_counter()
        adapted = await client.get_tools(server_name=key)
        spawn_count += 1
        adapted_view = [_tool_view(tool, key) for tool in adapted]
        adapted_ms = _elapsed_ms(adapted_started)
        total_tools += len(adapted)

        servers.append(
            {
                "key": key,
                "script": SERVER_SCRIPTS[key],
                "tools": tools_raw,
                "resources": resources,
                "prompts": prompts,
                "adapted": adapted_view,
                "session_ms": session_ms,
                "adapted_ms": adapted_ms,
                "note": (
                    "原始 list_tools 给的是 inputSchema（JSON Schema）；"
                    "get_tools() 把它转成 LangChain Tool 的 args_schema，模型看到的就是后者的结构化说明。"
                ),
            }
        )

    return {
        "servers": servers,
        "spawn_count": spawn_count,
        "total_tools": total_tools,
        "elapsed_ms": _elapsed_ms(started),
        "handshake": [
            {"step": "spawn 子进程", "detail": "每个 stdio 连接 = 一个 python 子进程（command + args）"},
            {"step": "initialize / initialized", "detail": "协议握手，交换 protocolVersion 与 capabilities"},
            {"step": "tools/list", "detail": "Server 把自己注册的工具连同 inputSchema 一起交出来"},
            {"step": "resources/list + prompts/list", "detail": "资源与提示词是另外两个原语，同样走 list 发现"},
            {"step": "get_tools()", "detail": "adapter 把 MCP Tool 转成 langchain_core.tools.Tool（name/description/args_schema）"},
        ],
        "bridge_note": (
            "MCP 世界和 Function Calling 世界的桥就在 get_tools()：转换后模型选工具的方式"
            "与 Day 03 完全一样，MCP 只替换了『工具从哪来』。"
        ),
        "note": (
            f"本次请求 spawn 了 {spawn_count} 个子进程（2 个 Server × 协议会话 + adapter 会话各一次）。"
            "stdio 不是常驻服务：请求结束、会话关闭，子进程随之退出。"
        ),
    }


# ============================================================
# 3. 直接调用工具（tools/call，不经模型）
# ============================================================


async def call_tool(payload: Day14CallRequest) -> dict[str, Any]:
    """直接发 ``tools/call``：不经过模型，方便验证 Server 端的校验逻辑。

    重点：Server 端报错/校验失败**不会抛异常**，而是以 ``isError`` + 文本内容的
    形式回填——这就是 Day 05「错误信息即 prompt」在跨进程场景下的延续。
    """
    _require_mcp()
    from langchain_mcp_adapters.client import MultiServerMCPClient

    if payload.server not in {"docs", "crm"}:
        raise ValueError(f"只能连 docs / crm 两个演示 Server，收到：{payload.server}")

    started = time.perf_counter()
    client = MultiServerMCPClient({payload.server: _stdio_connection(payload.server)})
    async with client.session(payload.server) as session:
        result = await session.call_tool(payload.tool, payload.arguments)

    return {
        "server": payload.server,
        "tool": payload.tool,
        "arguments": payload.arguments,
        "is_error": bool(getattr(result, "isError", False)),
        "content": _content_texts(result),
        "structured": _jsonable(getattr(result, "structuredContent", None)),
        "elapsed_ms": _elapsed_ms(started),
        "note": (
            "调用发生在 Server 子进程里：参数校验、余额检查、审计日志都由 Server 负责，"
            "Host 只拿到一行文本结果。错误也是结果，不是异常。"
        ),
    }


# ============================================================
# 4. 课程 03：客户端接入 LangGraph（单 Agent 端到端）
# ============================================================


async def run_agent(db: AsyncSession, payload: Day14AgentRequest) -> dict[str, Any]:
    """一个 Agent 横跨两个 Server：查 CRM + 查文档，模型层毫无感知差异。"""
    _require_mcp()
    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient

    started = time.perf_counter()
    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg)

    client = MultiServerMCPClient({"docs": _stdio_connection("docs"), "crm": _stdio_connection("crm")})
    discovery_started = time.perf_counter()
    tools: list[Any] = []
    tools_view: list[dict[str, Any]] = []
    for key in ("docs", "crm"):
        adapted = await client.get_tools(server_name=key)
        tools.extend(adapted)
        tools_view.extend(_tool_view(tool, key) for tool in adapted)
    discovery_ms = _elapsed_ms(discovery_started)

    agent = create_agent(llm, tools=tools)
    result = await agent.ainvoke(
        {"messages": [("user", payload.question)]},
        config={"recursion_limit": max(6, payload.max_steps * 2)},
    )
    messages = result.get("messages") or []
    steps = _trace_steps(messages)
    answers = [step["content"] for step in steps if step["kind"] == "answer"]

    return {
        "question": payload.question,
        "tools": tools_view,
        "steps": steps,
        "tool_call_count": sum(1 for step in steps if step["kind"] == "tool_call"),
        "answer": answers[-1] if answers else "",
        "discovery_ms": discovery_ms,
        "elapsed_ms": _elapsed_ms(started),
        "model": cfg.model_name,
        "note": (
            "工具来自两个不同子进程，但 create_agent 拿到的就是一个扁平 tools 列表："
            "ToolNode 按名字路由，对 MCP / 本地一视同仁。"
        ),
    }


# ============================================================
# 5. 课程 05：MCP 工具 + 本地工具混装（含流式）
# ============================================================


async def run_mixed(db: AsyncSession, payload: Day14MixedRequest) -> dict[str, Any]:
    """混装演练：扣款被 Server 端拦住 → 本地 escalate_to_human 收尾，流式照常可用。

    "MCP 工具的执行结果、报错、流式事件全部走既有管线"——因为到了 LangGraph 层面，
    工具来源早被 ``get_tools()`` 抹平了。
    """
    _require_mcp()
    from langchain.agents import create_agent
    from langchain_core.tools import tool as langchain_tool
    from langchain_mcp_adapters.client import MultiServerMCPClient

    started = time.perf_counter()
    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg)

    @langchain_tool
    def escalate_to_human(reason: str) -> str:
        """当任务超出能力范围或涉及高风险操作时，转人工处理。

        Args:
            reason: 转人工的原因，写清楚已尝试过什么
        """
        return f"已转人工（原因: {reason}）——人工将在 5 分钟内介入"

    client = MultiServerMCPClient({"crm": _stdio_connection("crm")})
    mcp_tools = await client.get_tools()
    all_tools = mcp_tools + [escalate_to_human]

    # 课程 05 的轨迹依赖"先核对 → 尝试扣款 → 失败后转人工"这条固定链路，
    # 而模型默认会偷懒（余额不足就直接答复，不调用 deduct / escalate）。
    # 这里用系统提示把行为钉住，教学演示才稳定；生产里这段应该来自 Server 的 Prompt 原语。
    system_prompt = (
        "你是客服助手，处理扣款请求时严格遵守三步：\n"
        "1) 先调用 get_customer 核对客户余额；\n"
        "2) 无论余额是否充足，都必须调用 deduct_balance 实际尝试一次，并把返回的失败原因原样转述；\n"
        "3) 如果扣款失败或金额超出能力范围，必须调用 escalate_to_human 转人工，"
        "reason 里写清已经尝试过什么。"
    )
    agent = create_agent(llm, tools=all_tools, system_prompt=system_prompt)
    tokens: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    final_state: dict[str, Any] = {}

    async for mode, chunk in agent.astream(
        {"messages": [("user", payload.question)]},
        stream_mode=["messages", "updates", "values"],
        config={"recursion_limit": 30},
    ):
        if mode == "messages":
            message, meta = chunk
            content = getattr(message, "content", "")
            # 只收模型吐的 token：ToolMessage 也会出现在 messages 流里，
            # 不筛的话工具返回值会被当成"模型说的话"混进气泡。
            is_ai = getattr(message, "type", "") in {"AIMessageChunk", "ai", "AIMessage"}
            if is_ai and isinstance(content, str) and content:
                tokens.append(
                    {
                        "node": (meta or {}).get("langgraph_node", ""),
                        "content": content,
                        "offset_ms": _elapsed_ms(started),
                    }
                )
        elif mode == "updates":
            for node, update in (chunk or {}).items():
                keys = list(update.keys()) if isinstance(update, dict) else []
                updates.append({"node": node, "keys": keys, "offset_ms": _elapsed_ms(started)})
        else:
            final_state = chunk if isinstance(chunk, dict) else final_state

    messages = final_state.get("messages") or []
    steps = _trace_steps(messages)
    answers = [step["content"] for step in steps if step["kind"] == "answer"]
    escalated = any(step.get("name") == "escalate_to_human" for step in steps if step["kind"] == "tool_call")

    return {
        "question": payload.question,
        "tools": [_tool_view(tool, "crm" if tool.name != "escalate_to_human" else "local") for tool in all_tools],
        "bubbles": _group_bubbles(tokens),
        "updates": updates,
        "steps": steps,
        "escalated": escalated,
        "final_answer": answers[-1] if answers else "",
        "elapsed_ms": _elapsed_ms(started),
        "model": cfg.model_name,
        "note": (
            "余额不足的报错由 CRM Server 返回、本地工具负责转人工、token 走 messages 流："
            "三种来源在 Agent 层看不出差别。"
        ),
    }


# ============================================================
# 6. 课程 04：streamable-http（后台线程起临时服务）
# ============================================================


async def run_http(payload: Day14HttpRequest) -> dict[str, Any]:
    """把 Server 部署成独立 HTTP 服务，再让客户端用 ``transport="http"`` 连上来。

    为了不污染用户环境，这里在**后台线程**里起一个随机端口的 uvicorn，
    调用完立刻 ``should_exit`` 关掉——equivalent to 课程里的
    ``uvicorn 04_http_server:mcp_app --port 8001``，只是端口随机、生命周期跟着请求。
    """
    _require_mcp()
    import uvicorn
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from mcp.server.fastmcp import FastMCP

    started = time.perf_counter()
    port = _free_port()
    mcp = FastMCP("team-tools", host="127.0.0.1", port=port)

    @mcp.tool()
    def team_announcement(topic: str = "本周") -> str:
        """获取团队公告。所有同事的 Agent 都可以调用这个共享工具。

        Args:
            topic: 时间范围，例如 "本周"
        """
        return f"{topic}：周五 17:00 全员同步会；新工具请登记到 MCP Server 列表。"

    app = mcp.streamable_http_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="day14-mcp-http")
    thread.start()

    url = f"http://127.0.0.1:{port}/mcp"
    startup_started = time.perf_counter()
    for _ in range(200):  # 最多等 10 秒
        if server.started:
            break
        await asyncio.sleep(0.05)
    startup_ms = _elapsed_ms(startup_started)

    tools_view: list[dict[str, Any]] = []
    content: list[str] = []
    is_error = False
    try:
        client = MultiServerMCPClient({"team": {"url": url, "transport": "http"}})
        tools = await client.get_tools()
        tools_view = [_tool_view(tool, "team") for tool in tools]
        async with client.session("team") as session:
            result = await session.call_tool("team_announcement", {"topic": payload.topic})
            content = _content_texts(result)
            is_error = bool(getattr(result, "isError", False))
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    return {
        "url": url,
        "startup_ms": startup_ms,
        "stopped": not thread.is_alive(),
        "tools": tools_view,
        "call_arguments": {"topic": payload.topic},
        "content": content,
        "is_error": is_error,
        "elapsed_ms": _elapsed_ms(started),
        "notes": [
            "Server 是独立进程/服务：多个 Host 可以同时连，状态与 Host 生命周期无关",
            'transport 写 "http"（streamable-http），旧写法 "sse" 已仅作向后兼容',
            "URL 以 /mcp 结尾——FastMCP 的 streamable-http 默认挂载路径",
            "客户端断开时 Server 日志里可能出现一条 ClosedResourceError(standalone SSE)，属正常现象",
        ],
    }


# ============================================================
# 7. 坑 1 现场：stdout 污染协议通道
# ============================================================


async def run_noisy(payload: Day14NoisyRequest) -> dict[str, Any]:
    """三种污染形态各跑一遍，用实测结果说明「stdout 是协议通道」到底多脆。

    实测结论（mcp 1.24 + langchain-mcp-adapters 0.3.1）：

    - ``print``  ：整行垃圾（``print`` 默认带换行）——客户端**可能**把这一行当垃圾跳过，
      也可能报 ``Invalid JSON``，取决于这行到达的时刻；
    - ``fd_line``：直接写 fd 1、带换行，和上一种同属"整行垃圾"，同样时好时坏；
    - ``fd``     ：直接写 fd 1、**不带换行**，半行文本和下一帧 JSON 粘在一起，
      客户端等不到一个完整帧 → **必然**挂死到超时。

    可以用 ``repeat`` 多跑几次来观察前两种的抖动——"时好时坏"本身就是要传达的信息：
    别赌客户端宽容。
    """
    _require_mcp()
    from langchain_mcp_adapters.client import MultiServerMCPClient

    modes = ["print", "fd_line", "fd"] if payload.mode == "all" else [payload.mode]
    timeout = payload.timeout_seconds
    repeat = payload.repeat
    cases: list[dict[str, Any]] = []

    for mode in modes:
        runs: list[dict[str, Any]] = []
        for _ in range(repeat):
            started = time.perf_counter()
            client = MultiServerMCPClient({"noisy": _stdio_connection("noisy", {"DAY14_NOISY_MODE": mode})})
            run: dict[str, Any] = {"ok": False, "tools": [], "call": [], "error": ""}
            try:
                tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)
                run["ok"] = True
                run["tools"] = [tool.name for tool in tools]
                async with client.session("noisy") as session:
                    result = await asyncio.wait_for(
                        session.call_tool("search_docs", {"query": "MCP"}), timeout=timeout
                    )
                    run["call"] = _content_texts(result)
            except asyncio.TimeoutError:
                run["error"] = f"TimeoutError：{timeout}s 内没等到完整帧（客户端挂死）"
            except Exception as exc:  # noqa: BLE001  解析错误/连接错误都按现场记录
                run["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
            run["elapsed_ms"] = _elapsed_ms(started)
            runs.append(run)

        ok_runs = sum(1 for run in runs if run["ok"])
        first = runs[0]
        cases.append(
            {
                "mode": mode,
                "repeat": repeat,
                "ok_runs": ok_runs,
                "stable": ok_runs in {0, repeat},
                "ok": ok_runs == repeat,
                "tools": first["tools"],
                "call": first["call"],
                "error": first["error"],
                "runs": runs,
                "elapsed_ms": sum(run["elapsed_ms"] for run in runs),
                "verdict": {
                    "print": "整行垃圾（print 自带换行）：可能被跳过，也可能报 Invalid JSON —— 看时序",
                    "fd_line": "整行垃圾（直接写 fd 1 + 换行）：同上，时好时坏",
                    "fd": "帧边界破裂（半行文本粘住 JSON）：必然挂死到超时",
                }[mode],
            }
        )

    return {
        "cases": cases,
        "explain": (
            "stdio 传输里 stdout 是 JSON-RPC 通道（一行一帧）。所以日志必须写 stderr："
            "logging 默认就是 stderr，或 print(..., file=sys.stderr)。"
        ),
        "sdk_note": (
            "课程原文说『客户端报 Invalid JSON 或直接挂死』——本机实测两种结局都复现了："
            "带换行的垃圾行通常被跳过（偶尔报错），不带换行的半行文本必然挂死。"
            "真正危险的不是 print 本身，而是『有人以为 stdout 可以随便写』这个前提。"
        ),
        "elapsed_ms": _elapsed_ms(started),
    }


# ============================================================
# 8. 工具函数给单测用（不需要起子进程）
# ============================================================


def load_server_module(key: str) -> Any:
    """按 key 导入 ``mcp_servers`` 里的脚本模块（单测用：不 spawn 也能看清单）。

    只 import，不运行 ``if __name__ == "__main__"`` 里的 ``mcp.run``，
    所以拿到的就是一个建好工具的 FastMCP 实例。
    """
    import importlib

    if key not in SERVER_SCRIPTS:
        raise ValueError(f"未知的 MCP Server：{key}")
    module_name = SERVER_SCRIPTS[key].removesuffix(".py")
    return importlib.import_module(f"app.modules.study.services.mcp_servers.{module_name}")


def static_tool_schema(key: str) -> dict[str, Any]:
    """不起子进程地读一个 Server 的工具定义（title / 参数 / description）。"""
    _require_mcp()
    module = load_server_module(key)
    server = module.mcp
    manager = getattr(server, "_tool_manager", None)
    tools = list(getattr(manager, "_tools", {}).values()) if manager else []
    return {
        "server": server.name,
        "tools": [
            {
                "name": getattr(tool, "name", ""),
                "description": (getattr(tool, "description", "") or "").strip(),
                "input_schema": _jsonable(getattr(tool, "parameters", {}) or {}),
            }
            for tool in tools
        ],
    }


def parse_json_arguments(raw: str | None) -> dict[str, Any]:
    """前端传上来的参数可能是 JSON 字符串，解析失败要给可读错误而不是 500。"""
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"参数不是合法 JSON：{exc.msg}（位置 {exc.pos}）") from exc
    if not isinstance(value, dict):
        raise ValueError("参数必须是一个 JSON 对象，例如 {\"customer_id\": \"C001\", \"amount\": 100}")
    return value
