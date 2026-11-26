"""Day 15 可观测性：Agent 上线后，你怎么知道它哪里错了。

课程 day15 的五个脚本在这里收成一个模块：

    LangSmith 零代码接入   → ``run_platform(provider="langsmith")`` （01：环境变量 + tags/metadata）
    自建 CallbackHandler   → ``run_callback``                     （02：从回调事件还原 trace 树）
    Langfuse 接入与打分    → ``run_platform(provider="langfuse")`` （03：追踪 + 用户反馈回填）
    自建轻量 Trace         → ``run_own_trace``                    （04：八字段 JSONL + 回放 + 指标聚合）
    跨进程全链路            → ``run_full_stack``                   （05：trace_id 传给 MCP Server 串起两侧）

四条贯穿全篇的结论：

1. **Agent 的 bug 大多不是异常，而是"软失败"**：流程跑完、不报错、答案错了。
   唯一可靠的排查手段是把每一次运行的完整轨迹（输入 messages / 输出 / 耗时 / token）存下来；
2. **可观测性三大件是层层推导的关系**：Trace 是原料 → Metrics 从 Trace 聚合 → 告警与优化决策从 Metrics 来。
   平台只是替你实现了"采集 + 存储 + UI"这三段管道；
3. **所有平台的接入点都是 ``config={"callbacks": [...]}``**：LangSmith / Langfuse 都是往这个口子里塞 Handler，
   理解了它，换任何平台都是十分钟的事；
4. **跨进程链路靠一个 ``trace_id`` 串起来**：LangGraph 内部有 trace、MCP Server 端没有，
   所以客户端把 trace_id 当工具参数传过去，两侧各写各的 JSONL，排查时按 id 一拼就是完整链路。

与课程示例的五个工程差异（都是踩过之后改的）：

- 课程 02 用"全局 ``self._depth`` 加减缩进"近似还原调用树；本模块改用回调自带的
  ``parent_run_id`` 精确建树——并发时缩进会串，parent_id 不会；
- 课程 04 在 ``call_llm`` 里把 token 存到 ``_ctx.attrs``，但 ``span`` 在 ``finally`` 里写盘，
  那份 attrs 根本没进 JSONL；本模块让 ``span()`` **yield 出 record**，调用方直接改
  ``record["attributes"]``，写盘时才做脱敏，token 一定落得下去；
- 课程 04 用 ``threading.local`` 存当前 span；本模块是 async FastAPI 服务，改用
  ``contextvars.ContextVar``（``threading.local`` 在 asyncio 里会让同线程的并发任务互相串线）；
- 课程 05 的 MCP Server 直接写死日志路径；本模块通过环境变量 ``DAY15_SERVER_SPANS`` 下传，
  子进程和宿主写同一份目录，单测也就能指向临时文件；
- 平台接入**不在请求里改环境变量**：LangSmith 的 key 只从 ``.env`` 读，没配就返回"怎么配"的说明，
  而不是悄悄替你打开数据出境开关。

langchain / langgraph / mcp / langfuse 全部延迟导入：缺依赖时接口返回可读的 400，
而不是让整个应用起不来。
"""

import asyncio
import json
import math
import os
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import (
    Day15CallbackRequest,
    Day15FeedbackRequest,
    Day15FullStackRequest,
    Day15OwnTraceRequest,
    Day15PlatformRequest,
)
from app.modules.study.services.day09_langchain import build_chat_model
from app.modules.study.services.day10_langgraph import _jsonable, _text
from app.modules.study.services.llm import resolve_model

INSTALL_HINT = "pip install langgraph langchain-openai（Langfuse 另需 pip install langfuse）"

# ============================================================
# 运行时文件目录：var/ 已在 .gitignore，重启不清空也是本意（能回看历史 trace）
# ============================================================

TRACE_DIR = Path(__file__).resolve().parents[4] / "var" / "day15"
TRACE_FILE = TRACE_DIR / "traces.jsonl"
BUSINESS_FILE = TRACE_DIR / "business_log.jsonl"
SERVER_FILE = TRACE_DIR / "server_spans.jsonl"
FEEDBACK_FILE = TRACE_DIR / "feedback.jsonl"

# MCP Server 脚本：跨进程追踪的演示
EXPENSE_SERVER_PATH = Path(__file__).parent / "mcp_servers" / "expense_server.py"

# 成本估算单价（元 / 1k token）：按自己的模型单价替换，这里只为把"token → 钱"这条链路跑通
PRICE_IN_PER_1K = 0.15
PRICE_OUT_PER_1K = 0.60

# 报销助手的两套提示词：普通版 / 全链路版（后者要把 trace_id 钉进工具参数）
EXPENSE_PROMPT = (
    "你是公司报销助手。先查制度再提交，金额不确定必须询问用户。"
    "工具返回什么就转述什么，不要编造单号。"
)
FULLSTACK_PROMPT = (
    "你是公司报销助手。先查制度再提交，金额不确定必须询问用户。"
    "调用任何工具时，必须把 trace_id='{trace_id}' 作为参数原样传入；"
    "工具返回什么就转述什么，不要编造单号。"
)

# ============================================================
# 静态概念数据（供前端渲染，全部来自课程原文）
# ============================================================

# 1.1 传统应用 vs Agent 的失败形态
FAILURE_MODES: list[dict[str, str]] = [
    {
        "dimension": "失败形态",
        "web": "抛异常、5xx、日志里有堆栈",
        "agent": "「软失败」：流程走完、不报错，但答案错了",
    },
    {
        "dimension": "一次请求",
        "web": "一个请求一条日志链",
        "agent": "一次提问 = N 次 LLM 调用 + N 次工具调用，嵌套链路",
    },
    {
        "dimension": "确定性",
        "web": "同样输入同样输出",
        "agent": "同样输入每次都不同，复现靠运气",
    },
    {
        "dimension": "上下文",
        "web": "请求参数即全部",
        "agent": "模型「看到了什么」（完整 messages）才是一切，不记录就永远丢了",
    },
    {
        "dimension": "排查手段",
        "web": "grep 日志基本够用",
        "agent": "grep 日志基本没用——要看每一步的输入/输出/耗时/token",
    },
]

# 1.2 三大件在 Agent 语境下的含义
PILLARS: list[dict[str, str]] = [
    {
        "pillar": "Tracing 链路追踪",
        "traditional": "span 树、调用链",
        "agent": "一次 run 的完整树：node → LLM call → tool call，每层记输入/输出/耗时/token/错误",
        "source": "LangSmith / Langfuse / 自建 span()",
    },
    {
        "pillar": "Metrics 指标",
        "traditional": "QPS、错误率",
        "agent": "token 消耗、每次成本、端到端 p50/p95、工具报错率、终止原因分布（正常收敛/超步截断/降级交付）",
        "source": "从 Trace 聚合（本模块 summarize）",
    },
    {
        "pillar": "Logging 日志",
        "traditional": "结构化日志",
        "agent": "结构化 JSONL 事件流（不是 print——Day 14 刚踩过 stdout 污染协议通道的坑）",
        "source": "业务事件 / 审计留痕",
    },
]

# 1.3 两条技术路线
ROUTES: list[dict[str, str]] = [
    {
        "route": "平台托管",
        "examples": "LangSmith（官方）、Langfuse（开源可自部署）",
        "pros": "零侵入、UI 强大、自带评估与数据集",
        "cons": "LangSmith 数据出境/收费；Langfuse 自部署要运维",
    },
    {
        "route": "自建轻量",
        "examples": "自定义 CallbackHandler + JSONL + 聚合脚本",
        "pros": "数据完全自持、零依赖、适合内网/合规场景",
        "cons": "没有 UI，查询要自己写脚本",
    },
    {
        "route": "今天的选择",
        "examples": "两条都跑一遍",
        "pros": "先用平台建立「轨迹长什么样」的直觉，再用自建拆开原理",
        "cons": "你在任何平台看到的树，结构和自建 JSONL 一模一样",
    },
]

# trace 数据模型的八个字段：记住它，任何 APM 平台都能看懂
TRACE_FIELDS: list[dict[str, str]] = [
    {"field": "trace_id", "meaning": "一次完整 Agent 请求（一条链路）", "example": "3f9a2b1c8d4e"},
    {"field": "span_id", "meaning": "链路里的一个节点（LLM / 工具 / 步骤）", "example": "a1b2c3d4"},
    {"field": "parent_id", "meaning": "父 span；为 null 就是根节点", "example": "3f9a2b1c8d4e（根）"},
    {"field": "name", "meaning": "节点名字（agent-run / chat / tool:xxx）", "example": "tool:search_policy"},
    {"field": "start_ts", "meaning": "开始时间戳（排序与回放用）", "example": "1764216000.123"},
    {"field": "duration_ms", "meaning": "耗时（性能瓶颈只能从它看出来）", "example": "1830.4"},
    {"field": "attributes", "meaning": "任意维度：model / tokens / user_id / 工具参数", "example": "{'input_tokens': 289}"},
    {"field": "status", "meaning": "ok / error（错误要能定位到具体 span）", "example": "error"},
]

# 指标定义：Metrics 全是从 Trace 聚合出来的
METRIC_DEFS: list[dict[str, str]] = [
    {
        "metric": "端到端延迟 p50 / p95",
        "how": "按 trace 根 span 的 duration_ms 取分位",
        "why": "p95 才能暴露「少量用户很慢」，平均值会把它抹平",
    },
    {
        "metric": "token 与成本",
        "how": "累加 llm span 的 input/output_tokens，乘单价",
        "why": "成本异常往往先于质量异常出现（prompt 膨胀、工具结果过长）",
    },
    {
        "metric": "工具报错率",
        "how": "tool span 里 status=error 的占比",
        "why": "工具坏了模型只会换个说法继续编，不聚合就看不出系统性失败",
    },
    {
        "metric": "终止原因分布",
        "how": "根 span 的 termination 字段计数",
        "why": "正常收敛 / 超步截断 / 降级交付的比例，直接反映 Agent 是否真的能干活",
    },
    {
        "metric": "点赞率（用户反馈）",
        "how": "trace 上的 score 聚合",
        "why": "软失败机器判不了，用户点踩是免费的真值标签，也是明天的评估集来源",
    },
]

# 第 4 节：五个常见坑
PITFALLS: list[dict[str, str]] = [
    {
        "no": "1",
        "title": "只记 LLM 调用，不记工具输入输出全文",
        "symptom": "token 数只告诉你「贵」，查不出「错」",
        "fix": "工具参数与返回原文都要留（可截断到几千字符）；保存成本高不过一次线上事故",
    },
    {
        "no": "2",
        "title": "trace 里塞了敏感数据又不做隔离",
        "symptom": "messages 全文含手机号、身份证、密钥，存储侧裸奔",
        "fix": "Handler / span 里做字段过滤与 PII 脱敏（本模块 span() 落盘前统一 scrub）",
    },
    {
        "no": "3",
        "title": "Handler 忘记 flush",
        "symptom": "本地跑 10 次只有 6 条 trace，以为是采样，其实是丢报",
        "fix": "Langfuse Handler 异步批量上报；短命进程（CLI / serverless）必须 flush()",
    },
    {
        "no": "4",
        "title": "多用户并发时 trace 串线",
        "symptom": "两个请求的 parent/child 互相污染，树长得莫名其妙",
        "fix": "同步用 threading.local，异步用 contextvars.ContextVar；绝不用模块级全局变量",
    },
    {
        "no": "5",
        "title": "把可观测性当成「上了平台就完事」",
        "symptom": "数据没人看，告警没阈值，事故后还是靠猜",
        "fix": "指标看板 + 告警阈值（如错误 trace 占比 > 5%）+ 每周 trace review 例会，缺一不可",
    },
]

# 第 5 节：速查表
CHEATSHEET: list[dict[str, str]] = [
    {"topic": "最快接入追踪", "content": "LangSmith 环境变量：LANGSMITH_TRACING=true + API_KEY + PROJECT"},
    {"topic": "开源自部署", "content": "from langfuse.langchain import CallbackHandler"},
    {"topic": "拿本次 trace id", "content": "handler.get_trace_id()"},
    {"topic": "用户反馈落 trace", "content": "langfuse.score(trace_id, name='user-feedback', value=1)"},
    {"topic": "run 上打标签", "content": "config={'tags': [...], 'metadata': {...}}"},
    {"topic": "自建探针", "content": "config={'callbacks': [MyHandler()]}"},
    {"topic": "自建数据模型", "content": "八字段：trace_id/span_id/parent_id/name/start/duration/attributes/status"},
    {"topic": "并发不串线", "content": "threading.local（同步）/ contextvars.ContextVar（异步）"},
    {"topic": "跨进程串联", "content": "把 trace_id 当工具参数或 header 传给 MCP Server"},
    {"topic": "核心指标", "content": "端到端 p50/p95、token 与成本、工具错误率、终止原因分布"},
]

CHECKLIST: list[str] = [
    "能说清 Agent 的「软失败」为什么让 grep 日志失效",
    "能画出 Agent trace 树，并说出每个节点该记什么",
    "能用三个环境变量接入 LangSmith，用 tags/metadata 做维度筛选",
    "能写自定义 CallbackHandler，并从 parent_run_id 还原调用树",
    "能说出八字段数据模型，并自己实现 span 上下文管理器",
    "能分清 threading.local 与 contextvars 的适用场景",
    "能把 trace 聚合成 p50/p95、token 成本、工具错误率、终止原因分布",
    "能把 trace_id 传过进程边界，把客户端与服务端日志拼成一条链路",
    "知道 PII 脱敏和 Handler flush 两条上线前的必做项",
]

PRESETS: dict[str, list[str]] = {
    "callback": [
        "帮我报销昨天的午餐 45 元",
        "报销高铁票 237 元",
    ],
    "own": [
        "报销高铁票 237 元",
        "帮我报销打车费 32 元，先看下交通制度",
    ],
    "fullstack": [
        "报销昨天的晚餐 78 元",
        "报销高铁票 237 元",
    ],
}


# ============================================================
# 0. 公共设施：依赖探测 / 版本 / 时间 / 脱敏
# ============================================================


def _require_langchain() -> None:
    """依赖探测：LangChain 缺失时给安装命令，而不是 ImportError 堆栈。"""
    try:
        import langchain.agents  # noqa: F401
        import langchain_core.tools  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 只在缺依赖的环境触发
        raise ValueError(f"未安装 langchain，请先执行：{INSTALL_HINT}") from exc


def _require_mcp() -> None:
    """依赖探测：跨进程全链路演示需要 mcp + adapters。"""
    try:
        import langchain_mcp_adapters.client  # noqa: F401
        import mcp.server.fastmcp  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ValueError('未安装 mcp / langchain-mcp-adapters，请先执行：pip install "mcp[cli]" langchain-mcp-adapters') from exc


def _require_langfuse() -> None:
    """依赖探测：Langfuse 是可选依赖，没装就告诉用户怎么装。"""
    try:
        import langfuse.langchain  # noqa: F401
    except ImportError as exc:
        raise ValueError("未安装 langfuse，请先执行：pip install langfuse") from exc


_DOTENV_LOADED = False


def _load_env_once() -> None:
    """把 .env 里的键补进 os.environ（只补缺失的，不覆盖真实环境变量）。

    为什么要这一步：LangSmith / Langfuse 的 SDK 直接读 ``os.environ``，
    而 pydantic-settings 只把 .env 读进 ``settings`` 对象、**不会导出到进程环境**。
    不桥接的话，用户在 .env 里配了 key，追踪却依然是关的——这是非常难查的坑。
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except Exception:  # noqa: BLE001  python-dotenv 缺失时退化为只看真实环境变量
        return


def _version(name: str) -> str:
    """读已安装包版本：可观测性 SDK 的 API 各版本差异不小，先看版本再查文档。"""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001  没装就是没装
        return "未安装"


def _installed(name: str) -> bool:
    """只判断包在不在（不 import，避免触发副作用）。"""
    try:
        from importlib.util import find_spec

        return find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def _elapsed_ms(started: float) -> int:
    """从起点到现在的毫秒数。"""
    return int((time.perf_counter() - started) * 1000)


def _short(value: Any, limit: int = 200) -> str:
    """把任意值压成单行短文本（轨迹展示用，别把整份 messages 塞进响应）。"""
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ---- PII 脱敏：坑 2 的落地实现 ----

# 手机号 / 18 位身份证 / 邮箱：三类最常见的画像字段
_PII_PATTERNS: list[tuple[re.Pattern[str], Any]] = [
    (re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)"), lambda m: f"{m.group(1)[:3]}****{m.group(1)[-4:]}"),
    (re.compile(r"(?<!\d)(\d{6})\d{8}(\d{3}[\dXx])(?!\d)"), lambda m: f"{m.group(1)}********{m.group(2)}"),
    (re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*(@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"), lambda m: f"{m.group(1)}***{m.group(2)}"),
]

# 这些 key 的值直接整段打码，连长度都不留
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "password",
        "passwd",
        "secret",
        "secret_key",
        "token",
        "access_token",
        "id_card",
        "身份证",
    }
)


def mask_pii(text: str) -> str:
    """把文本里的手机号 / 身份证 / 邮箱打码，其余内容保持原样。"""
    masked = text
    for pattern, repl in _PII_PATTERNS:
        masked = pattern.sub(repl, masked)
    return masked


def redact(value: Any, *, key: str = "") -> Any:
    """递归脱敏：敏感 key 整段打码，字符串里的 PII 做掩码。

    写盘前统一调用（而不是在构造 record 时调用）：调用方后补的 token 数、
    工具返回值也能一并被处理，不会出现「先脱敏后补数据」的漏洞。
    """
    if key.lower() in SENSITIVE_KEYS:
        return "***"
    if isinstance(value, dict):
        return {str(k): redact(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item, key=key) for item in value]
    if isinstance(value, str):
        return mask_pii(value)
    return value


# ============================================================
# 1. 自建 Trace：span 上下文管理器 + JSONL 落盘（课程 04 的骨架）
# ============================================================


class SpanTracer:
    """自建轻量追踪器：一个实例 = 一条链路的事件收集器。

    并发隔离用 ``contextvars.ContextVar``（不是 ``threading.local``）：
    FastAPI 里同一个线程会跑很多个协程，thread-local 变量会被这些协程共享，
    一个请求的 parent 很容易串到另一个请求上。ContextVar 的每个 asyncio task
    自带一份上下文副本，天然隔离。

    ``persist=False`` 时只存在内存（单测、并发演示用，不污染 traces.jsonl）。
    """

    def __init__(self, path: Path | None = None, *, persist: bool = True, scrub: bool = True) -> None:
        """初始化：path 为 JSONL 落盘路径，persist=False 只留内存，scrub 控制是否脱敏。"""
        self.path = Path(path) if path is not None else TRACE_FILE
        self.persist = persist
        self.scrub = scrub
        self.records: list[dict[str, Any]] = []
        self._state: ContextVar[tuple[str | None, str | None]] = ContextVar(
            f"day15-span-{id(self)}", default=(None, None)
        )
        self._lock = threading.Lock()

    @property
    def trace_id(self) -> str | None:
        """当前上下文所属的 trace_id（没进 span 时为 None）。"""
        return self._state.get()[0]

    @contextmanager
    def span(self, name: str, kind: str = "step", **attributes: Any):
        """开一个 span：``with tracer.span("chat", kind="llm", model=...) as record:``。

        yield 出来的是 record 本身——调用方在 with 块里往 ``record["attributes"]``
        补 token、工具返回值，退出时统一落盘（这就修掉了课程 04 里
        ``_ctx.attrs`` 写不进 JSONL 的问题）。

        最外层 span（上下文里没有 trace_id）自动开一条新 trace；
        其他情况沿用当前 trace 并把当前 span 记为 parent。
        """
        trace_id, parent_id = self._state.get()
        is_root = trace_id is None
        if trace_id is None:
            trace_id = uuid.uuid4().hex[:12]
        span_id = uuid.uuid4().hex[:8]

        record: dict[str, Any] = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_id": parent_id,
            "name": name,
            "kind": kind,
            "start_ts": time.time(),
            "attributes": dict(attributes),
            "status": "ok",
        }
        token = self._state.set((trace_id, span_id))
        started = time.perf_counter()
        error: BaseException | None = None
        try:
            yield record
        except BaseException as exc:  # noqa: BLE001  KeyboardInterrupt 也要记一笔再抛
            error = exc
            raise
        finally:
            record["duration_ms"] = round((time.perf_counter() - started) * 1000, 1)
            if error is not None:
                record["status"] = "error"
                record["error"] = repr(error)
            # 允许调用方在块内主动把 status 标成 error（工具报错但被吞掉的情形）
            record["attributes"] = (
                redact(record.get("attributes") or {}) if self.scrub else record.get("attributes") or {}
            )
            self._emit(record)
            self._state.reset(token)

    def _emit(self, record: dict[str, Any]) -> None:
        """写内存 + 追加 JSONL（加锁，避免多线程把一行写成两半）。"""
        with self._lock:
            self.records.append(record)
            if not self.persist:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, default=str)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def tree(self) -> list[dict[str, Any]]:
        """把已收集的 record 还原成树（这就是你在 LangSmith UI 里看到的那棵树）。"""
        return span_tree(self.records)


def span_tree(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 parent_id 把扁平 record 还原成嵌套树（纯函数，可单测）。"""
    nodes: dict[str, dict[str, Any]] = {}
    for record in records:
        nodes[record["span_id"]] = {
            "span_id": record["span_id"],
            "trace_id": record["trace_id"],
            "parent_id": record.get("parent_id"),
            "name": record.get("name", ""),
            "kind": record.get("kind", "step"),
            "status": record.get("status", "ok"),
            "start_ts": record.get("start_ts", 0),
            "duration_ms": record.get("duration_ms", 0),
            "attributes": record.get("attributes") or {},
            "error": record.get("error"),
            "children": [],
        }

    roots: list[dict[str, Any]] = []
    for node in nodes.values():
        parent = nodes.get(node["parent_id"]) if node["parent_id"] else None
        if parent is None:
            roots.append(node)
        else:
            parent["children"].append(node)

    def sort_children(node: dict[str, Any]) -> None:
        node["children"].sort(key=lambda item: item.get("start_ts") or 0)
        for child in node["children"]:
            sort_children(child)

    roots.sort(key=lambda item: item.get("start_ts") or 0)
    for root in roots:
        sort_children(root)
    return roots


def flatten_tree(tree: list[dict[str, Any]], depth: int = 0) -> list[dict[str, Any]]:
    """把树压平成带 depth 的行（前端用表格画树比递归组件省事）。"""
    rows: list[dict[str, Any]] = []
    for node in tree:
        rows.append({**node, "depth": depth, "children": None})
        rows.extend(flatten_tree(node.get("children") or [], depth + 1))
    return rows


def percentile(values: list[float], ratio: float) -> float:
    """最近秩分位数（nearest-rank）：p95 用这个算法就够，别引入 numpy。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(ratio * len(ordered)) - 1))
    return float(ordered[index])


def summarize(
    records: list[dict[str, Any]],
    *,
    price_in: float = PRICE_IN_PER_1K,
    price_out: float = PRICE_OUT_PER_1K,
) -> dict[str, Any]:
    """从原始 trace 聚合出 Metrics（可观测性三大件的第三件）。

    - 延迟只看**根 span**：子 span 累加会把并行分支重复计算；
    - 成本按 ``llm`` span 的 token 累加；
    - 终止原因读根 span 的 ``attributes.termination``（没有就记 unknown）。
    """
    spans = list(records or [])
    roots: dict[str, dict[str, Any]] = {}
    for record in spans:
        if record.get("parent_id") is None:
            roots.setdefault(record["trace_id"], record)

    llm_spans = [record for record in spans if record.get("kind") == "llm"]
    tool_spans = [record for record in spans if record.get("kind") == "tool"]

    latencies = [float(root.get("duration_ms") or 0) for root in roots.values()]
    input_tokens = sum(int((record.get("attributes") or {}).get("input_tokens") or 0) for record in llm_spans)
    output_tokens = sum(int((record.get("attributes") or {}).get("output_tokens") or 0) for record in llm_spans)
    tool_errors = sum(1 for record in tool_spans if record.get("status") == "error")
    error_traces = sum(1 for root in roots.values() if root.get("status") == "error")

    termination: dict[str, int] = {}
    steps: list[int] = []
    for root in roots.values():
        attributes = root.get("attributes") or {}
        reason = str(attributes.get("termination") or ("error" if root.get("status") == "error" else "unknown"))
        termination[reason] = termination.get(reason, 0) + 1
        if attributes.get("steps") is not None:
            steps.append(int(attributes["steps"]))

    trace_count = len(roots)
    return {
        "traces": trace_count,
        "spans": len(spans),
        "llm_calls": len(llm_spans),
        "tool_calls": len(tool_spans),
        "tool_errors": tool_errors,
        "tool_error_rate": round(tool_errors / len(tool_spans), 4) if tool_spans else 0.0,
        "error_traces": error_traces,
        "error_rate": round(error_traces / trace_count, 4) if trace_count else 0.0,
        "latency_p50_ms": round(percentile(latencies, 0.50), 1),
        "latency_p95_ms": round(percentile(latencies, 0.95), 1),
        "latency_max_ms": round(max(latencies), 1) if latencies else 0.0,
        "avg_steps": round(sum(steps) / len(steps), 2) if steps else 0.0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_cny": round(input_tokens / 1000 * price_in + output_tokens / 1000 * price_out, 4),
        "termination": termination,
    }


# ============================================================
# 2. 自建 CallbackHandler：从回调事件还原 trace 树（课程 02）
# ============================================================


class CallbackCollector:
    """回调事件的收集器（纯逻辑，可单测）。

    每个 LangChain 组件在生命周期关键点都会广播 on_xxx_start / on_xxx_end，
    并带上 ``run_id`` 与 ``parent_run_id``。收集器只做三件事：
    开记录、关记录（补耗时与状态）、按 parent 建树。
    """

    def __init__(self) -> None:
        """初始化空的记录表。"""
        self.records: list[dict[str, Any]] = []
        self._open: dict[str, dict[str, Any]] = {}

    def start(
        self,
        run_id: Any,
        name: str,
        kind: str,
        parent_run_id: Any = None,
        **attributes: Any,
    ) -> dict[str, Any]:
        """记一个节点开始（run_id 转成字符串，便于 JSON 与前端使用）。"""
        key = str(run_id)
        parent = str(parent_run_id) if parent_run_id is not None else None
        # 父节点没被记录（例如被过滤掉的通用 chain）时改挂到已有根上，避免整段子树丢失
        if parent is not None and parent not in {record["span_id"] for record in self.records}:
            roots = [record for record in self.records if record.get("parent_id") is None]
            parent = roots[0]["span_id"] if roots else None
        record: dict[str, Any] = {
            "trace_id": str(run_id) if parent is None else "",
            "span_id": key,
            "parent_id": parent,
            "name": name,
            "kind": kind,
            "start_ts": time.time(),
            "duration_ms": 0.0,
            "attributes": dict(attributes),
            "status": "ok",
        }
        self._open[key] = record
        self.records.append(record)
        return record

    def end(
        self,
        run_id: Any,
        *,
        status: str = "ok",
        error: BaseException | None = None,
        output: Any = None,
        **attributes: Any,
    ) -> dict[str, Any] | None:
        """关闭一个节点：补耗时、状态、输出与额外属性；找不到就返回 None（如被过滤）。"""
        record = self._open.pop(str(run_id), None)
        if record is None:
            return None
        record["duration_ms"] = round((time.time() - record["start_ts"]) * 1000, 1)
        record["status"] = status
        record["attributes"].update(attributes)
        if output is not None:
            record["attributes"]["output"] = _short(output, 2000)
        if error is not None:
            record["status"] = "error"
            record["error"] = repr(error)
        return record

    def tree(self) -> list[dict[str, Any]]:
        """把记录还原成树（统一 trace_id 后交给 span_tree）。"""
        roots = [record for record in self.records if record.get("parent_id") is None]
        trace_id = roots[0]["span_id"] if roots else "callback"
        for record in self.records:
            record["trace_id"] = trace_id
        return span_tree(self.records)


# 只关心这些 chain：LangGraph 的节点名 + create_agent 的包装层
INTERESTING_CHAINS = {"LangGraph", "agent", "tools", "model", "should_continue"}


def _model_name(serialized: Any, metadata: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
    """从回调参数里尽量还原模型名（不同厂商放在不同字段里）。"""
    params = kwargs.get("invocation_params") or {}
    for key in ("model_name", "model", "_model_name"):
        value = params.get(key)
        if value:
            return str(value)
    if isinstance(serialized, dict):
        for key in ("name", "id"):
            if serialized.get(key):
                return str(serialized[key])
    return str((metadata or {}).get("ls_model_name") or "llm")


def _extract_usage(response: Any) -> dict[str, int]:
    """从 LLMResult 里抠 token 用量：llm_output 与 usage_metadata 两种形态都兼容。"""
    usage: dict[str, Any] = {}
    llm_output = getattr(response, "llm_output", None) or {}
    raw = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if raw:
        usage = {
            "input_tokens": raw.get("prompt_tokens") or raw.get("input_tokens") or 0,
            "output_tokens": raw.get("completion_tokens") or raw.get("output_tokens") or 0,
            "total_tokens": raw.get("total_tokens") or 0,
        }
    if not usage.get("total_tokens"):
        for generation_list in getattr(response, "generations", []) or []:
            for generation in generation_list or []:
                meta = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                if meta:
                    usage = {
                        "input_tokens": meta.get("input_tokens") or 0,
                        "output_tokens": meta.get("output_tokens") or 0,
                        "total_tokens": meta.get("total_tokens") or 0,
                    }
                    break
            if usage:
                break
    return {key: int(value or 0) for key, value in usage.items()}


def build_callback_handler(collector: CallbackCollector):
    """构造一个订阅生命周期事件的 Handler（延迟导入 BaseCallbackHandler）。

    过滤策略：LLM 与工具全记；chain 只记 LangGraph 节点与根链路，
    否则 RunnableSequence 之类的通用包装会把树糊成一片，反而看不出业务步骤。
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class CallbackTraceHandler(BaseCallbackHandler):
        """把回调事件喂给 CallbackCollector 的探针（就是平台的接入点）。"""

        raise_error = False

        def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs):
            """只记录有意义的 chain：根链路 + LangGraph 节点名。

            实测（langchain 1.x）：run 的名字在 ``kwargs["name"]`` 里，
            ``serialized`` 往往是空的；LangGraph 节点还会在
            ``metadata["langgraph_node"]`` 里再标一次，优先用它。
            """
            name = (metadata or {}).get("langgraph_node") or kwargs.get("name")
            if not name and isinstance(serialized, dict):
                name = serialized.get("name")
            name = str(name or "chain")
            if parent_run_id is not None and name not in INTERESTING_CHAINS:
                return
            collector.start(run_id, name, "chain", parent_run_id, inputs=_short(inputs, 400))

        def on_chain_end(self, outputs, *, run_id, **kwargs):
            """关闭 chain，输出只留短预览。"""
            collector.end(run_id, output=outputs)

        def on_chain_error(self, error, *, run_id, **kwargs):
            """chain 出错：状态标 error，异常原文进 record。"""
            collector.end(run_id, error=error)

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, metadata=None, **kwargs):
            """对话模型开始：记模型名、消息条数、最后一条用户消息的预览。"""
            previews: list[dict[str, str]] = []
            for message in (messages[0] if messages else [])[-4:]:
                role = getattr(message, "type", "message")
                previews.append({"role": role, "content": _short(_text(message), 200)})
            collector.start(
                run_id,
                _model_name(serialized, metadata, kwargs),
                "llm",
                parent_run_id,
                n_messages=len(messages[0]) if messages else 0,
                preview=previews,
                tags=kwargs.get("tags") or [],
            )

        def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, metadata=None, **kwargs):
            """文本补全入口（老式 LLM）：同样记一条，避免漏掉非 chat 模型。"""
            collector.start(
                run_id,
                _model_name(serialized, metadata, kwargs),
                "llm",
                parent_run_id,
                n_messages=len(prompts or []),
                preview=[{"role": "prompt", "content": _short(prompts[0] if prompts else "", 200)}],
            )

        def on_llm_end(self, response, *, run_id, **kwargs):
            """LLM 结束：token 用量就在这里，平台统计成本的原始数据。"""
            usage = _extract_usage(response)
            text = ""
            try:
                text = _text(response.generations[0][0].message)
            except Exception:  # noqa: BLE001  非 chat 返回没有 message
                text = ""
            collector.end(run_id, output=text, **usage)

        def on_llm_error(self, error, *, run_id, **kwargs):
            """LLM 报错：状态标 error（超时、限流、内容审核拒答都走这里）。"""
            collector.end(run_id, error=error)

        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
            """工具开始：参数原文要留（坑 1：没有它 trace 白记）。"""
            name = (serialized or {}).get("name") if isinstance(serialized, dict) else "tool"
            collector.start(run_id, str(name), "tool", parent_run_id, input=_short(input_str, 2000))

        def on_tool_end(self, output, *, run_id, **kwargs):
            """工具结束：返回原文（截断到 2000 字符）。"""
            collector.end(run_id, output=output)

        def on_tool_error(self, error, *, run_id, **kwargs):
            """工具报错：这是工具错误率指标的原料。"""
            collector.end(run_id, error=error)

    return CallbackTraceHandler()


def callback_timeline(records: list[dict[str, Any]], started: float) -> list[dict[str, Any]]:
    """把回调记录摊平成时间轴（前端按 offset_ms 画甘特条）。"""
    baseline = min((record.get("start_ts") or started for record in records), default=started)
    rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: item.get("start_ts") or 0):
        attributes = record.get("attributes") or {}
        detail = ""
        if record.get("kind") == "llm":
            detail = f"tokens={attributes.get('total_tokens', 0)} · {attributes.get('model', '')}"
        elif record.get("kind") == "tool":
            detail = _short(attributes.get("input") or attributes.get("output") or "", 120)
        elif record.get("kind") == "chain":
            detail = _short(attributes.get("inputs") or attributes.get("output") or "", 120)
        rows.append(
            {
                "name": record.get("name"),
                "kind": record.get("kind"),
                "status": record.get("status"),
                "offset_ms": round(((record.get("start_ts") or 0) - baseline) * 1000),
                "duration_ms": record.get("duration_ms"),
                "detail": detail,
                "parent_id": record.get("parent_id"),
            }
        )
    return rows


# ============================================================
# 3. 报销助手工具（两套演示共用同一份实现）
# ============================================================

_EXPENSE_SEQ = count(2043)

EXPENSE_POLICIES: dict[str, str] = {
    "餐饮": "差旅餐饮每日报销上限 100 元，需发票。",
    "交通": "高铁二等座全额报销，打车需注明事由。",
    "住宿": "一线城市住宿每晚上限 600 元，需水单。",
    "退款": "商品退款 3 个工作日内到账。",
}


def build_expense_tools() -> list[Any]:
    """构造课程 01/02 的两个本地工具（进程内版本，MCP 版本在 expense_server.py）。"""
    from langchain_core.tools import tool

    @tool
    def search_expense_policy(keyword: str) -> str:
        """查询公司报销制度。输入关键词，返回相关条款。

        Args:
            keyword: 制度关键词，例如 餐饮 / 交通 / 住宿 / 退款
        """
        return EXPENSE_POLICIES.get(keyword, "未找到相关条款。")

    @tool
    def submit_expense(amount: float, category: str, reason: str) -> str:
        """提交报销申请，返回单号。

        Args:
            amount: 金额（元）
            category: 类别，例如 餐饮 / 交通 / 住宿
            reason: 事由
        """
        return f"报销单已创建：{category} {amount} 元，事由：{reason}。单号 EXP-{next(_EXPENSE_SEQ)}。"

    return [search_expense_policy, submit_expense]


def _tool_call_names(messages: list[Any]) -> list[dict[str, Any]]:
    """从消息序列里抽出工具调用（轨迹与指标共用）。"""
    calls: list[dict[str, Any]] = []
    for message in messages or []:
        for call in getattr(message, "tool_calls", None) or []:
            calls.append({"name": call.get("name", ""), "args": _jsonable(call.get("args") or {})})
    return calls


def classify_termination(messages: list[Any], *, max_steps: int | None = None) -> str:
    """判定终止原因：normal / truncated / pending / empty。

    软失败排查里很关键：truncated（超步截断）和 normal（模型自己收口）
    在日志里长得几乎一样，但前者意味着「答案没写完」。
    """
    if not messages:
        return "empty"
    last = messages[-1]
    if not (getattr(last, "tool_calls", None) or []):
        return "normal"
    if max_steps:
        ai_rounds = sum(1 for message in messages if getattr(message, "type", "") in {"ai", "AIMessage"})
        if ai_rounds >= max_steps:
            return "truncated"
    return "pending"


# ============================================================
# 4. 课程 02：自建 CallbackHandler 端到端
# ============================================================


async def run_callback(db: AsyncSession, payload: Day15CallbackRequest) -> dict[str, Any]:
    """跑一次真实 Agent，用自建 Handler 收集轨迹并还原成树。

    关键认知：LangSmith 没做什么魔法，它只是订阅了同一批回调事件。
    ``config={"callbacks": [...]}`` 就是所有平台通用的「注入探针」接口。
    """
    _require_langchain()
    from langchain.agents import create_agent

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day15-callback")
    tools = build_expense_tools()
    agent = create_agent(llm, tools=tools, system_prompt=EXPENSE_PROMPT)

    collector = CallbackCollector()
    handler = build_callback_handler(collector)
    tags = ["day15", "callback", "v1"]
    metadata = {"user_id": payload.user_id, "env": "dev", "source": "custom-handler"}
    started = time.perf_counter()

    result = await agent.ainvoke(
        {"messages": [("user", payload.question)]},
        config={"callbacks": [handler], "tags": tags, "metadata": metadata},
    )
    elapsed = _elapsed_ms(started)
    messages = result.get("messages") or []
    answer = _text(messages[-1]) if messages else ""

    llm_records = [record for record in collector.records if record["kind"] == "llm"]
    tool_records = [record for record in collector.records if record["kind"] == "tool"]
    input_tokens = sum(int((record.get("attributes") or {}).get("input_tokens") or 0) for record in llm_records)
    output_tokens = sum(int((record.get("attributes") or {}).get("output_tokens") or 0) for record in llm_records)

    return {
        "question": payload.question,
        "answer": answer,
        "tree": collector.tree(),
        "timeline": callback_timeline(collector.records, started),
        "events": [redact(record) for record in collector.records],
        "llm_calls": len(llm_records),
        "tool_calls": len(tool_records),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "tags": tags,
        "metadata": metadata,
        "model": cfg.model_name,
        "note": (
            "这棵树和 LangSmith 里看到的是同一份数据：parent_run_id 决定父子关系，"
            "on_llm_end 带来 token，on_tool_end 带来工具返回值。"
        ),
        "elapsed_ms": elapsed,
    }


# ============================================================
# 5. 课程 01/03：平台接入（LangSmith / Langfuse）
# ============================================================


def platform_status() -> dict[str, Any]:
    """探测两个平台的就绪状态：缺哪些环境变量、包装没装。

    只读环境变量，**不在请求里改**——追踪开关属于部署配置，
    由 .env 决定，避免一次点击就把数据送出境。
    """
    _load_env_once()
    langsmith_keys = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]
    langfuse_keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    langsmith_missing = [key for key in langsmith_keys if not os.environ.get(key)]
    langfuse_missing = [key for key in langfuse_keys if not os.environ.get(key)]
    if os.environ.get("LANGSMITH_TRACING", "").lower() not in {"true", "1", "yes"}:
        if "LANGSMITH_TRACING" not in langsmith_missing:
            langsmith_missing.append("LANGSMITH_TRACING（需为 true）")

    return {
        "langsmith": {
            "name": "LangSmith",
            "installed": _installed("langsmith"),
            "configured": not langsmith_missing,
            "missing": langsmith_missing,
            "project": os.environ.get("LANGSMITH_PROJECT", ""),
            "dashboard": "https://smith.langchain.com",
            "install_hint": "langsmith 已随 langchain 安装；只需在 agent_backend/.env 配三个变量",
        },
        "langfuse": {
            "name": "Langfuse",
            "installed": _installed("langfuse"),
            "configured": not langfuse_missing,
            "missing": langfuse_missing,
            "host": os.environ.get("LANGFUSE_HOST", ""),
            "dashboard": os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            "install_hint": "pip install langfuse  # 开源可自部署，数据不出内网",
        },
    }


def _latest_langsmith_run(project: str) -> dict[str, Any]:
    """尽力读出最近一条 run 的 id / 名称，读不到不影响主流程。

    LangSmith SDK 的查询接口各版本有差异，这里全部 try 包住——
    「拿不到链接」不该让一次成功的 Agent 运行变成失败。
    """
    try:
        from langsmith import Client

        client = Client()
        runs = list(client.list_runs(project_name=project, limit=1, is_root=True))
        if not runs:
            return {}
        run = runs[0]
        return {
            "run_id": str(getattr(run, "id", "")),
            "run_name": str(getattr(run, "name", "")),
            "start_time": str(getattr(run, "start_time", "")),
        }
    except Exception as exc:  # noqa: BLE001  网络/权限/SDK 版本都可能有差异
        return {"query_error": _short(exc, 160)}


def _platform_instructions(provider: str, status: dict[str, Any]) -> list[str]:
    """没配置时返回「怎么配」，而不是抛错——探针页面本身要能打开。"""
    info = status.get(provider) or {}
    if provider == "langsmith":
        return [
            "在 agent_backend/.env 添加：",
            "LANGSMITH_TRACING=true",
            "LANGSMITH_API_KEY=lsv2_pt_你的key   # smith.langchain.com 注册获取",
            "LANGSMITH_PROJECT=agent-course-day15",
            "重启后端（配置只在进程启动时读一次），再回来点运行。",
            f"当前缺少：{'、'.join(info.get('missing') or []) or '无'}",
        ]
    return [
        "pip install langfuse  # 可选依赖，本仓库没锁版本以免与现有栈冲突",
        "在 agent_backend/.env 添加：",
        "LANGFUSE_PUBLIC_KEY=pk-lf-...",
        "LANGFUSE_SECRET_KEY=sk-lf-...",
        "LANGFUSE_HOST=https://cloud.langfuse.com   # 自部署填内网地址",
        "重启后端后重试。",
        f"当前缺少：{'、'.join(info.get('missing') or []) or '无'}",
    ]


async def run_platform(db: AsyncSession, payload: Day15PlatformRequest) -> dict[str, Any]:
    """平台接入的两条路：LangSmith 零代码 / Langfuse Handler + 打分。

    没配置就返回配置说明（``configured=false``），不写任何环境变量、
    也只是在本地跑一次 Agent（不产生任何上报）。
    """
    _require_langchain()
    from langchain.agents import create_agent

    provider = payload.provider
    status = platform_status()
    info = status.get(provider) or {}
    tags = ["day15", provider, "v1"]
    metadata = {"user_id": payload.user_id, "env": "dev", "source": provider}

    if provider == "langfuse":
        _require_langfuse()

    if not info.get("configured"):
        return {
            "provider": provider,
            "configured": False,
            "missing": info.get("missing") or [],
            "answer": "",
            "trace_id": "",
            "tags": tags,
            "metadata": metadata,
            "dashboard": info.get("dashboard", ""),
            "instructions": _platform_instructions(provider, status),
            "notes": [
                "追踪开关属于部署配置，接口不会替你改环境变量。",
                "没配置时这里不会跑模型——省得你花了 token 才发现没上报。",
            ],
            "model": "",
            "elapsed_ms": 0,
        }

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name=f"day15-{provider}")
    tools = build_expense_tools()
    agent = create_agent(llm, tools=tools, system_prompt=EXPENSE_PROMPT)
    started = time.perf_counter()
    notes: list[str] = []

    if provider == "langsmith":
        # LangSmith 走环境变量自动上报，这里唯一要做的就是别挡路 + 打标签
        config: dict[str, Any] = {"tags": tags, "metadata": metadata}
        result = await agent.ainvoke({"messages": [("user", payload.question)]}, config=config)
        trace_id = ""
        run_info = _latest_langsmith_run(os.environ.get("LANGSMITH_PROJECT", "default"))
        if run_info:
            notes.append("已尝试读取最近一条 run：平台 UI 里按 tags=day15 筛选即可定位本次请求。")
        else:
            notes.append("已开启上报；查询最近 run 需要网络与有效 key，读不到不影响本次运行。")
    else:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        config = {"callbacks": [handler], "tags": tags, "metadata": metadata}
        result = await agent.ainvoke({"messages": [("user", payload.question)]}, config=config)
        try:
            handler.langfuse.flush()
        except Exception as exc:  # noqa: BLE001
            notes.append(f"flush 失败（短命进程最容易踩的坑 3）：{_short(exc, 120)}")
        trace_id = ""
        try:
            trace_id = str(handler.get_trace_id() or "")
        except Exception as exc:  # noqa: BLE001  SDK 版本差异
            notes.append(f"get_trace_id 不可用：{_short(exc, 120)}")
        if payload.feedback and trace_id:
            try:
                handler.langfuse.score(
                    trace_id=trace_id,
                    name="user-feedback",
                    value=1,
                    comment="Day 15 演示：用户点了赞",
                )
                notes.append("已把点赞 score 回填到这条 trace（Day 16 评估集的原料）。")
            except Exception as exc:  # noqa: BLE001
                notes.append(f"score 上报失败：{_short(exc, 120)}")
        run_info = {}

    messages = result.get("messages") or []
    return {
        "provider": provider,
        "configured": True,
        "missing": [],
        "answer": _text(messages[-1]) if messages else "",
        "trace_id": trace_id,
        "tags": tags,
        "metadata": metadata,
        "dashboard": info.get("dashboard", ""),
        "instructions": [],
        "notes": notes,
        "model": cfg.model_name,
        "elapsed_ms": _elapsed_ms(started),
        "run_info": run_info,
    }


# ============================================================
# 6. 课程 04：自建轻量 Trace（全流程可回放）
# ============================================================


async def demo_concurrency(runs: int = 3, steps: int = 2) -> dict[str, Any]:
    """并发不串线演示：N 个协程各自开 trace，检查父子关系是否互相污染。

    这是坑 4 的反面证据。``threading.local`` 版本在 asyncio 下会失败：
    协程都在同一线程里跑，thread-local 变量被共享，parent 会被隔壁请求覆盖。

    这个函数**必须是 async**：它会被 run_own_trace 在事件循环里 await，
    再套一层 ``asyncio.run`` 会直接抛 "cannot be called from a running event loop"。
    """
    tracer = SpanTracer(persist=False)

    async def one(worker: int) -> None:
        with tracer.span("agent-run", kind="agent", worker=worker):
            for step in range(steps):
                with tracer.span("step", kind="step", worker=worker, step=step) as record:
                    await asyncio.sleep(0)  # 主动让出，制造真实的协程交错
                    record["attributes"]["owner"] = f"worker-{worker}"

    await asyncio.gather(*(one(worker) for worker in range(runs)))

    by_trace: dict[str, list[dict[str, Any]]] = {}
    for record in tracer.records:
        by_trace.setdefault(record["trace_id"], []).append(record)

    traces: list[dict[str, Any]] = []
    ok = len(by_trace) == runs
    for trace_id, records in by_trace.items():
        roots = [record for record in records if record["parent_id"] is None]
        steps_rows = [record for record in records if record["kind"] == "step"]
        worker = roots[0]["attributes"].get("worker") if roots else None
        same_worker = all(record["attributes"].get("worker") == worker for record in steps_rows)
        parent_ok = all(
            record["parent_id"] == roots[0]["span_id"] for record in steps_rows
        ) if roots else False
        ok = ok and len(roots) == 1 and len(steps_rows) == steps and same_worker and parent_ok
        traces.append(
            {
                "trace_id": trace_id,
                "worker": worker,
                "spans": len(records),
                "roots": len(roots),
                "steps": len(steps_rows),
                "same_worker": same_worker,
                "parent_ok": parent_ok,
            }
        )

    return {
        "runs": runs,
        "steps_per_run": steps,
        "ok": ok,
        "traces": traces,
        "note": (
            "每个协程各得一条 trace，step 的 parent 都是自己那条 trace 的根——"
            "ContextVar 在 task 创建时复制上下文，所以并发不会串线。"
        ),
    }


async def run_own_trace(db: AsyncSession, payload: Day15OwnTraceRequest) -> dict[str, Any]:
    """手写循环 + 自建 span：把 Day 01 的 while 循环每一步都写进 JSONL。

    这就是课程 04 的完整版：``agent_run`` = 一次 while 循环，``call_llm`` = 每轮迭代，
    span 的嵌套弹栈 = trace 树的 parent/child。跑完顺手回放成树并聚合指标。
    """
    _require_langchain()
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day15-own-trace")
    tools = build_expense_tools()
    tool_map = {tool.name: tool for tool in tools}
    llm_with_tools = llm.bind_tools(tools)

    tracer = SpanTracer()
    started = time.perf_counter()
    state_messages: list[Any] = [SystemMessage(content=EXPENSE_PROMPT), HumanMessage(content=payload.question)]
    answer = ""
    stop_reason = "max_steps"
    rounds = 0

    with tracer.span(
        "agent-run",
        kind="agent",
        user_id=payload.user_id,
        question=payload.question,
        model=cfg.model_name,
    ) as root:
        for round_index in range(payload.max_steps):
            rounds = round_index + 1
            with tracer.span(
                "chat",
                kind="llm",
                model=cfg.model_name,
                round=round_index,
                n_msgs=len(state_messages),
            ) as record:
                response = await llm_with_tools.ainvoke(state_messages)
                usage = getattr(response, "usage_metadata", None) or {}
                record["attributes"]["input_tokens"] = int(usage.get("input_tokens") or 0)
                record["attributes"]["output_tokens"] = int(usage.get("output_tokens") or 0)
                record["attributes"]["finish_reason"] = str(
                    (getattr(response, "response_metadata", None) or {}).get("finish_reason") or ""
                )
            state_messages.append(response)
            calls = getattr(response, "tool_calls", None) or []
            if not calls:
                stop_reason = "normal"
                answer = _text(response)
                break

            for call in calls:
                name = str(call.get("name") or "")
                arguments = call.get("args") or {}
                with tracer.span(f"tool:{name}", kind="tool", tool=name, args=_jsonable(arguments)) as record:
                    tool = tool_map.get(name)
                    if tool is None:
                        output = f"未知工具：{name}"
                        record["status"] = "error"
                        record["attributes"]["error_type"] = "unknown_tool"
                    else:
                        try:
                            output = await tool.ainvoke(arguments)
                        except Exception as exc:  # noqa: BLE001  工具错误是结果，不是崩溃
                            output = f"工具报错：{exc}"
                            record["status"] = "error"
                            record["attributes"]["error"] = repr(exc)
                        record["attributes"]["output"] = _short(output, 2000)
                state_messages.append(
                    ToolMessage(content=str(output), tool_call_id=str(call.get("id") or ""))
                )

        if not answer:
            # 超步截断时模型停在 tool_calls 上：Day 05 会再发一次「无工具」请求强制收敛，
            # 这里只做标注，不额外烧一次 token——指标里 truncated 会如实计数。
            answer = "（达到 max_steps 被截断：本轮没有产出最终回答，对应 Day 05 的降级交付场景）"

        root["attributes"].update(
            termination=stop_reason,
            steps=rounds,
            answer_chars=len(answer),
        )

    records = tracer.records
    root_record = next(record for record in records if record["parent_id"] is None)
    return {
        "trace_id": root_record["trace_id"],
        "tree": tracer.tree(),
        "spans": [redact(record) for record in records],
        "metrics": summarize(records),
        "global_metrics": summarize(load_records(TRACE_FILE)),
        "concurrency": await demo_concurrency(payload.concurrency_runs),
        "answer": answer,
        "stop_reason": stop_reason,
        "model": cfg.model_name,
        "file": str(TRACE_FILE),
        "note": (
            "八字段落在每行 JSONL 里：parent_id 建树、duration_ms 找瓶颈、"
            "attributes 存 token 与工具原文。trace 数、p50/p95、成本全部从同一批记录算出来。"
        ),
        "elapsed_ms": _elapsed_ms(started),
    }


# ============================================================
# 7. 课程 05：LangGraph + MCP 全链路（trace_id 跨进程传播）
# ============================================================


def _expense_connection(trace_file: Path) -> dict[str, Any]:
    """stdio 连接配置：用 sys.executable 拉起真实脚本，并通过环境变量下发日志路径。"""
    return {
        "command": sys.executable,
        "args": [str(EXPENSE_SERVER_PATH)],
        "transport": "stdio",
        "env": {**os.environ, "DAY15_SERVER_SPANS": str(trace_file)},
    }


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """追加一行 JSON（自动建目录；脱敏由调用方决定，业务日志本身就是脱敏后的字段）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**payload}
    row.setdefault("ts", time.time())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_records(path: Path) -> list[dict[str, Any]]:
    """读 JSONL：文件不存在返回空，坏行跳过（日志坏了不该让查询接口 500）。"""
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def merge_timeline(client_rows: list[dict[str, Any]], server_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把「客户端视角」与「服务端视角」按时间合并——同一条 trace 的完整链路。"""
    rows: list[dict[str, Any]] = []
    for row in client_rows:
        rows.append(
            {
                "side": "client",
                "name": row.get("name") or row.get("type") or "event",
                "ts": float(row.get("ts") or 0),
                "duration_ms": float(row.get("duration_ms") or 0),
                "status": row.get("status") or "ok",
                "detail": _short(row.get("detail") or row.get("answer") or row.get("question") or "", 200),
            }
        )
    for row in server_rows:
        rows.append(
            {
                "side": "server",
                "name": row.get("tool") or "tool",
                "ts": float(row.get("ts") or 0),
                "duration_ms": float(row.get("dur_ms") or row.get("duration_ms") or 0),
                "status": row.get("status") or "ok",
                "detail": _short(row.get("args") or "", 200),
            }
        )
    rows.sort(key=lambda item: item["ts"])
    baseline = rows[0]["ts"] if rows else 0.0
    for row in rows:
        row["offset_ms"] = round((row["ts"] - baseline) * 1000)
    return rows


async def run_full_stack(db: AsyncSession, payload: Day15FullStackRequest) -> dict[str, Any]:
    """LangGraph 侧有平台 trace，MCP Server 侧没有——用 trace_id 把两边串起来。

    客户端做两件事：① 把 trace_id 钉进提示词，让它随工具参数过进程边界；
    ② 自己写业务事件（提问 / 工具调用 / 最终答案 / 终止原因）。
    Server 端按收到的 trace_id 记自己的 span，排查时 grep 一个 id 就能拼出完整链路。
    """
    _require_mcp()
    _require_langchain()
    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day15-fullstack")
    trace_id = uuid.uuid4().hex[:12]
    client = MultiServerMCPClient({"expense": _expense_connection(SERVER_FILE)})
    tools = await client.get_tools()
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=FULLSTACK_PROMPT.format(trace_id=trace_id),
    )

    started = time.perf_counter()
    append_jsonl(
        BUSINESS_FILE,
        {
            "trace_id": trace_id,
            "type": "question",
            "name": "question",
            "question": payload.question,
            "source": "client",
        },
    )

    final_state: dict[str, Any] = {}
    seen = 0
    tool_calls: list[dict[str, Any]] = []
    async for chunk in agent.astream(
        {"messages": [("user", payload.question)]},
        stream_mode="values",
        config={"recursion_limit": 16, "metadata": {"trace_id": trace_id}},
    ):
        final_state = chunk if isinstance(chunk, dict) else final_state
        messages = final_state.get("messages") or []
        # 逐条 diff：新出现的 AIMessage 里带 tool_calls 就落一条客户端事件（带真实时间戳）
        for message in messages[seen:]:
            for call in getattr(message, "tool_calls", None) or []:
                row = {"name": call.get("name", ""), "args": _jsonable(call.get("args") or {})}
                tool_calls.append(row)
                append_jsonl(
                    BUSINESS_FILE,
                    {
                        "trace_id": trace_id,
                        "type": "tool_call",
                        "name": row["name"],
                        "args": row["args"],
                        "source": "client",
                    },
                )
        seen = len(messages)

    messages = final_state.get("messages") or []
    answer = _text(messages[-1]) if messages else ""
    termination = classify_termination(messages, max_steps=8)
    elapsed = _elapsed_ms(started)

    append_jsonl(
        BUSINESS_FILE,
        {
            "trace_id": trace_id,
            "type": "answer",
            "name": "answer",
            "answer": answer,
            "termination": termination,
            "steps": len(messages),
            "duration_ms": elapsed,
            "source": "client",
        },
    )

    client_rows = [row for row in load_records(BUSINESS_FILE) if row.get("trace_id") == trace_id]
    server_rows = [row for row in load_records(SERVER_FILE) if row.get("trace_id") == trace_id]
    input_tokens = sum(
        int((getattr(message, "usage_metadata", None) or {}).get("input_tokens") or 0)
        for message in messages
    )
    output_tokens = sum(
        int((getattr(message, "usage_metadata", None) or {}).get("output_tokens") or 0)
        for message in messages
    )

    return {
        "trace_id": trace_id,
        "answer": answer,
        "terminated_normally": termination == "normal",
        "termination": termination,
        "steps": len(messages),
        "tool_calls": tool_calls,
        "business": next((row for row in client_rows if row.get("type") == "answer"), None),
        "server_spans": server_rows,
        "merged": merge_timeline(client_rows, server_rows),
        "matched": bool(server_rows),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "files": {
            "business": str(BUSINESS_FILE),
            "server": str(SERVER_FILE),
            "traces": str(TRACE_FILE),
        },
        "notes": [
            "客户端与 Server 各写各的 JSONL，唯一共享的是 trace_id。",
            "matched=false 通常意味着模型没把 trace_id 传给工具（提示词没钉住），不是链路设计的问题。",
            "Server 端错误也要带 trace_id 记录，否则『工具返回了错误』这句话永远定位不到现场。",
        ],
        "model": cfg.model_name,
        "elapsed_ms": _elapsed_ms(started),
    }


# ============================================================
# 8. 查询与反馈：把 trace 变成可看的指标（模板 metrics 的落地）
# ============================================================


def feedback_summary() -> dict[str, Any]:
    """用户反馈聚合：点赞 / 点踩 / 点赞率（Day 16 评估集的第一批真值标签）。"""
    rows = load_records(FEEDBACK_FILE)
    likes = sum(1 for row in rows if int(row.get("value") or 0) > 0)
    dislikes = sum(1 for row in rows if int(row.get("value") or 0) < 0)
    total = likes + dislikes
    return {
        "total": len(rows),
        "likes": likes,
        "dislikes": dislikes,
        "like_rate": round(likes / total, 4) if total else 0.0,
        "recent": rows[-5:],
    }


def record_feedback(payload: Day15FeedbackRequest) -> dict[str, Any]:
    """把一条用户反馈落到 trace 上（自建版 score）。"""
    row = {
        "trace_id": payload.trace_id,
        "value": payload.value,
        "comment": payload.comment,
        "name": "user-feedback",
    }
    append_jsonl(FEEDBACK_FILE, row)
    return {
        "trace_id": payload.trace_id,
        "value": payload.value,
        "comment": payload.comment,
        "summary": feedback_summary(),
    }


def list_traces(limit: int = 5) -> dict[str, Any]:
    """回放最近 N 条自建 trace，并给出全局指标（前端「指标看板」的数据源）。"""
    records = load_records(TRACE_FILE)
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_trace.setdefault(str(record.get("trace_id")), []).append(record)

    # 按根 span 的开始时间倒序取最近几条
    def root_start(rows: list[dict[str, Any]]) -> float:
        root = next((row for row in rows if row.get("parent_id") is None), None)
        return float((root or {}).get("start_ts") or 0)

    trace_ids = sorted(by_trace, key=lambda tid: root_start(by_trace[tid]), reverse=True)[: max(1, limit)]
    traces: list[dict[str, Any]] = []
    for trace_id in trace_ids:
        rows = by_trace[trace_id]
        root = next((row for row in rows if row.get("parent_id") is None), {})
        traces.append(
            {
                "trace_id": trace_id,
                "question": (root.get("attributes") or {}).get("question", ""),
                "termination": (root.get("attributes") or {}).get("termination", "unknown"),
                "started_at": root.get("start_ts"),
                "tree": span_tree(rows),
                "metrics": summarize(rows),
            }
        )

    return {
        "traces": traces,
        "metrics": summarize(records),
        "feedback": feedback_summary(),
        "files": {"traces": str(TRACE_FILE), "feedback": str(FEEDBACK_FILE)},
    }


# ============================================================
# 9. 概念表
# ============================================================


async def mapping() -> dict[str, Any]:
    """Day 15 静态概念数据：失败形态 / 三大件 / 两条路线 / 八字段 / 指标定义 / 五个坑 / 速查表。"""
    return {
        "failure_modes": FAILURE_MODES,
        "pillars": PILLARS,
        "routes": ROUTES,
        "trace_fields": TRACE_FIELDS,
        "metric_defs": METRIC_DEFS,
        "pitfalls": PITFALLS,
        "cheatsheet": CHEATSHEET,
        "checklist": CHECKLIST,
        "presets": PRESETS,
        "platform": platform_status(),
        "files": {
            "traces": str(TRACE_FILE),
            "business": str(BUSINESS_FILE),
            "server": str(SERVER_FILE),
            "feedback": str(FEEDBACK_FILE),
        },
        "versions": {
            "langsmith": _version("langsmith"),
            "langfuse": _version("langfuse"),
            "langchain": _version("langchain"),
            "langgraph": _version("langgraph"),
            "mcp": _version("mcp"),
        },
        "install_hint": INSTALL_HINT,
    }
