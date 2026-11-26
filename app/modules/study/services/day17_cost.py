"""Day 17 成本与延迟优化：同样的正确率，便宜一半、快一半。

课程 day17 的六个脚本在这里收成一个模块：

    成本账本        → ``simulate_run`` / ``ledger_demo``   （01：每步 in/out token、延迟、累计成本、放大率）
    模型分级路由    → ``router_demo``                       （02：规则 → 小模型分类 → 三信号升级）
    上下文压缩      → ``compress_demo``                     （03：静态瘦身 + 工具结果剪裁 + 预算触发摘要）
    缓存三层        → ``cache_demo``                        （04：只读白名单 / 实体校验 / 用户隔离 / 前缀哈希）
    并行与流式      → ``parallel_demo``                     （05：Semaphore + 超时 + 单点失败不炸整批）
    预算熔断 + A/B  → ``budget_demo`` / ``ab_demo``          （06：三闸熔断 + 降级交付 + 接回 Day 16 门禁）

三条贯穿全篇的结论：

1. **Agent 的成本不是「单价 × 次数」，而是「历史重放」**：一次 N 步的 run 把 system+user
   重新发送了 N+1 次，输入 token 近似 O(N²)。所以第一优先级永远是「少发 token」，
   而不是「换个便宜模型」——后者的收益最小、对正确率的伤害最大；
2. **延迟优化的第一刀是「把串行等待并行化 + 把输出变短」**，不是换更快的模型；流式只降首字
   延迟（TTFT），不降总延迟，别把它当性能优化；
3. **缓存收益最直接、风险也最贵**：省下的钱永远小于一次错答的赔付。所以三个安全阀
   （写操作不进缓存 / 语义命中必须实体一致 / 默认按用户隔离）在代码里写死，不靠自觉。

与课程示例的工程差异（都是踩过之后改的）：

- 课程 01 的账本在 ``call_model`` 里用 ``llm.model_name`` 取价格，模型名不在表里会 KeyError；
  本模块 ``price_of`` 按名字模糊匹配并给 unknown 兜底价，同时把价格表返回给前端（避免"抄死数字"）；
- 课程 03 的 ``compress_if_needed`` 用 ``convo[2:-keep_recent]`` 硬切下标，消息形态一变就会把
  ``tool_calls`` 和配对的 ``ToolMessage`` 切开；本模块**按「轮」切**（user → … → assistant 为一轮），
  并提供 ``check_tool_pairing`` 把孤儿 tool 消息显式查出来（课程 Day 06 踩过的 400 坑）；
- 课程 04 的 ``trim_tool_output`` 用正则抠 JSON 字段，遇到嵌套/转义就抠歪；本模块先 ``json.loads``，
  失败才退化成"头部截断 + 正则找关键字段"；
- 课程 05 的并行用 ``random.uniform`` 假装 IO，无法断言；本模块的 ``parallel_duration`` 是**纯函数**
  （按并发上限分批取 max），异步实测只负责演示时间轴；
- 课程 06 的 A/B 表是写死的占位数字（86% → 85%、$0.048 → $0.009）；本模块**按同一套杠杆参数
  现算成本与延迟**，并明确说明「正确率这一列必须由 Day 16 的门禁实测，不许照抄课程数字」——
  优化报告里最危险的就是把别人的实测值当自己的结论。

tiktoken / langchain / langgraph 全部延迟导入：缺依赖时接口返回可读的 400，而不是让整个应用起不来。
"""

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import (
    Day17AbRequest,
    Day17BudgetRequest,
    Day17CacheRequest,
    Day17CompressRequest,
    Day17LedgerRequest,
    Day17ParallelRequest,
    Day17RouterRequest,
)
from app.modules.study.services.day09_langchain import build_chat_model
from app.modules.study.services.llm import resolve_model
from app.modules.study.services.rag import cosine, hash_embed

INSTALL_HINT = "pip install langchain langchain-openai langgraph tiktoken"

# ============================================================
# 运行时文件目录：var/ 已在 .gitignore；账本与 A/B 报告跨重启保留
# ============================================================

VAR_DIR = Path(__file__).resolve().parents[4] / "var" / "day17"
LEDGER_FILE = VAR_DIR / "last_ledger.json"
AB_FILE = VAR_DIR / "ab_report.json"

# ============================================================
# 静态概念数据（全部来自课程原文）
# ============================================================

# 1.1 单次调用 vs 一次 run
COST_STRUCTURE: list[dict[str, str]] = [
    {
        "scope": "单次调用",
        "formula": "输入 token × 输入单价 + 输出 token × 输出单价",
        "note": "大多数人第一次估成本都算这个——对 Agent 严重低估",
    },
    {
        "scope": "一次 run",
        "formula": "Σ(N 次调用的输入 + 输出)；每步都要把完整历史重发一遍",
        "note": "输入 token 累加，因此 N 步 run 的总输入 ≈ O(N²)，不是 O(N)",
    },
    {
        "scope": "历史重放",
        "formula": "第 k 步输入 = system + user + 前 k-1 步的助手输出与工具结果",
        "note": "6 步 run 能把「一次性问答」的输入放大 15 倍以上，这是最大的浪费项",
    },
]

# 1.3 延迟的结构
LATENCY_BREAKDOWN: list[dict[str, str]] = [
    {
        "segment": "prefill",
        "what": "把输入 token 编码成 KV",
        "scales_with": "与输入长度近似线性（长上下文时是大头）",
        "lever": "前缀缓存 + 上下文压缩",
    },
    {
        "segment": "decode",
        "what": "逐 token 生成",
        "scales_with": "与输出长度线性",
        "lever": "让模型少说废话（同时省钱和省时间）",
    },
    {
        "segment": "工具执行",
        "what": "外部 API / DB 的 IO",
        "scales_with": "Σ(工具耗时)，默认串行",
        "lever": "只读工具并行 + Semaphore + 超时",
    },
    {
        "segment": "编排开销",
        "what": "图调度、序列化、网络往返",
        "scales_with": "步数 × 每次往返（MCP 远程时更明显）",
        "lever": "减少步数、合并调用",
    },
]

# 1.4 四个杠杆与优先级
LEVERS: list[dict[str, str]] = [
    {
        "priority": "1",
        "lever": "上下文压缩（少发 token）",
        "gain": "成本 -50%~-70%",
        "risk": "裁多了能力下降（约束锚点、tool 配对不能切）",
        "source": "Day 06 短期记忆 / Day 08 上下文工程",
    },
    {
        "priority": "2",
        "lever": "并行 / 流式（少等待）",
        "gain": "延迟 -40%~-60%",
        "risk": "限流、依赖错乱、写操作重试",
        "source": "Day 10 Send 扇出 / Day 13 流式",
    },
    {
        "priority": "3",
        "lever": "缓存（少重算）",
        "gain": "成本 -20%~-40%",
        "risk": "错答 / 越权（最危险：省的钱不够赔一次）",
        "source": "Day 07 RAG / Day 15 trace",
    },
    {
        "priority": "4",
        "lever": "模型分级（少花钱的调用）",
        "gain": "成本 -30%~-60%",
        "risk": "小模型能力不足（必须由 Day 16 门禁兜住）",
        "source": "Day 02 结构化输出 / Day 03 FC",
    },
]

# 2.4 缓存三层
CACHE_LAYERS: list[dict[str, str]] = [
    {
        "layer": "L1 工具结果缓存",
        "what": "只读工具的返回（查订单、查 FAQ）",
        "hit_when": "工具名 + 规范化参数完全一致",
        "risk": "数据过期",
        "valve": "只读白名单（写操作永不缓存）+ TTL",
    },
    {
        "layer": "L2 语义缓存",
        "what": "相似问题的答案",
        "hit_when": "embedding 相似度 ≥ 阈值",
        "risk": "错答、越权（45 元 vs 450 元）",
        "valve": "实体集合必须一致 + 默认按用户隔离 + 版本可清空",
    },
    {
        "layer": "L3 前缀缓存",
        "what": "provider 侧的 KV 前缀",
        "hit_when": "请求开头 N token 完全一致（OpenAI ≥1024）",
        "risk": "无（但会被时间戳 / 随机 ID 打穿）",
        "valve": "稳定内容放最前且逐字节不变 + 前缀哈希进监控",
    },
]

# 第 4 节：常见的坑（课程标题写「五个」，正文实际列了 6 条，这里全部保留）
PITFALLS: list[dict[str, str]] = [
    {
        "no": "1",
        "title": "先优化，后评估",
        "symptom": "没建基线就改 prompt / 换模型 / 加缓存，无法区分「省了钱」和「变笨了」",
        "fix": "建基线 → 一次只改一个变量 → 跑 Day 16 评估 → 达标才留下",
    },
    {
        "no": "2",
        "title": "只盯单价，不看结构",
        "symptom": "看到「输入 95% 是重复发送」之前，换模型是唯一能想到的手段",
        "fix": "先解剖账单：算放大率、找重复发送占比，再决定动哪一刀",
    },
    {
        "no": "3",
        "title": "缓存没有安全阀",
        "symptom": "语义缓存命中「退款 45 元」去回答「退款 450 元」——资损级错答",
        "fix": "只读白名单 + 实体一致 + 用户隔离，三缺一即事故（写死在代码里）",
    },
    {
        "no": "4",
        "title": "为了省钱破坏前缀缓存",
        "symptom": "system 最前面拼带秒的时间戳 / uuid4 / 顺序不稳的字段，命中率归零",
        "fix": "稳定前缀 + 变动后缀；前缀哈希进监控（它不会报错，只会多付钱）",
    },
    {
        "no": "5",
        "title": "没有预算熔断",
        "symptom": "模型陷入重复调用时，账单是唯一会「无限增长」的指标",
        "fix": "max_cost / max_steps / max_wall 三闸同时在；熔断后走降级交付而不是报错",
    },
    {
        "no": "6",
        "title": "把流式当成延迟优化",
        "symptom": "后端 8 秒才出第一个 token，接上流式照样卡；总延迟一秒没少",
        "fix": "先砍串行步数与输出长度，再谈流式（它只降 TTFT）",
    },
]

# 第 6 节：速查表
CHEATSHEET: list[dict[str, str]] = [
    {"topic": "先搞清楚钱花哪了", "content": "账本记录每步 in/out token + 延迟 + 成本", "key": "算放大率，找重复发送占比"},
    {"topic": "砍 token（第一优先）", "content": "静态 prompt 瘦身 + 工具结果剪裁 + 预算触发摘要", "key": "按轮切，别切 tool_call 配对"},
    {"topic": "砍调用（省钱）", "content": "规则路由 → 小模型分类 → 失败升级", "key": "路由本身必须便宜；升级要有上限"},
    {"topic": "砍重复（缓存）", "content": "L1 只读工具 / L2 语义 / L3 前缀", "key": "白名单 + 实体校验 + 用户隔离"},
    {"topic": "砍等待", "content": "并行只读工具 + Semaphore + 超时", "key": "有依赖的不并行；写操作不并行"},
    {"topic": "改善感知", "content": "流式输出", "key": "只降 TTFT，不降总延迟"},
    {"topic": "可控成本", "content": "max_cost / max_steps / max_wall 三闸", "key": "熔断后降级交付，不报错"},
    {"topic": "验证没变坏", "content": "Day 16 评估集 A/B", "key": "一次只改一个变量；正确率降幅 ≤2pt"},
    {"topic": "监控", "content": "每 thread 成本、缓存命中率、前缀哈希、p95", "key": "成本异常往往先于质量异常出现"},
    {"topic": "优化顺序", "content": "压缩 > 并行 > 缓存 > 分级", "key": "收益 / 风险比从高到低"},
]

CHECKLIST: list[str] = [
    "能说清「单次调用成本」和「一次 run 成本」的差别，并解释为什么是 O(N²)",
    "能给一次 run 记账，算出历史重放放大率与重复发送占比",
    "能按 prefill / decode / 工具 / 编排四段拆解延迟，并说出各段最有效的杠杆",
    "能写出规则路由（零成本、可解释），并知道为什么不能用主力模型去做路由",
    "能用结构化输出接住小模型的 JSON 错误，并设计三信号升级链与升级上限",
    "能做静态 prompt 瘦身与工具结果剪裁，并说明为什么不能按单条消息裁剪",
    "能实现三层缓存，并把三个安全阀写死在代码里（白名单 / 实体 / 用户隔离）",
    "能解释「45 元 vs 450 元」为什么任何 0.95 阈值都拦不住，以及实体校验怎么救",
    "能把串行工具改成并行，并给出无依赖 / 幂等 / 并发上限三个前提",
    "能给出三闸预算熔断 + 降级交付，并用 Day 16 门禁证明「没变坏」",
]

# 各实验的预设（前端一键填入）
PRESETS: dict[str, list[str]] = {
    "router": [
        "营业时间几点",
        "帮我对比一下这两个套餐的性价比",
        "我上个月买的耳机坏了，订单号 SO20261001，但我不确定是不是在保修期内，而且我想知道能不能直接换新而不是维修",
    ],
    "budget": [
        "我的订单到哪了？顺便确认金额，能不能用券，能不能开发票",
        "帮我查一下订单 SO20261115001 的物流，再算算我这单总共花多少钱",
    ],
    "cache": [
        "退款 45 元多久到账",
        "退款 450 元多久到账",
    ],
}

# ============================================================
# 价格表：$/1M token (输入, 输出)
# 真实项目请按自己的账号账单替换；unknown 一定给兜底价，别让 KeyError 打断课堂
# ============================================================

PRICE: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    "unknown": (1.00, 3.00),
}

PRICE_TABLE: list[dict[str, str]] = [
    {"model": "gpt-4o-mini", "input": "0.15", "output": "0.60", "tier": "小模型（分类/抽取/改写）"},
    {"model": "gpt-4o", "input": "2.50", "output": "10.00", "tier": "主力（推理 + 工具决策）"},
    {"model": "deepseek-chat", "input": "0.27", "output": "1.10", "tier": "主力（本仓库默认档）"},
    {"model": "deepseek-reasoner", "input": "0.55", "output": "2.19", "tier": "疑难兜底"},
    {"model": "unknown", "input": "1.00", "output": "3.00", "tier": "兜底价（表里查不到时）"},
]

# 三级路由的档位定义：真实项目换成你账号里的三个模型即可
TIER_MODEL: dict[str, dict[str, Any]] = {
    "small": {"model": "gpt-4o-mini", "role": "分类 / 抽取 / 改写", "temp": 0.0},
    "main": {"model": "gpt-4o", "role": "主推理 + 工具决策", "temp": 0.0},
    "big": {"model": "gpt-4o", "role": "疑难兜底（多约束 / 长文档）", "temp": 0.0},
}

# 预算三闸默认值（课程 06）：一次 run 最多 2 美分 / 6 步 / 20 秒
BUDGET_DEFAULTS = {
    "max_cost": 0.02,
    "max_steps": 6,
    "max_wall": 20.0,
    "degrade_policy": "摘掉工具 → 逼模型用已有信息总结 → 信息不足就转人工（不抛 500）",
}

# 延迟线性模型的常数（示意值：用于相对比较，不是 SLA。真实值用 live_probe 校准）
PREFILL_SEC_PER_K = 0.06
DECODE_SEC_PER_TOKEN = 0.02
OVERHEAD_SEC_PER_STEP = 0.05

# ============================================================
# 1. token 估算与账本
# ============================================================

_ENCODER_LOADED = False
_ENCODER: Any = None
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _get_encoder() -> Any:
    """惰性加载 tiktoken 编码器；拿不到（没装 / 词表下载失败）返回 None，由调用方退化估算。

    加载失败也缓存住（``_ENCODER_LOADED``）：网络不通时不要每次调用都重试下载词表。
    """
    global _ENCODER_LOADED, _ENCODER
    if _ENCODER_LOADED:
        return _ENCODER
    _ENCODER_LOADED = True
    try:
        import tiktoken

        _ENCODER = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001  没装 tiktoken / 词表下载失败都走启发式
        _ENCODER = None
    return _ENCODER


def ntok(text: str) -> int:
    """估算 token 数。

    优先用 tiktoken 的 ``cl100k_base``（与 OpenAI 系对齐）；不可用时退化成启发式：
    中文按 1 字 ≈ 1 token、其余按 4 字符 ≈ 1 token。**估算只用于相对比较**，
    真实账单请以 provider 返回的 usage 为准（``live_probe`` 就是用来校准这个偏差的）。
    """
    if not text:
        return 0
    encoder = _get_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:  # noqa: BLE001  编码器异常也不要中断整个请求
            pass
    if not text.strip():
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = max(0, len(text) - cjk)
    return cjk + max(1, round(other / 4))


def _field_of(message: Any, key: str) -> Any:
    """统一读取 dict 与对象属性（消息既可能是 dict，也可能是 LangChain 消息对象）。"""
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def _content_of(message: Any) -> str:
    """取消息正文（兼容 str 与「内容块 list」两种返回形态，与 Day 10 同款）。"""
    content = _field_of(message, "content")
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
    return "" if content is None else str(content)


def ntok_messages(messages: list[Any]) -> int:
    """估算一组消息的 token：正文 + 工具调用结构（``tool_calls`` 也占 token）。"""
    total = 0
    for message in messages or []:
        total += ntok(_content_of(message))
        calls = _field_of(message, "tool_calls") or []
        if calls:
            total += len(json.dumps(calls, ensure_ascii=False, default=str)) // 4
    return total


def price_of(model: str) -> tuple[float, float]:
    """按模型名模糊匹配价格，查不到给兜底价（课程版直接 ``PRICE[model]`` 会 KeyError）。"""
    name = (model or "").lower()
    if "mini" in name:
        return PRICE["gpt-4o-mini"]
    if "reasoner" in name:
        return PRICE["deepseek-reasoner"]
    if "deepseek" in name:
        return PRICE["deepseek-chat"]
    if "4o" in name or "gpt-4" in name or "gpt-5" in name:
        return PRICE["gpt-4o"]
    return PRICE["unknown"]


def step_cost(model: str, in_tok: int, out_tok: int) -> float:
    """单步费用（美元）：输入 / 输出分开计价。"""
    price_in, price_out = price_of(model)
    return round(in_tok / 1e6 * price_in + out_tok / 1e6 * price_out, 6)


@dataclass
class LedgerStep:
    """账本的一行：哪一步、用哪个模型、进出 token、耗时、费用。"""

    idx: int
    model: str
    in_tok: int
    out_tok: int
    latency: float
    cost: float
    note: str = ""

    def row(self) -> dict[str, Any]:
        """转成前端表格行。"""
        return {
            "idx": self.idx,
            "model": self.model,
            "in_tok": self.in_tok,
            "out_tok": self.out_tok,
            "latency": round(self.latency, 2),
            "cost": self.cost,
            "note": self.note,
        }


@dataclass
class Ledger:
    """一次 run 的账本：只做累加与聚合，不关心 token 从哪来。"""

    steps: list[LedgerStep] = field(default_factory=list)

    def add(self, model: str, in_tok: int, out_tok: int, latency: float, note: str = "") -> LedgerStep:
        """记一步，返回该步记录（调用方可直接读 ``.cost`` 做预算判断）。"""
        item = LedgerStep(
            idx=len(self.steps) + 1,
            model=model,
            in_tok=int(in_tok),
            out_tok=int(out_tok),
            latency=round(float(latency), 3),
            cost=step_cost(model, in_tok, out_tok),
            note=note,
        )
        self.steps.append(item)
        return item

    @property
    def total_cost(self) -> float:
        return round(sum(step.cost for step in self.steps), 6)

    @property
    def total_in(self) -> int:
        return sum(step.in_tok for step in self.steps)

    @property
    def total_out(self) -> int:
        return sum(step.out_tok for step in self.steps)

    @property
    def total_latency(self) -> float:
        return round(sum(step.latency for step in self.steps), 2)

    def amplification(self, baseline_in: int) -> float:
        """历史重放放大率：本次总输入 / 一次性问答的输入。"""
        if not baseline_in:
            return 0.0
        return round(self.total_in / baseline_in, 2)

    def waste_ratio(self, baseline_in: int) -> float:
        """「重复发送」的浪费占比（1 - 基线 / 总输入）。"""
        if not baseline_in or not self.total_in:
            return 0.0
        return round(max(0.0, 1 - baseline_in / self.total_in), 4)

    def summary(self, baseline_in: int = 0) -> dict[str, Any]:
        """聚合结果，直接给前端指标卡。"""
        return {
            "steps": len(self.steps),
            "in_tok": self.total_in,
            "out_tok": self.total_out,
            "cost": self.total_cost,
            "latency": self.total_latency,
            "baseline_in": baseline_in,
            "amplification": self.amplification(baseline_in),
            "waste_ratio": self.waste_ratio(baseline_in),
        }


def simulate_run(
    *,
    model: str = "deepseek-chat",
    system_tokens: int = 700,
    question_tokens: int = 200,
    tool_tokens: int = 800,
    steps: int = 5,
    first_output: int = 200,
    decision_output: int = 150,
    final_output: int = 500,
    prefill_ms_per_1k: float = 60.0,
    decode_ms_per_token: float = 18.0,
    tool_ms: float = 250.0,
) -> Ledger:
    """模拟一次 N 步 run 的账本（课程 1.2 那张表的通用版）。

    关键在 ``acc``：每步的输入 = 基线 + 之前所有助手输出与工具结果之和。
    这就是 O(N²) 的来源——不要写成「每步输入都等于基线」。
    """
    ledger = Ledger()
    base = system_tokens + question_tokens
    acc = 0
    for index in range(1, steps + 1):
        in_tok = base + acc
        out_tok = first_output if index == 1 else decision_output
        latency = in_tok / 1000 * prefill_ms_per_1k + out_tok * decode_ms_per_token + tool_ms
        ledger.add(model, in_tok, out_tok, latency / 1000, note=f"决策{index}")
        acc += out_tok + tool_tokens
    final_in = base + acc
    final_latency = final_in / 1000 * prefill_ms_per_1k + final_output * decode_ms_per_token
    ledger.add(model, final_in, final_output, final_latency / 1000, note="汇总")
    return ledger


def _dump(path: Path, payload: dict[str, Any]) -> Path:
    """把报告落盘（var/ 已 gitignore，跨重启可回看）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ============================================================
# 2. 杠杆一：模型分级路由
# ============================================================


class Tier(str, Enum):
    """三档模型：SMALL 便宜档 / MAIN 主力 / BIG 兜底。"""

    SMALL = "small"
    MAIN = "main"
    BIG = "big"


# 规则关键词：故意保持「短、可解释、零成本」——路由本身贵就是负优化
SIMPLE_HINTS = ("几点", "在哪", "是什么", "多少钱", "怎么查", "营业时间")
HARD_HINTS = ("对比", "为什么", "规划", "分析一下", "如果", "权衡", "方案")
# 转折 / 追加词：出现两个以上就说明「一句话提了好几个要求」，便宜档容易只答一半
MULTI_HINTS = ("但", "而且", "顺便", "另外", "还要", "不确定")


def route_by_rules(question: str, history_rounds: int = 0) -> Tier:
    """规则路由（零成本、可解释，先做这个）。

    顺序很重要：先判「难」（宁可高估，别把难题丢给小模型），再判「简单」，
    最后按长度兜底——短问题默认小模型，长问题默认主力。

    **对课程做了一处修正**：课程只用 ``len(q) > 60`` 判长短，但它自己给的第三个例子
    （「我上个月买的耳机坏了…而且我想知道能不能直接换新」）只有 58 个字，按原规则会被
    丢给便宜档。这里补一条「多约束」判据：分句 ≥2 个或命中多个转折/追加词也算难题。
    这也正是「规则要用真实数据校准」的现场——别对着想当然的阈值调参。
    """
    text = (question or "").strip()
    if any(key in text for key in HARD_HINTS) or history_rounds >= 4:
        return Tier.MAIN
    if _is_multi_constraint(text):
        return Tier.MAIN
    if len(text) <= 15 and any(key in text for key in SIMPLE_HINTS):
        return Tier.SMALL
    return Tier.MAIN if len(text) > 60 else Tier.SMALL


def _clause_count(text: str) -> int:
    """粗算分句数：逗号 / 分号 / 转折词各算一处（中文没有可靠分句器，够用即可）。"""
    return len(re.findall(r"[，,；;]", text)) + sum(1 for key in MULTI_HINTS if key in text)


def _is_multi_constraint(text: str) -> bool:
    """多约束提问：多个分句或多个「但 / 而且 / 顺便」= 需要主力模型同时满足多个条件。"""
    return _clause_count(text) >= 2


def explain_route(question: str, history_rounds: int = 0) -> dict[str, Any]:
    """给出路由结论 + 触发原因（课堂上看「为什么被判成 main」比看结论有用）。"""
    text = (question or "").strip()
    tier = route_by_rules(question, history_rounds)
    reasons: list[str] = []
    hit_hard = [key for key in HARD_HINTS if key in text]
    hit_simple = [key for key in SIMPLE_HINTS if key in text]
    if hit_hard:
        reasons.append(f"命中困难词 {hit_hard} → main")
    if history_rounds >= 4:
        reasons.append(f"对话轮次 {history_rounds} ≥ 4（上下文已长，小模型容易断片）→ main")
    if not reasons and tier is Tier.MAIN and _is_multi_constraint(text):
        reasons.append(f"多约束提问（分句/转折共 {_clause_count(text)} 处）→ main")
    if not reasons:
        if tier is Tier.SMALL:
            if hit_simple:
                reasons.append(f"短问题（{len(text)} 字）+ 命中简单词 {hit_simple} → small")
            else:
                reasons.append(f"短问题（{len(text)} ≤ 60 字）→ small")
        else:
            reasons.append(f"长问题（{len(text)} > 60 字）→ main")
    return {"tier": tier.value, "reason": "；".join(reasons)}


class Escalator:
    """升级策略：什么时候承认「便宜档不够用」。

    三信号：结构化输出校验失败 / 工具连续报错 ≥2 次 / 答案为空或含糊。
    升级到 BIG 就封顶——再失败应该走降级交付 + 转人工，而不是无限重试（Day 05）。
    """

    def __init__(self) -> None:
        self.tool_errors = 0

    def note_tool_error(self) -> None:
        """记录一次工具失败（连续失败会触发升级）。"""
        self.tool_errors += 1

    def next_tier(
        self, current: Tier, *, schema_error: bool = False, empty_answer: bool = False
    ) -> Tier:
        """按信号决定下一档；没有信号就原地不动。"""
        if schema_error or empty_answer or self.tool_errors >= 2:
            return {Tier.SMALL: Tier.MAIN, Tier.MAIN: Tier.BIG, Tier.BIG: Tier.BIG}[current]
        return current


def simulate_escalation(signals: list[str], start: Tier = Tier.SMALL) -> list[dict[str, Any]]:
    """把一串失败信号喂给升级链，返回每一跳的档位与原因（纯逻辑，可断言）。

    信号取值：``schema_error`` / ``tool_error`` / ``empty_answer`` / ``ok``。
    ``ok`` 会清零工具错误计数（模型自己纠正了，就不该继续升级）。
    """
    escalator = Escalator()
    tier = start
    chain: list[dict[str, Any]] = [{"hop": 0, "tier": tier.value, "trigger": "初始路由"}]
    for index, signal in enumerate(signals, start=1):
        trigger = ""
        if signal == "tool_error":
            escalator.note_tool_error()
            trigger = f"工具报错（累计 {escalator.tool_errors} 次）"
        elif signal == "schema_error":
            trigger = "结构化输出校验失败"
        elif signal == "empty_answer":
            trigger = "答案为空 / 含糊"
        elif signal == "ok":
            trigger = "工具成功，模型自我纠正"
            # 先清零再判档：成功之后不该继续背着之前的错误计数（否则一次 ok 也会被升级）
            escalator.tool_errors = 0
        next_tier = escalator.next_tier(
            tier,
            schema_error=signal == "schema_error",
            empty_answer=signal == "empty_answer",
        )
        upgraded = next_tier is not tier
        chain.append(
            {
                "hop": index,
                "tier": next_tier.value,
                "trigger": (trigger or "无升级信号") + ("（升级）" if upgraded else "（保持）"),
            }
        )
        tier = next_tier
    return chain


def mix_saving(
    counts: dict[str, int], *, tokens_in: int = 1200, tokens_out: int = 150
) -> dict[str, Any]:
    """分级路由的钱包效应：同样的调用量，全走主力 vs 按档位分流。

    ``counts`` 形如 ``{"small": 4, "main": 5, "big": 1}``（共 10 次调用）。
    """
    total_calls = sum(counts.values()) or 1
    mixed = 0.0
    detail: list[dict[str, Any]] = []
    for tier, count in counts.items():
        model = TIER_MODEL[tier]["model"]
        per_call = step_cost(model, tokens_in, tokens_out)
        mixed += per_call * count
        detail.append(
            {
                "tier": tier,
                "model": model,
                "calls": count,
                "per_call": round(per_call, 6),
                "subtotal": round(per_call * count, 6),
            }
        )
    main_model = TIER_MODEL["main"]["model"]
    all_main = step_cost(main_model, tokens_in, tokens_out) * total_calls
    small_per_call = step_cost(TIER_MODEL["small"]["model"], tokens_in, tokens_out)
    main_per_call = step_cost(main_model, tokens_in, tokens_out)
    return {
        "counts": counts,
        "total_calls": total_calls,
        "tokens_per_call": {"in": tokens_in, "out": tokens_out},
        "detail": detail,
        "mixed_cost": round(mixed, 6),
        "all_main_cost": round(all_main, 6),
        "saving_ratio": round(1 - mixed / all_main, 4) if all_main else 0.0,
        "per_call_saving": round(1 - small_per_call / main_per_call, 4) if main_per_call else 0.0,
    }


# ============================================================
# 3. 杠杆二：上下文压缩
# ============================================================

# 课程 03 的「胖 prompt」：一半 token 在礼貌用语和重复强调上
PROMPT_FAT = """你是一个非常专业、非常有耐心、非常乐于助人的电商客服助手。
你的职责是帮助用户解决各种问题，包括但不限于订单查询、物流查询、退换货、
发票开具、优惠券使用、会员权益等等。请你一定要保持礼貌，使用敬语，
每次回答都要先问候用户，然后再回答问题，最后询问是否还有其他可以帮助的地方。
请注意：不要编造信息。不要编造信息。绝对不要编造任何你不确定的信息。
如果有工具可以查询，请优先使用工具查询，而不是自己猜测。
回答要简洁。回答不要太长。回答要专业。回答要口语化。
以下是几个示例：
用户：你好 → 助手：您好，很高兴为您服务，请问有什么可以帮您？
用户：谢谢 → 助手：不客气，随时为您服务。
用户：在吗 → 助手：在的，请问有什么可以帮您？
用户：订单在哪 → 助手：我帮您查一下订单，请提供订单号。
用户：发票 → 助手：发票需要在订单完成后申请，我帮您看下入口。
用户：退货运费 → 助手：退货运费由责任方承担，质量问题我们承担。
"""

# 瘦身版：同样的语义密度，去掉寒暄、示例与重复强调（每次调用都省，累积到 N 步就是 N 倍）
PROMPT_LEAN = """你是电商客服助手。
规则：
1. 事实必须来自工具返回，不确定就调用工具；工具也查不到就说"查不到"并转人工。
2. 回答 ≤3 句，不复述用户问题，不寒暄，不加"请问还有其他可以帮您"。
3. 涉及金额、时效、政策时，必须引用工具返回的原始字段。
"""

# 工具结果剪裁时优先保留的字段（真实项目按业务字段表配置）
KEEP_KEYS = ("status", "amount", "eta", "order_id", "error", "code")

# 约束锚点：压缩时绝不能丢的规则（Day 04 确认门 / Day 05 预算）
CONSTRAINT_ANCHOR = "退款/改地址必须用户确认；单笔金额上限 500 元；查不到就说查不到并转人工。"


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    """把嵌套 JSON 摊平成 ``key -> 标量``，用于按字段名挑关键值。"""
    flat: dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                flat.update(_flatten(value, prefix))
            else:
                flat.setdefault(str(key), value)
    elif isinstance(data, list):
        for value in data:
            flat.update(_flatten(value, prefix))
    return flat


def trim_tool_output(raw: str, max_chars: int = 400) -> dict[str, Any]:
    """剪裁工具返回：能解析成 JSON 就只留关键字段 + 头部摘要，否则退化成截断。

    原则：**宁可留字段名和值，不要留整页原文**。返回结构里带前后 token，
    方便前端直接展示「这一刀省了多少」。
    """
    raw = raw or ""
    before = ntok(raw)
    text = raw
    kept: list[str] = []
    if raw.strip().startswith(("{", "[")):
        try:
            flat = _flatten(json.loads(raw))
            kept = [f"{key}={flat[key]}" for key in KEEP_KEYS if key in flat]
        except (json.JSONDecodeError, TypeError):
            kept = []
        if kept:
            text = "；".join(kept)[:max_chars]
    if not kept:
        if len(raw) <= max_chars:
            text = raw
        else:
            head = raw[:max_chars]
            found = [
                f"{key}={match}"
                for key in KEEP_KEYS
                for match in re.findall(rf'"{key}"\s*:\s*"?([^",}}]+)', raw)[:1]
            ]
            text = head
            if found:
                text += "\n关键字段：" + "；".join(found)
            text += f"\n[已截断 {len(raw) - max_chars} 字]"
    after = ntok(text)
    return {
        "text": text,
        "before_tokens": before,
        "after_tokens": after,
        "saved_tokens": max(0, before - after),
        "saved_ratio": round(1 - after / before, 4) if before else 0.0,
        "kept_fields": kept,
        "truncated_chars": max(0, len(raw) - len(text)),
    }


def split_rounds(messages: list[Any]) -> list[list[Any]]:
    """按「轮」切分消息：一条 user 消息开启一轮，直到下一条 user。

    为什么必须按轮切：``assistant(tool_calls)`` 和配对的 ``tool`` 消息一旦被切开，
    provider 会直接 400（Day 06 踩过的坑）。
    """
    rounds: list[list[Any]] = []
    current: list[Any] = []
    for message in messages or []:
        if _field_of(message, "role") == "user" and current:
            rounds.append(current)
            current = [message]
        else:
            current.append(message)
    if current:
        rounds.append(current)
    return rounds


def check_tool_pairing(messages: list[Any]) -> list[str]:
    """检查 tool_calls ↔ tool 消息配对，返回问题清单（空列表 = 健康）。

    - 助手说有 tool_calls，后面必须有对应 ``tool_call_id`` 的 tool 消息；
    - tool 消息不能凭空出现（没有对应的 tool_calls）。
    """
    issues: list[str] = []
    pending: dict[str, int] = {}
    for index, message in enumerate(messages or []):
        for call in _field_of(message, "tool_calls") or []:
            call_id = str((call or {}).get("id") or f"unknown-{index}")
            pending[call_id] = index
        if _field_of(message, "role") == "tool":
            call_id = str(_field_of(message, "tool_call_id") or "")
            if call_id in pending:
                pending.pop(call_id)
            else:
                issues.append(f"第 {index} 条 tool 消息无对应 tool_calls（孤儿）")
    for call_id, index in pending.items():
        issues.append(f"第 {index} 条 tool_calls({call_id}) 没有 tool 结果（悬空）")
    return issues


def extractive_summary(messages: list[Any], max_chars: int = 600) -> str:
    """抽取式摘要（不调模型）：每条消息取首句，去重后拼接。

    真实项目这里换成小模型做摘要（课程明确要求「摘要用便宜模型」，压缩本身也要花钱）；
    留一个零成本实现是为了让接口在没配模型时也能跑、单测也能断言。
    """
    picked: list[str] = []
    seen: set[str] = set()
    for message in messages or []:
        text = _content_of(message).strip()
        if not text:
            continue
        first = re.split(r"[。！？\n]", text)[0].strip()
        if len(first) < 6 or first in seen:
            continue
        seen.add(first)
        picked.append(first)
    summary = "；".join(picked)
    return summary[:max_chars] + ("…" if len(summary) > max_chars else "")


def compress_messages(
    messages: list[Any],
    *,
    budget_tokens: int = 6000,
    trigger_ratio: float = 0.7,
    keep_recent: int = 2,
    summary: str | None = None,
) -> dict[str, Any]:
    """按预算触发的滚动摘要：保留 system + 最早一轮 + 最近 N 轮 + 一份摘要。

    四条红线（课程 03）：
    1. 不按单条消息裁（会切断 tool_calls ↔ ToolMessage 配对）；
    2. 约束锚点永驻 system（裁掉等于放弃安全边界）；
    3. 摘要用便宜模型做（这里允许外部传入，缺省用抽取式）；
    4. 原始工具结果要落 trace，剪裁只影响「喂给模型的部分」。
    """
    before = ntok_messages(messages)
    system = [m for m in messages or [] if _field_of(m, "role") == "system"]
    rest = [m for m in messages or [] if _field_of(m, "role") != "system"]
    rounds = split_rounds(rest)

    if before < budget_tokens * trigger_ratio or len(rounds) <= keep_recent + 1:
        return {
            "compressed": False,
            "before_tokens": before,
            "after_tokens": before,
            "saved_ratio": 0.0,
            "rounds_before": len(rounds),
            "rounds_after": len(rounds),
            "summary": "",
            "messages": list(messages or []),
            "pairing_issues": check_tool_pairing(list(messages or [])),
            "reason": f"当前 {before} tok < 触发线 {int(budget_tokens * trigger_ratio)} tok，无需压缩",
        }

    head = rounds[0]
    tail = rounds[-keep_recent:]
    middle = rounds[1:-keep_recent]
    summary_text = summary or extractive_summary([m for group in middle for m in group])
    merged = (
        system
        + head
        + [{"role": "system", "content": f"[前情摘要] {summary_text}"}]
        + [m for group in tail for m in group]
    )
    after = ntok_messages(merged)
    return {
        "compressed": True,
        "before_tokens": before,
        "after_tokens": after,
        "saved_ratio": round(1 - after / before, 4) if before else 0.0,
        "rounds_before": len(rounds),
        # 首轮 + 最近 keep_recent 轮（摘要那一条 system 另算，不占「轮」）
        "rounds_after": 1 + len(tail),
        "summary": summary_text,
        "messages": merged,
        "pairing_issues": check_tool_pairing(merged),
        "reason": f"当前 {before} tok ≥ 触发线 {int(budget_tokens * trigger_ratio)} tok，按轮压缩中间段",
    }


def synthetic_history(rounds: int, *, tool_chars: int = 900) -> list[dict[str, Any]]:
    """造一段多轮客服历史（含 tool_calls 配对），用于演示压缩。

    刻意保留「配对」：assistant 带 tool_calls → tool 消息回填 → user 追问 → assistant 收口。
    这样 ``naive_slice`` 的配对问题才暴露得出来。
    """
    history: list[dict[str, Any]] = [
        {"role": "system", "content": PROMPT_LEAN + "\n\n约束锚点：" + CONSTRAINT_ANCHOR},
        {"role": "user", "content": "我的订单到哪了？帮我看下物流，顺便确认金额。"},
    ]
    for index in range(1, max(1, rounds) + 1):
        history.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "name": "get_shipping",
                        "args": {"order_id": f"SO2026111500{index}"},
                    }
                ],
            }
        )
        history.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{index}",
                "content": json.dumps(
                    {
                        "order_id": f"SO2026111500{index}",
                        "status": "已发货",
                        "amount": f"{index * 45}.00",
                        "eta": "2026-11-18",
                        "desc": "详细物流节点说明：" + "转运中心已扫描。" * (tool_chars // 8),
                    },
                    ensure_ascii=False,
                ),
            }
        )
        history.append(
            {
                "role": "user",
                "content": f"第 {index} 次追问：那 {index * 45} 元什么时候退？能不能加急？",
            }
        )
        history.append(
            {
                "role": "assistant",
                "content": f"第 {index} 轮：已发货，预计 11-18 送达，金额 {index * 45} 元；退款需先确认收货。",
            }
        )
    return history


def naive_slice(messages: list[Any], keep: int) -> list[Any]:
    """反面教材：按「条数」从尾部切——课程反复警告的错误姿势，专门用来对照。"""
    return list(messages[-keep:]) if keep > 0 else []


# ============================================================
# 4. 杠杆三：缓存三层
# ============================================================

# 只读白名单：不在名单里的工具（写操作）永不进缓存。
# 生产里除了查订单，还该把 search_faq / query_policy 这类检索工具加进来——
# 这里的五个工具与 TOOL_SCHEMAS 一一对应，保证「白名单里的工具都真实存在」这条自检成立。
READONLY_TOOLS = {"get_order", "get_shipping", "get_coupon", "get_invoice"}

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_ID_RE = re.compile(r"[A-Z]{2}\d{6,}")


def entities(text: str) -> set[str]:
    """抽出「数字」和「单号」——它们不同就代表语义不同，语义缓存绝不能命中。"""
    return set(_NUM_RE.findall(text or "")) | set(_ID_RE.findall(text or ""))


# 语义向量会「忽略」的功能词：真实 embedding 把它们压得极低，这里显式抹掉，
# 才能稳定复现「措辞不同 = 相似度 0.95+」这个真实行为（而不是靠随机抖动碰运气）
FILLER_WORDS = (
    "多久",
    "什么时候",
    "什么时间",
    "何时",
    "怎么",
    "如何",
    "请问",
    "一下",
    "可以",
    "吗",
    "呢",
    "？",
    "?",
    "，",
    ",",
    "。",
    " ",
    "元",
)


def semantic_core(text: str) -> str:
    """抽「语义骨架」：去掉数字与功能词，只留实义词（退款 / 到账 / 发票…）。"""
    core = re.sub(r"\d", "", text or "")
    for word in FILLER_WORDS:
        core = core.replace(word, "")
    return core


def demo_embed(text: str) -> list[float]:
    """演示用的「数字盲」向量：抹掉数字与功能词后哈希。

    真实 embedding 模型对「45 / 450」「多久 / 什么时候」这类差异极不敏感（余弦常 0.97+），
    这个实现把那种失败模式稳定复现出来——于是「实体校验」那一行的价值看得见。
    """
    return hash_embed(semantic_core(text))


class ToolCache:
    """L1 工具结果缓存：只读白名单 + 参数规范化 + TTL。"""

    def __init__(self, ttl: float = 300.0) -> None:
        self.ttl = ttl
        self.store: dict[str, tuple[Any, float]] = {}
        self.hits = 0
        self.misses = 0
        self.blocked = 0

    @staticmethod
    def key(name: str, args: dict[str, Any]) -> str:
        """缓存键 = 工具名 + 排序后的参数 JSON 的哈希（防止 dict 顺序抖动打穿缓存）。"""
        normalized = json.dumps(args or {}, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
        return f"{name}:{digest}"

    def lookup(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """查缓存，返回命中与否 + 原因（原因要能解释给用户看）。"""
        if name not in READONLY_TOOLS:
            self.blocked += 1
            return {"hit": False, "value": None, "reason": "写操作 / 未在白名单：永不缓存（安全阀 1）"}
        item = self.store.get(self.key(name, args))
        if item and time.time() - item[1] < self.ttl:
            self.hits += 1
            return {"hit": True, "value": item[0], "reason": f"命中（TTL {self.ttl:.0f}s 内）"}
        self.misses += 1
        return {"hit": False, "value": None, "reason": "未命中 / 已过期"}

    def save(self, name: str, args: dict[str, Any], value: Any) -> bool:
        """写缓存；写操作直接拒绝（返回 False，便于前端展示被拦下的调用）。"""
        if name not in READONLY_TOOLS:
            self.blocked += 1
            return False
        self.store[self.key(name, args)] = (value, time.time())
        return True

    def stats(self) -> dict[str, Any]:
        """命中统计：命中率只统计「可缓存」的查询，写操作单独计数。"""
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "blocked": self.blocked,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


@dataclass
class CacheItem:
    """语义缓存的一条记录：问题原文 + 向量 + 答案 + 归属用户 + 写入时间。"""

    question: str
    vector: list[float]
    answer: str
    user_id: str
    created_at: float


class SemanticCache:
    """L2 语义缓存：相似度阈值 + **实体一致** + 默认用户隔离。

    ``entities(item.question) != entities(question)`` 这一行是整个 Day 17 最重要的代码：
    「退款 45 元多久到账」与「退款 450 元多久到账」的向量相似度极高（本模块的 demo_embed
    给出 1.0），任何 0.95 阈值都会命中；跳过它就是资损事故。
    """

    def __init__(
        self,
        *,
        threshold: float = 0.95,
        ttl: float = 1800.0,
        cross_user: bool = False,
        embed_fn: Any = None,
    ) -> None:
        self.threshold = threshold
        self.ttl = ttl
        self.cross_user = cross_user
        self.embed = embed_fn or demo_embed
        self.items: list[CacheItem] = []
        self.hits = 0
        self.blocked_entity = 0
        self.blocked_user = 0

    def lookup(self, question: str, user_id: str) -> dict[str, Any]:
        """按「最近写入优先」找相似问题；被安全阀拦下时把原因写清楚。"""
        query_entities = entities(question)
        vector = self.embed(question)
        now = time.time()
        best_score = 0.0
        for item in reversed(self.items):
            if now - item.created_at > self.ttl:
                continue
            score = round(cosine(vector, item.vector), 4)
            best_score = max(best_score, score)
            if score < self.threshold:
                continue
            if not self.cross_user and item.user_id != user_id:
                self.blocked_user += 1
                return {
                    "hit": False,
                    "score": score,
                    "reason": "相似度够但跨用户：拒绝命中（安全阀 3：用户隔离，否则泄露别人的订单）",
                    "matched": item.question,
                }
            if entities(item.question) != query_entities:
                self.blocked_entity += 1
                return {
                    "hit": False,
                    "score": score,
                    "reason": (
                        f"相似度 {score} 够但实体不同：{sorted(entities(item.question))}"
                        f" ≠ {sorted(query_entities)}（安全阀 2）"
                    ),
                    "matched": item.question,
                }
            self.hits += 1
            return {
                "hit": True,
                "score": score,
                "reason": "命中（实体一致 + 同用户）",
                "matched": item.question,
            }
        return {
            "hit": False,
            "score": best_score,
            "reason": f"未命中（最高相似度 {best_score} < 阈值 {self.threshold}）",
            "matched": "",
        }

    def save(self, question: str, answer: str, user_id: str) -> None:
        """写一条缓存；真实项目这里还要加 ``version`` 前缀，改政策就 bump 清空。"""
        self.items.append(
            CacheItem(
                question=question,
                vector=self.embed(question),
                answer=answer,
                user_id=user_id,
                created_at=time.time(),
            )
        )

    def stats(self) -> dict[str, Any]:
        """命中与被拦统计：``blocked_*`` 是「缓存本来会给错答案」的次数。"""
        return {
            "hits": self.hits,
            "blocked_entity": self.blocked_entity,
            "blocked_user": self.blocked_user,
            "items": len(self.items),
        }


def prefix_fingerprint(stable_prefix: str) -> str:
    """前缀指纹：它一变，provider 侧缓存就废了。必须进监控。"""
    return hashlib.sha1((stable_prefix or "").encode("utf-8")).hexdigest()[:10]


def build_request(stable_prefix: str, dynamic_part: str) -> dict[str, Any]:
    """L3 前缀缓存的正确姿势：稳定内容放最前、且逐字节不变；变动内容全部后置。

    返回的 ``prefix_hash`` 就是该进监控的那个数字（成本异常往往先于质量异常）。
    """
    return {
        "messages": [
            {"role": "system", "content": stable_prefix},
            {"role": "user", "content": dynamic_part},
        ],
        "prefix_hash": prefix_fingerprint(stable_prefix),
    }


def prefix_stability(requests: int = 5) -> dict[str, Any]:
    """前缀缓存「坏姿势 / 好姿势」对照（纯逻辑，不调 provider）。

    坏姿势：system 最前面拼 ``当前时间：YYYY-MM-DD HH:MM:SS``（带秒）。
    好姿势：system 只放稳定规则，时间只给日期且挪到用户问题之后（请求末尾）。
    """
    bad_hashes: list[str] = []
    good_hashes: list[str] = []
    for index in range(requests):
        stamp = f"2026-11-15 14:32:{index:02d}"
        bad_prefix = f"当前时间：{stamp}\n你是电商客服助手。规则 1/2/3……"
        bad_hashes.append(prefix_fingerprint(bad_prefix))
        good_hashes.append(prefix_fingerprint("你是电商客服助手。规则 1/2/3……"))

    def _hit_rate(hashes: list[str]) -> float:
        if not hashes:
            return 0.0
        return round((len(hashes) - len(set(hashes))) / len(hashes), 4)

    return {
        "requests": requests,
        "bad": {
            "hashes": bad_hashes,
            "unique": len(set(bad_hashes)),
            "hit_rate": _hit_rate(bad_hashes),
            "note": "带秒时间戳塞在 system 开头：每次请求前缀都不同，缓存命中率归零、成本翻倍",
        },
        "good": {
            "hashes": good_hashes,
            "unique": len(set(good_hashes)),
            "hit_rate": _hit_rate(good_hashes),
            "note": "稳定前缀 + 日期放 messages 末尾：前缀逐字节一致，命中率稳定",
        },
    }


# ============================================================
# 5. 杠杆四：并行与流式
# ============================================================


def serial_duration(durations_ms: list[int]) -> int:
    """串行耗时 = Σ(工具耗时)（有依赖时必须串行，这就是代价）。"""
    return int(sum(durations_ms or []))


def parallel_duration(durations_ms: list[int], limit: int = 4) -> int:
    """并行耗时的纯函数模型：按并发上限分批，每批取 max。

    比 ``random.uniform`` 好断言：同样的输入永远同样的输出，单测能直接比大小。
    """
    if not durations_ms:
        return 0
    limit = max(1, limit)
    ordered = sorted((int(value) for value in durations_ms), reverse=True)
    total = 0
    for start in range(0, len(ordered), limit):
        total += max(ordered[start : start + limit])
    return int(total)


def speedup(durations_ms: list[int], limit: int = 4) -> float:
    """并行加速比（串行 / 并行）：扇出越多、单个越短，收益越接近并发上限。"""
    parallel = parallel_duration(durations_ms, limit)
    if not parallel:
        return 0.0
    return round(serial_duration(durations_ms) / parallel, 2)


async def _run_one(name: str, ms: int, timeout_ms: int, fail: bool) -> dict[str, Any]:
    """跑一个假 IO：``asyncio.sleep`` 模拟外部调用，可注入一次失败。"""
    await asyncio.wait_for(asyncio.sleep(ms / 1000), timeout_ms / 1000)
    if fail:
        raise RuntimeError("模拟工具失败（限流 / 500）")
    return {"name": name, "ok": True, "ms": ms, "error": ""}


async def run_tools_parallel(
    calls: list[dict[str, Any]],
    *,
    limit: int = 4,
    timeout_ms: int = 3000,
    fail_index: int = -1,
) -> dict[str, Any]:
    """并行执行多个工具：Semaphore 控并发、``wait_for`` 控超时、``return_exceptions`` 保整批。

    三道保险缺一不可——一次性扇出 20 个请求会触发 429，然后你重试，然后更糟。
    """
    semaphore = asyncio.Semaphore(max(1, limit))
    started = time.perf_counter()
    timeline: list[dict[str, Any]] = []

    async def one(index: int, call: dict[str, Any]) -> dict[str, Any]:
        name = str(call.get("name") or f"tool{index}")
        async with semaphore:
            begin = time.perf_counter() - started
            try:
                result = await _run_one(name, int(call.get("ms") or 600), timeout_ms, index == fail_index)
                return {**result, "start_ms": round(begin * 1000)}
            except asyncio.TimeoutError:
                return {
                    "name": name,
                    "ok": False,
                    "ms": timeout_ms,
                    "error": f"超时（> {timeout_ms}ms）",
                    "start_ms": round(begin * 1000),
                }
            except Exception as exc:  # noqa: BLE001  单点失败不炸整批
                return {
                    "name": name,
                    "ok": False,
                    "ms": int(call.get("ms") or 0),
                    "error": f"{type(exc).__name__}: {exc}",
                    "start_ms": round(begin * 1000),
                }
            finally:
                timeline.append(
                    {
                        "name": name,
                        "start_ms": round(begin * 1000),
                        "end_ms": round((time.perf_counter() - started) * 1000),
                    }
                )

    results = await asyncio.gather(*(one(index, call) for index, call in enumerate(calls)))
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    durations = [int(call.get("ms") or 0) for call in calls]
    return {
        "results": results,
        "timeline": sorted(timeline, key=lambda item: item["start_ms"]),
        "parallel_ms": elapsed_ms,
        "serial_ms": serial_duration(durations),
        "theoretical_ms": parallel_duration(durations, limit),
        "ok_count": sum(1 for item in results if item["ok"]),
        "fail_count": sum(1 for item in results if not item["ok"]),
    }


# ============================================================
# 6. 预算熔断 + 降级交付
# ============================================================


@dataclass
class Budget:
    """三闸预算：钱 / 步数 / 墙钟。任一超限即熔断。

    三者必须同时在——只限步数挡不住「单步巨贵」，只限钱挡不住「无限等待」。
    """

    max_cost: float = 0.02
    max_steps: int = 6
    max_wall: float = 20.0
    cost: float = 0.0
    steps: int = 0
    t0: float = field(default_factory=time.time)

    def check(self) -> str | None:
        """超限返回原因（用于日志与降级文案），未超限返回 None。"""
        if self.cost >= self.max_cost:
            return f"cost {self.cost:.4f} >= {self.max_cost}"
        if self.steps >= self.max_steps:
            return f"steps {self.steps} >= {self.max_steps}"
        wall = time.time() - self.t0
        if wall >= self.max_wall:
            return f"wall {wall:.1f}s >= {self.max_wall}s"
        return None

    def snapshot(self) -> dict[str, Any]:
        """当前预算状态（前端进度条用）。"""
        return {
            "max_cost": self.max_cost,
            "max_steps": self.max_steps,
            "max_wall": self.max_wall,
            "cost": round(self.cost, 6),
            "steps": self.steps,
            "wall": round(time.time() - self.t0, 2),
            "cost_ratio": round(self.cost / self.max_cost, 4) if self.max_cost else 0.0,
            "step_ratio": round(self.steps / self.max_steps, 4) if self.max_steps else 0.0,
        }


# 演示用的电商只读 / 写工具集：名字与 READONLY_TOOLS 对齐，缓存演示才能自洽
ORDERS: dict[str, dict[str, str]] = {
    "SO20261115001": {
        "order_id": "SO20261115001",
        "status": "已发货",
        "amount": "45.00",
        "eta": "2026-11-18",
        "coupon": "满100减10",
        "invoice": "可开电子发票",
    },
    "SO20261115002": {
        "order_id": "SO20261115002",
        "status": "待发货",
        "amount": "128.00",
        "eta": "2026-11-20",
        "coupon": "无可用券",
        "invoice": "可开电子发票",
    },
}

_REFUND_SEQ = iter(range(9001, 9999))


def _json(data: Any) -> str:
    """工具统一返回 JSON 字符串（模型读字段比读自然语言稳）。"""
    return json.dumps(data, ensure_ascii=False)


def call_tool(name: str, args: dict[str, Any]) -> str:
    """演示工具的执行入口：只做本地查表，不发网络请求。

    未知订单返回结构化错误而不是抛异常——「错误即结果」才能让模型自我纠正（Day 03）。
    """
    args = args or {}
    order_id = str(args.get("order_id") or "")
    order = ORDERS.get(order_id)
    if name == "get_order":
        if not order:
            return _json({"ok": False, "error": "order_not_found", "hint": f"未找到订单 {order_id}"})
        return _json({"ok": True, **order})
    if name == "get_shipping":
        if not order:
            return _json({"ok": False, "error": "order_not_found", "hint": f"未找到订单 {order_id}"})
        return _json(
            {
                "ok": True,
                "order_id": order_id,
                "status": order["status"],
                "eta": order["eta"],
                "desc": "11-15 已出库 → 11-16 到达杭州转运中心 → 预计 11-18 派送。",
            }
        )
    if name == "get_coupon":
        return _json({"ok": True, "order_id": order_id, "coupon": (order or {}).get("coupon", "无可用券")})
    if name == "get_invoice":
        return _json({"ok": True, "order_id": order_id, "invoice": (order or {}).get("invoice", "不支持")})
    if name == "create_refund":
        return _json(
            {
                "ok": True,
                "refund_no": f"RF{next(_REFUND_SEQ)}",
                "order_id": order_id,
                "amount": args.get("amount"),
                "note": "写操作：绝不进缓存，且需要用户确认（Day 04 确认门）",
            }
        )
    return _json({"ok": False, "error": "unknown_tool", "hint": f"未知工具 {name}"})


# 工具说明书（Day 03 三要素：name / description / parameters）
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "按订单号查询订单状态、金额与预计送达时间。需要订单状态/金额时调用。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string", "description": "订单号，形如 SO20261115001"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shipping",
            "description": "查询物流轨迹。用户问「到哪了 / 什么时候到」时调用。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_coupon",
            "description": "查询该订单可用的优惠券。用户问「能不能用券」时调用。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_invoice",
            "description": "查询该订单能否开发票。用户问发票时调用。",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_refund",
            "description": "发起退款（写操作，需用户确认）。不要在未确认收货时调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "amount": {"type": "number", "description": "退款金额（元）"},
                },
                "required": ["order_id", "amount"],
            },
        },
    },
]

DEGRADED_ANSWER = (
    "已查到订单 SO20261115001：状态=已发货，金额=45.00 元，预计 11-18 送达。\n"
    "（本次因预算熔断提前收口：优惠券与发票信息未查到，已转人工继续处理，不会重复扣费。）"
)


def simulate_budget_agent(
    *,
    question: str,
    budget: Budget,
    model: str = "deepseek-chat",
    tool_tokens: int = 260,
    system_tokens: int = 180,
    decision_output: int = 90,
    final_output: int = 220,
    plan_steps: int = 9,
    tool_ms: float = 200.0,
) -> dict[str, Any]:
    """脚本化「模型一直想调工具」的 Agent：跑到预算熔断 → 摘工具 → 降级交付。

    为什么不调模型：熔断逻辑本身是纯逻辑，用脚本化模型才能把「第几步熔断、熔断后说了什么」
    变成可断言的确定性结果（真实模型下这三件事每次都不一样）。
    """
    ledger = Ledger()
    base = system_tokens + ntok(question)
    acc = 0
    steps: list[dict[str, Any]] = []
    stop_reason = "converged"

    for index in range(1, plan_steps + 1):
        reason = budget.check()
        if reason:
            stop_reason = reason
            break
        in_tok = base + acc
        out_tok = decision_output
        latency = in_tok / 1000 * 60 + out_tok * 18 + tool_ms
        record = ledger.add(model, in_tok, out_tok, latency / 1000, note=f"决策{index}")
        budget.cost += record.cost
        budget.steps += 1
        acc += out_tok + tool_tokens
        steps.append(
            {
                "idx": index,
                "node": "agent",
                "tier": "main",
                "model": model,
                "in_tok": in_tok,
                "out_tok": out_tok,
                "cost": record.cost,
                "cost_so_far": round(budget.cost, 6),
                "action": "调用工具 get_order（脚本化：永不收敛）",
                "budget": budget.snapshot(),
                "stopped": False,
            }
        )
    else:
        stop_reason = f"steps {budget.steps} >= {budget.max_steps}"

    reason = budget.check() or stop_reason
    in_tok = base + acc
    latency = in_tok / 1000 * 60 + final_output * 18
    record = ledger.add(model, in_tok, final_output, latency / 1000, note="熔断后的降级交付")
    budget.cost += record.cost
    budget.steps += 1
    steps.append(
        {
            "idx": len(steps) + 1,
            "node": "degrade",
            "tier": "main",
            "model": model,
            "in_tok": in_tok,
            "out_tok": final_output,
            "cost": record.cost,
            "cost_so_far": round(budget.cost, 6),
            "action": f"预算熔断（{reason}）→ 摘掉工具，仅基于已有信息总结",
            "budget": budget.snapshot(),
            "stopped": True,
        }
    )
    return {
        "answer": DEGRADED_ANSWER,
        "stop_reason": reason,
        "degraded": True,
        "steps": steps,
        "budget": budget.snapshot(),
        "ledger": ledger.summary(baseline_in=base),
    }


# ============================================================
# 7. 成本 / 延迟投影（A/B 报告）
# ============================================================


def estimate_latency(
    *,
    steps: int,
    in_tokens: int,
    out_tokens: int,
    tool_ms_serial: int = 0,
    tool_ms_parallel: int | None = None,
) -> dict[str, float]:
    """四段式延迟估算：prefill + decode + 工具 + 编排（单位秒）。

    ``tool_ms_parallel`` 传了就用并行耗时（max 而非 sum），否则用串行。
    这是**线性近似**，用途是相对比较；绝对值请用 live 探针校准。
    """
    steps = max(1, steps)
    prefill = in_tokens / 1000 * PREFILL_SEC_PER_K
    decode = out_tokens * DECODE_SEC_PER_TOKEN
    tool_sec = (tool_ms_parallel if tool_ms_parallel is not None else tool_ms_serial) / 1000
    overhead = steps * OVERHEAD_SEC_PER_STEP
    return {
        "prefill": round(prefill, 3),
        "decode": round(decode, 3),
        "tools": round(tool_sec, 3),
        "overhead": round(overhead, 3),
        "avg_in_per_step": round(in_tokens / steps, 1),
        "total": round(prefill + decode + tool_sec + overhead, 3),
    }


def project_levers(payload: Day17AbRequest, model: str) -> dict[str, Any]:
    """按四个杠杆的开关，逐档现算成本与延迟（课程 06 那张表的「真数据版」）。

    每一档只改一个变量，收益才归因得清：压缩改 token、路由改单价、缓存改步数、并行改等待。
    **正确率这一列故意留空**——它只能由 Day 16 的评估集实测，抄课程数字等于自欺欺人。
    """
    steps = max(1, payload.steps)
    tool_ms = max(10, payload.tool_ms)
    tools_per_step = max(1, payload.parallel_tools)
    serial_tool_ms = steps * tools_per_step * tool_ms
    parallel_tool_ms = steps * tool_ms
    hit_rate = min(0.95, max(0.0, payload.cache_hit_rate))
    mix = mix_saving({"small": 4, "main": 5, "big": 1})

    configs: list[dict[str, Any]] = []

    def add_row(
        name: str,
        *,
        system_tokens: int,
        tool_tokens: int,
        effective_steps: int,
        cost_factor: float,
        parallel: bool,
        note: str,
    ) -> None:
        ledger = simulate_run(
            model=model,
            system_tokens=system_tokens,
            tool_tokens=tool_tokens,
            steps=effective_steps,
            final_output=500,
        )
        summary = ledger.summary(system_tokens + 200)
        latency = estimate_latency(
            steps=len(ledger.steps),
            in_tokens=summary["in_tok"],
            out_tokens=summary["out_tok"],
            tool_ms_serial=effective_steps * tools_per_step * tool_ms,
            tool_ms_parallel=effective_steps * tool_ms if parallel else None,
        )
        configs.append(
            {
                "name": name,
                "steps": effective_steps,
                "in_tok": summary["in_tok"],
                "out_tok": summary["out_tok"],
                "cost": round(summary["cost"] * cost_factor, 6),
                "latency": latency["total"],
                "latency_breakdown": latency,
                "accuracy": None,
                "note": note,
            }
        )

    add_row(
        "基线（全 main + 全量上下文 + 串行工具）",
        system_tokens=700,
        tool_tokens=payload.tool_tokens,
        effective_steps=steps,
        cost_factor=1.0,
        parallel=False,
        note="课程里那张表的起点：每次 run 都在重发整页工具结果",
    )
    add_row(
        "+ 上下文压缩（工具结果 400 字 + prompt 瘦身）",
        system_tokens=180,
        tool_tokens=min(payload.tool_tokens, 120),
        effective_steps=steps,
        cost_factor=1.0,
        parallel=False,
        note="收益最大、风险最小：只影响「喂给模型的部分」",
    )
    add_row(
        f"+ 分级路由（{int(mix['counts']['small'] / mix['total_calls'] * 100)}% 调用走便宜档）",
        system_tokens=180,
        tool_tokens=min(payload.tool_tokens, 120),
        effective_steps=steps,
        cost_factor=1 - mix["saving_ratio"],
        parallel=False,
        note="单价降了，token 一个没少——所以它排在压缩后面",
    )
    effective_steps = max(1, round(steps * (1 - hit_rate)))
    add_row(
        f"+ 工具缓存（命中 {hit_rate * 100:.0f}% → 少跑 {steps - effective_steps} 步）",
        system_tokens=180,
        tool_tokens=min(payload.tool_tokens, 120),
        effective_steps=effective_steps,
        cost_factor=1 - mix["saving_ratio"],
        parallel=False,
        note="缓存省的是「整步」，所以同时砍成本和延迟；风险也最高",
    )
    add_row(
        f"+ 并行工具（每步 {tools_per_step} 个只读工具并发）",
        system_tokens=180,
        tool_tokens=min(payload.tool_tokens, 120),
        effective_steps=effective_steps,
        cost_factor=1 - mix["saving_ratio"],
        parallel=True,
        note="只改等待、不改 token：Σ(工具耗时) → max(工具耗时)",
    )

    base = configs[0]
    for row in configs:
        row["delta_cost"] = round(row["cost"] - base["cost"], 6)
        row["delta_cost_ratio"] = (
            round(1 - row["cost"] / base["cost"], 4) if base["cost"] else 0.0
        )
        row["delta_latency"] = round(row["latency"] - base["latency"], 3)
        row["delta_latency_ratio"] = (
            round(1 - row["latency"] / base["latency"], 4) if base["latency"] else 0.0
        )

    last = configs[-1]
    conclusion = (
        f"成本 ${base['cost']:.4f} → ${last['cost']:.4f}（-{last['delta_cost_ratio'] * 100:.0f}%），"
        f"延迟 {base['latency']:.1f}s → {last['latency']:.1f}s（-{last['delta_latency_ratio'] * 100:.0f}%）。"
        "正确率这一列必须由 Day 16 的门禁实测，任何人给你的「86% → 85%」都只对他自己的数据集成立。"
    )
    return {
        "model": model,
        "rows": configs,
        "conclusion": conclusion,
        "assumptions": {
            "prefill_sec_per_1k": PREFILL_SEC_PER_K,
            "decode_sec_per_token": DECODE_SEC_PER_TOKEN,
            "overhead_sec_per_step": OVERHEAD_SEC_PER_STEP,
            "tools_per_step": tools_per_step,
            "tool_ms": tool_ms,
            "serial_tool_ms": serial_tool_ms,
            "parallel_tool_ms": parallel_tool_ms,
            "cache_hit_rate": hit_rate,
            "router_mix": mix["counts"],
            "note": "延迟是线性近似，用于相对比较；绝对值请用 live 探针 / 真实 trace 校准",
        },
        "gate": {
            "accuracy_drop_max": 0.02,
            "cost_increase_max": 0.15,
            "rule": "正确率降幅 ≤2pt 且成本涨幅 ≤15% 才允许合并（Day 16）",
            "day16_hint": "把候选配置接到 /api/v1/study/day16/regression 双跑同一份评估集",
        },
    }


# ============================================================
# 8. 接口实现：账本
# ============================================================


def _require_langchain() -> None:
    """依赖探测：LangChain 缺失时给安装命令，而不是 ImportError 堆栈。"""
    try:
        import langchain_core  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 只在缺依赖的环境触发
        raise ValueError(f"未安装 langchain，请先执行：{INSTALL_HINT}") from exc


async def _model_name(db: AsyncSession, model_config_id: int | None) -> str:
    """取模型名（用于定价）；没有可用模型时回落到 deepseek-chat——账本实验不该被模型卡住。"""
    try:
        cfg = await resolve_model(db, model_config_id)
    except ValueError:
        return "deepseek-chat"
    return cfg.model_name


def ledger_demo(payload: Day17LedgerRequest, model_name: str) -> dict[str, Any]:
    """课程 01：给一次 6 步 run 记账，并给出两个「只改一个变量」的对照。

    基线 / 剪裁工具结果 / 再加 prompt 瘦身，三个变体用同一套步数与输出长度，
    唯一变量是每步工具结果与 system 的 token 数——这样收益归因才清晰。
    """
    model = payload.model or model_name
    baseline_in = 700 + 200

    baseline_run = simulate_run(
        model=model,
        system_tokens=700,
        tool_tokens=payload.tool_tokens,
        steps=payload.steps,
        final_output=payload.final_output,
    )
    trimmed_run = simulate_run(
        model=model,
        system_tokens=700,
        tool_tokens=min(payload.tool_tokens, 120),
        steps=payload.steps,
        final_output=payload.final_output,
    )
    lean_run = simulate_run(
        model=model,
        system_tokens=180,
        tool_tokens=min(payload.tool_tokens, 120),
        steps=payload.steps,
        final_output=payload.final_output,
    )
    variants = [
        {"name": "基线（全量工具结果 + 胖 prompt）", **baseline_run.summary(baseline_in)},
        {"name": "仅剪裁工具结果（→ 关键字段）", **trimmed_run.summary(baseline_in)},
        {"name": "剪裁 + prompt 瘦身（700 → 180 tok）", **lean_run.summary(baseline_in)},
    ]
    totals = baseline_run.summary(baseline_in)
    return {
        "model": model,
        "variants": variants,
        "rows": [step.row() for step in baseline_run.steps],
        "totals": totals,
        "amplification": totals["amplification"],
        "waste_ratio": totals["waste_ratio"],
        "baseline_in": baseline_in,
        "live_probe": None,
        "files": {"ledger": str(LEDGER_FILE)},
        "notes": [
            "看第一张表：输入 token 从第 1 步到第 6 步一路累加，这就是 O(N²) 的真相。",
            f"放大率 {totals['amplification']}x，其中「重复发送」占 {totals['waste_ratio'] * 100:.0f}%——所有优化都在打这个比例。",
            "第二个变体只剪工具结果：成本降幅远大于「换个便宜模型」，且几乎不掉正确率（课程轨迹一）。",
            "第三个变体再加 prompt 瘦身：静态改动一次生效，之后每次调用都省，累积到 N 步就是 N 倍。",
            "账本必须落到 trace 上（Day 15）按 thread_id / user_id 聚合；只在自己笔记本里算一次不会改变线上行为。",
            "价格按模型名模糊匹配（见 mapping.prices），查不到走兜底价——真实项目请按自己的账单替换。",
        ],
    }


async def run_ledger(db: AsyncSession, payload: Day17LedgerRequest) -> dict[str, Any]:
    """账本接口：默认纯估算；``live=True`` 时额外真调一次模型，用真实 usage 校准估算偏差。"""
    _require_langchain()
    model_name = (
        await _model_name(db, payload.model_config_id)
        if (payload.live or payload.model_config_id)
        else "deepseek-chat"
    )
    result = ledger_demo(payload, model_name)

    if payload.live:
        cfg = await resolve_model(db, payload.model_config_id)
        llm = build_chat_model(cfg, temperature=0.0, run_name="day17-ledger-probe")
        question = "我的订单 SO20261115001 到哪了？金额多少？"
        probe_messages = [{"role": "system", "content": PROMPT_LEAN}, {"role": "user", "content": question}]
        estimated = ntok_messages(probe_messages)
        started = time.perf_counter()
        try:
            resp = await llm.ainvoke([("system", PROMPT_LEAN), ("human", question)])
            usage = getattr(resp, "usage_metadata", None) or {}
            actual_in = int(usage.get("input_tokens") or 0)
            actual_out = int(usage.get("output_tokens") or 0)
            result["live_probe"] = {
                "model": cfg.model_name,
                "estimated_in": estimated,
                "actual_in": actual_in,
                "actual_out": actual_out,
                "error_ratio": round((estimated - actual_in) / actual_in, 4) if actual_in else None,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "cost": step_cost(cfg.model_name, actual_in, actual_out),
                "note": "估算 vs 实测：偏差超过 ±20% 就该换目标模型的编码器，别继续拿 cl100k 硬套",
            }
        except Exception as exc:  # noqa: BLE001  探针失败不该让账本实验整体失败
            result["live_probe"] = {
                "model": cfg.model_name,
                "estimated_in": estimated,
                "actual_in": 0,
                "actual_out": 0,
                "error_ratio": None,
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "cost": 0.0,
                "note": f"探针调用失败：{type(exc).__name__}: {exc}",
            }
        result["notes"].append("live_probe 是「估算 vs 实测」：估算只用于相对比较，账单以 usage 为准。")

    _dump(
        LEDGER_FILE,
        {"model": result["model"], "totals": result["totals"], "variants": result["variants"]},
    )
    return result


# ============================================================
# 9. 接口实现：分级路由
# ============================================================


async def _llm_route(db: AsyncSession, model_config_id: int | None, question: str) -> dict[str, Any]:
    """小模型分类器路由：结构化输出三级降级 → 提示词 JSON → 规则兜底。

    注意「路由本身必须便宜」：这里借用同一份配置只是因为环境只有一份模型，
    真实项目请把这一步换成便宜档模型，否则省下的钱还不够付路由。
    """
    from pydantic import BaseModel, Field

    class Route(BaseModel):
        """路由结论：只有档位和理由两个字段（没有 content，避免自由发挥）。"""

        tier: Tier = Field(description="small=可直接回答的简单问题; main=需要工具或多步推理; big=多约束/长文档")
        reason: str = Field(description="一句话理由")

    # 字段示例必须写进提示词：json_mode 不会自动注入 schema，模型很容易自己编个键名
    prompt = (
        "判断这个问题该用哪档模型回答。只输出 JSON，不要解释。\n"
        "small: 单点事实、寒暄、简单查询\n"
        "main: 需要调用工具、多步推理、信息整合\n"
        "big: 多约束冲突、需要长文档深度推理、需要严谨论证\n"
        '输出格式：{"tier": "small|main|big", "reason": "一句话理由"}'
    )
    cfg = await resolve_model(db, model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day17-router")
    attempts: list[str] = []
    for method in ("default", "function_calling", "json_mode"):
        try:
            runner = (
                llm.with_structured_output(Route)
                if method == "default"
                else llm.with_structured_output(Route, method=method)
            )
            verdict = await runner.ainvoke([("system", prompt), ("human", question)])
            data = verdict.model_dump() if hasattr(verdict, "model_dump") else dict(verdict)
            tier_value = str(data.get("tier") or "").split(".")[-1]
            return {
                "tier": tier_value if tier_value in {"small", "main", "big"} else "main",
                "reason": str(data.get("reason") or ""),
                "method": method,
                "attempts": attempts + [f"{method}: 成功"],
                "model": cfg.model_name,
            }
        except Exception as exc:  # noqa: BLE001  降级失败也是课堂素材
            attempts.append(f"{method}: {type(exc).__name__}: {exc}")

    # 第 4 级：不走 structured_output，纯提示词 + 自己解析 JSON。
    # 现实里最常见的原因：模型是「思考模式」，既不支持 response_format=json_schema，
    # 也不支持 tool_choice（langchain 的 function_calling 方法正好要它）——这三档全会 400。
    try:
        resp = await llm.ainvoke([("system", prompt), ("human", question)])
        text = _content_of(resp).strip()
        match = re.search(r"\{.*?\}", text, re.S)
        data = json.loads(match.group(0)) if match else {}
        route = Route.model_validate(data)
        return {
            "tier": route.tier.value,
            "reason": route.reason,
            "method": "prompt_json",
            "attempts": attempts + ["prompt_json: 成功（提示词 + 手动解析）"],
            "model": cfg.model_name,
        }
    except Exception as exc:  # noqa: BLE001  最后一级兜底
        attempts.append(f"prompt_json: {type(exc).__name__}: {exc}")

    fallback = explain_route(question, 0)
    return {
        "tier": fallback["tier"],
        "reason": f"结构化输出与提示词解析全失败，回落到规则：{fallback['reason']}",
        "method": "rules-fallback",
        "attempts": attempts,
        "model": cfg.model_name,
    }


async def router_demo(db: AsyncSession, payload: Day17RouterRequest) -> dict[str, Any]:
    """课程 02：规则路由 + 小模型分类器 + 三信号升级链 + 分流省钱的账。"""
    started = time.perf_counter()
    rules = explain_route(payload.question, payload.history_rounds)
    llm_result = (
        await _llm_route(db, payload.model_config_id, payload.question) if payload.use_llm else None
    )
    # 升级链现场：小模型先试 → 结构化失败升级 → 工具连续报错再升级 → 兜底封顶
    escalation = simulate_escalation(["schema_error", "ok", "tool_error", "tool_error"])
    mix = mix_saving({"small": 4, "main": 5, "big": 1})
    tiers = [
        {
            "tier": tier,
            "model": info["model"],
            "role": info["role"],
            "price_in": price_of(info["model"])[0],
            "price_out": price_of(info["model"])[1],
        }
        for tier, info in TIER_MODEL.items()
    ]
    return {
        "question": payload.question,
        "rules": rules,
        "llm": llm_result,
        "escalation": escalation,
        "tiers": tiers,
        "mix": mix,
        "notes": [
            "规则路由零成本、可解释，先做这个；小模型分类只在规则覆盖不到时用。",
            "路由本身不能贵——用主力模型判断「该用哪个模型」是负优化，省下的钱还不够付路由。",
            "小模型最容易在 JSON 字段上出错：结构化输出必须用 Pydantic 接住，失败就升级（这就是安全网）。",
            "本仓库的 deepseek 思考模式既不支持 json_schema 也不支持 tool_choice，所以前三级会 400，"
            "第 4 级「提示词 + 手动解析」才是真正跑通的那一跳——降级链要留够层级，别假设厂商都支持。",
            "升级有上限：BIG 封顶，再失败走「降级交付 + 转人工」，不能无限重试（Day 05）。",
            f"分流账：10 次调用里 4 次走便宜档，整体成本 -{mix['saving_ratio'] * 100:.0f}%；单次便宜档比主力省 {mix['per_call_saving'] * 100:.0f}%。",
            "但这 -1pt 正确率必须由 Day 16 的门禁来测，不能靠感觉——见 A/B 报告里的 gate。",
        ],
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ============================================================
# 10. 接口实现：上下文压缩
# ============================================================


def _dirty_tool_output(chars: int) -> str:
    """构造一段「JSON + 整页 HTML」的脏工具返回（课程轨迹一的现场）。"""
    noise = "<div class='row'><span>物流节点</span><span>已扫描</span></div>" * (chars // 60)
    return json.dumps(
        {
            "order_id": "SO20261115001",
            "status": "已发货",
            "amount": "45.00",
            "eta": "2026-11-18",
            "error": "",
            "code": "OK",
            "desc": "详细物流节点说明：" + noise,
        },
        ensure_ascii=False,
    )


async def _model_summary(db: AsyncSession, model_config_id: int | None, text: str) -> str:
    """用小模型做摘要（课程要求「压缩本身也要花钱，所以用便宜档」）。"""
    cfg = await resolve_model(db, model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day17-summary")
    prompt = (
        "把下面的客服会话压缩成不超过 120 字的事实摘要，只保留：订单号、金额、状态、"
        "用户诉求、是否已处理。不要寒暄，不要评论。\n\n" + text[:4000]
    )
    resp = await llm.ainvoke([("human", prompt)])
    return _content_of(resp).strip()


async def compress_demo(db: AsyncSession, payload: Day17CompressRequest) -> dict[str, Any]:
    """课程 03：静态 prompt 瘦身 + 工具结果剪裁 + 预算触发摘要（附反面教材对照）。"""
    started = time.perf_counter()

    fat_tokens = ntok(PROMPT_FAT)
    lean_tokens = ntok(PROMPT_LEAN)
    prompt_block = {
        "fat_tokens": fat_tokens,
        "lean_tokens": lean_tokens,
        "saving_ratio": round(1 - lean_tokens / fat_tokens, 4) if fat_tokens else 0.0,
        "fat": PROMPT_FAT,
        "lean": PROMPT_LEAN,
        "note": "每次调用都省，累积到 N 步就是 N 倍；这是唯一「改一次永久生效」的一刀",
    }

    trimmed = trim_tool_output(_dirty_tool_output(payload.tool_output_chars), max_chars=400)
    tool_block = {
        **{key: value for key, value in trimmed.items() if key != "text"},
        "sample_before": _dirty_tool_output(payload.tool_output_chars)[:300],
        "sample_after": trimmed["text"][:300],
        "note": "宁可留字段名和值，不要留整页原文；原始返回要落 trace（Day 15），剪裁只影响喂给模型的部分",
    }

    history = synthetic_history(payload.history_rounds, tool_chars=payload.tool_output_chars // 4)
    summary_text = ""
    summary_source = "extractive"
    if payload.use_model:
        middle_text = "\n".join(
            _content_of(message) for message in history if _content_of(message)
        )
        try:
            summary_text = await _model_summary(db, payload.model_config_id, middle_text)
            summary_source = "model"
        except Exception as exc:  # noqa: BLE001  模型摘要失败就退回抽取式，不阻塞课堂
            summary_text = ""
            summary_source = f"extractive（模型摘要失败：{type(exc).__name__}）"

    compressed = compress_messages(
        history,
        budget_tokens=payload.budget,
        keep_recent=payload.keep_recent,
        summary=summary_text or None,
    )
    # 反面教材：只留最近几条消息（=「按条数删历史」最常见的写法），配对必然被切断
    naive = naive_slice(history, keep=payload.keep_recent + 1)
    context_block = {
        "before_tokens": compressed["before_tokens"],
        "after_tokens": compressed["after_tokens"],
        "saved_ratio": compressed["saved_ratio"],
        "compressed": compressed["compressed"],
        "reason": compressed["reason"],
        "rounds_before": compressed["rounds_before"],
        "rounds_after": compressed["rounds_after"],
        "summary": compressed["summary"],
        "summary_source": summary_source,
        "pairing_ok": not compressed["pairing_issues"],
        "pairing_issues": compressed["pairing_issues"],
        "naive_tokens": ntok_messages(naive),
        "naive_pairing_issues": check_tool_pairing(naive),
        "preview": [
            {
                "role": str(_field_of(message, "role")),
                "content": _content_of(message)[:120],
                "tool_calls": len(_field_of(message, "tool_calls") or []),
            }
            for message in compressed["messages"]
        ],
        "budget": payload.budget,
        "trigger_tokens": int(payload.budget * 0.7),
    }
    return {
        "prompt": prompt_block,
        "tool_trim": tool_block,
        "context": context_block,
        "notes": [
            "静态瘦身 vs 动态裁剪要分开看：前者改 prompt 本身，后者按运行时 token 预算裁。",
            "红线 1：别按单条消息裁——上面的「反面教材」就是按条数切，配对问题会直接列出来。",
            "红线 2：约束锚点（退款需确认 / 金额上限）永驻 system，裁掉等于放弃安全边界。",
            "红线 3：摘要用便宜模型做，压缩本身也要花钱；模型不可用时退回抽取式摘要。",
            "红线 4：工具原文要落 trace，出问题才能复查——剪裁只影响「喂给模型的部分」。",
        ],
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ============================================================
# 11. 接口实现：缓存三层
# ============================================================


def cache_demo(payload: Day17CacheRequest) -> dict[str, Any]:
    """课程 04：L1 只读白名单 / L2 实体校验 + 用户隔离 / L3 前缀稳定性。"""
    # ---------- L1：工具结果缓存 ----------
    tool_cache = ToolCache(ttl=300)
    scenarios: list[dict[str, Any]] = []

    def hit(name: str, args: dict[str, Any], step: str) -> None:
        result = tool_cache.lookup(name, args)
        scenarios.append(
            {
                "step": step,
                "tool": name,
                "args": args,
                "hit": result["hit"],
                "reason": result["reason"],
            }
        )

    order_args = {"order_id": "SO20261115001"}
    hit("get_order", order_args, "第一次查订单（未命中 → 回源）")
    tool_cache.save("get_order", order_args, call_tool("get_order", order_args))
    hit("get_order", order_args, "第二次查同一订单（命中）")
    # 参数顺序不同也应命中：缓存键对参数排序
    hit("get_order", {"order_id": "SO20261115001"}, "参数顺序不同的同一调用（命中）")
    refund_args = {"order_id": "SO20261115001", "amount": 45}
    hit("create_refund", refund_args, "发起退款（写操作，被白名单拦下）")
    tool_cache.save("create_refund", refund_args, call_tool("create_refund", refund_args))
    hit("create_refund", refund_args, "再发起同一退款（仍然不缓存 → 不会重复退款的假命中）")

    l1 = {
        "scenarios": scenarios,
        "stats": tool_cache.stats(),
        "readonly_tools": sorted(READONLY_TOOLS),
        "note": "安全阀 1：写操作永不进缓存。缓存键用「工具名 + 排序后的参数」的哈希，防 dict 顺序抖动。",
    }

    # ---------- L2：语义缓存（含实体校验） ----------
    semantic = SemanticCache(threshold=payload.threshold, ttl=1800)
    semantic.save(payload.cached_question, payload.answer, payload.user_id)
    matrix: list[dict[str, Any]] = []
    probes = [
        (payload.question, payload.user_id, "同一用户、措辞相近 → 应命中"),
        (payload.other_question, payload.user_id, "同一用户、金额从 45 变 450 → 必须拒绝（实体不同）"),
        (payload.question, payload.other_user_id, "另一个用户问同一句话 → 必须拒绝（用户隔离）"),
    ]
    for question, user_id, expectation in probes:
        result = semantic.lookup(question, user_id)
        matrix.append(
            {
                "question": question,
                "user_id": user_id,
                "hit": result["hit"],
                "score": result["score"],
                "reason": result["reason"],
                "matched": result["matched"],
                "expectation": expectation,
            }
        )

    l2 = {
        "matrix": matrix,
        "stats": semantic.stats(),
        "threshold": payload.threshold,
        "cached_question": payload.cached_question,
        "cached_answer": payload.answer,
        "note": (
            "「退款 45 元多久到账」与「退款 450 元多久到账」的相似度在真实 embedding 下常 ≥0.97，"
            "任何 0.95 阈值都会命中；实体校验这一行是资损与正确率的分界线。"
        ),
    }

    # ---------- L3：前缀缓存 ----------
    sample = build_request("你是电商客服助手。规则 1/2/3……", "我的订单到哪了？（今天是 2026-11-15）")
    l3 = {
        "prefix_stability": prefix_stability(5),
        "sample_request": sample,
        "note": "前缀哈希必须进监控：它不会报错、不会掉正确率，只会让你多付钱。",
    }

    return {
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "notes": [
            "三层缓存的风险递增：L1 是数据过期，L3 几乎无风险，中间那层会直接给你错答。",
            "语义缓存只该缓存「事实型、无个性化」的答案（政策解释 / FAQ），涉及金额与账户的一律别缓存。",
            "阈值宁高不低：0.95 起步，上线后抽样看「命中但被点踩」的比例，超过 1% 就调高或下线。",
            "缓存要能一键清空：给键加 version 前缀（如 v3:），政策变更当天 bump。",
        ],
    }


# ============================================================
# 12. 接口实现：并行与流式
# ============================================================

DEFAULT_CALLS: list[dict[str, Any]] = [
    {"name": "get_order", "ms": 620},
    {"name": "get_shipping", "ms": 980},
    {"name": "get_coupon", "ms": 740},
    {"name": "get_invoice", "ms": 1150},
]


async def parallel_demo(payload: Day17ParallelRequest) -> dict[str, Any]:
    """课程 05：串行 vs 并行（真实 asyncio IO）+ 三道保险 + 时间轴。"""
    calls = payload.tools or DEFAULT_CALLS
    outcome = await run_tools_parallel(
        calls,
        limit=payload.limit,
        timeout_ms=payload.timeout_ms,
        fail_index=payload.fail_index,
    )
    durations = [int(call.get("ms") or 0) for call in calls]
    return {
        "serial_ms": outcome["serial_ms"],
        "parallel_ms": outcome["parallel_ms"],
        "theoretical_ms": outcome["theoretical_ms"],
        "speedup": round(outcome["serial_ms"] / outcome["parallel_ms"], 2) if outcome["parallel_ms"] else 0.0,
        "theoretical_speedup": speedup(durations, payload.limit),
        "results": outcome["results"],
        "timeline": outcome["timeline"],
        "notes": [
            "并行的三个前提：无依赖（参数只来自用户输入或常量）、工具幂等、有并发上限。",
            "三道保险：Semaphore 控并发、wait_for 控总超时、return_exceptions 让单点失败不炸整批。",
            "写操作不要并行：失败会重试，重试就是重复下单 / 重复退款。",
            "流式不降总延迟，只把首字时间从 8 秒变成 0.8 秒；对话类产品必须流式，后台批处理不需要。",
        ],
    }


# ============================================================
# 13. 接口实现：预算熔断 Agent
# ============================================================

LIVE_SYSTEM = """你是电商客服助手。
规则：事实必须来自工具；不确定就查；查不到就说查不到并转人工。
回答 ≤3 句，金额与时效必须引用工具原始字段。"""


async def _live_budget_agent(db: AsyncSession, payload: Day17BudgetRequest) -> dict[str, Any]:
    """真实模型 + 真工具的预算熔断现场（默认不跑，前端选 live 才走这里）。"""
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day17-budget")
    budget = Budget(max_cost=payload.max_cost, max_steps=payload.max_steps, max_wall=payload.max_wall)
    cache = ToolCache()
    ledger = Ledger()
    messages: list[Any] = [SystemMessage(LIVE_SYSTEM), HumanMessage(payload.question)]
    steps: list[dict[str, Any]] = []
    answer = ""
    stop_reason = "converged"

    for index in range(1, payload.max_steps + 1):
        over = budget.check()
        if over:
            stop_reason = over
            break
        started = time.perf_counter()
        resp = await llm.bind_tools(TOOL_SCHEMAS).ainvoke(messages)
        usage = getattr(resp, "usage_metadata", None) or {}
        in_tok = int(usage.get("input_tokens") or 0) or ntok_messages(messages)
        out_tok = int(usage.get("output_tokens") or 0) or ntok(_content_of(resp))
        record = ledger.add(
            cfg.model_name, in_tok, out_tok, time.perf_counter() - started, note=f"决策{index}"
        )
        budget.cost += record.cost
        budget.steps += 1
        messages.append(resp)
        calls = getattr(resp, "tool_calls", None) or []
        steps.append(
            {
                "idx": index,
                "node": "agent",
                "tier": "main",
                "model": cfg.model_name,
                "in_tok": in_tok,
                "out_tok": out_tok,
                "cost": record.cost,
                "cost_so_far": round(budget.cost, 6),
                "action": (
                    "调工具：" + ", ".join(str(call.get("name")) for call in calls)
                    if calls
                    else "直接回答（收敛）"
                ),
                "budget": budget.snapshot(),
                "stopped": False,
            }
        )
        if not calls:
            answer = _content_of(resp)
            break
        for call in calls:
            name = str(call.get("name") or "")
            args = call.get("args") or {}
            cached = cache.lookup(name, args)
            if cached["hit"]:
                text, source = str(cached["value"]), "cache"
            else:
                text = call_tool(name, args)
                source = "tool"
                cache.save(name, args, text)
            messages.append(ToolMessage(text[:400], tool_call_id=str(call.get("id"))))
            steps.append(
                {
                    "idx": len(steps) + 1,
                    "node": "tool",
                    "tier": "main",
                    "model": name,
                    "in_tok": 0,
                    "out_tok": ntok(text[:400]),
                    "cost": 0.0,
                    "cost_so_far": round(budget.cost, 6),
                    "action": f"{name}（{source}）→ {text[:80]}",
                    "budget": budget.snapshot(),
                    "stopped": False,
                }
            )

    degraded = stop_reason != "converged"
    if degraded or not answer:
        # 熔断 / 未收敛：摘掉工具逼模型总结 —— 用户拿到「部分答案 + 转人工」比 500 好
        messages.append(
            SystemMessage(
                f"[预算熔断: {stop_reason}] 禁止再调用工具，仅基于已获得的信息回答；"
                "信息不足就说明需转人工，不要编造。"
            )
        )
        started = time.perf_counter()
        try:
            resp = await llm.ainvoke(messages)
            answer = _content_of(resp)
            usage = getattr(resp, "usage_metadata", None) or {}
            record = ledger.add(
                cfg.model_name,
                int(usage.get("input_tokens") or 0) or ntok_messages(messages),
                int(usage.get("output_tokens") or 0) or ntok(answer),
                time.perf_counter() - started,
                note="熔断后的降级交付",
            )
        except Exception as exc:  # noqa: BLE001  连降级调用都失败就给兜底文案
            record = ledger.add(cfg.model_name, ntok_messages(messages), 60, 0.0, note="降级失败兜底")
            answer = f"抱歉，本次处理未完成（{type(exc).__name__}），已转人工继续，请稍候。"
        budget.cost += record.cost
        steps.append(
            {
                "idx": len(steps) + 1,
                "node": "degrade",
                "tier": "main",
                "model": cfg.model_name,
                "in_tok": record.in_tok,
                "out_tok": record.out_tok,
                "cost": record.cost,
                "cost_so_far": round(budget.cost, 6),
                "action": f"预算熔断（{stop_reason}）→ 摘掉工具，仅基于已有信息总结",
                "budget": budget.snapshot(),
                "stopped": True,
            }
        )

    return {
        "answer": answer,
        "stop_reason": stop_reason,
        "degraded": degraded,
        "steps": steps,
        "budget": budget.snapshot(),
        "ledger": ledger.summary(baseline_in=ntok(LIVE_SYSTEM) + ntok(payload.question)),
        "cache": cache.stats(),
    }


async def budget_demo(db: AsyncSession, payload: Day17BudgetRequest) -> dict[str, Any]:
    """课程 06：带三闸预算熔断的 Agent（simulate 确定性 / live 真模型）。"""
    started = time.perf_counter()
    model_name = (
        await _model_name(db, payload.model_config_id)
        if payload.mode == "live"
        else "deepseek-chat"
    )
    if payload.mode == "live":
        _require_langchain()
        outcome = await _live_budget_agent(db, payload)
    else:
        budget = Budget(
            max_cost=payload.max_cost, max_steps=payload.max_steps, max_wall=payload.max_wall
        )
        outcome = {
            **simulate_budget_agent(question=payload.question, budget=budget, model=model_name),
            "cache": {"hits": 0, "misses": 0, "blocked": 0, "hit_rate": 0.0},
        }

    return {
        "mode": payload.mode,
        "model": model_name,
        **outcome,
        "notes": [
            "熔断点是「步数闸」：脚本化模型永不收敛，所以必然在 max_steps 处被拦住——这正是要防的变体自旋。",
            "熔断后不报错：摘掉工具、逼模型用已有信息总结，用户拿到「部分答案 + 转人工」。",
            "三闸必须同时在：只限步数挡不住单步巨贵，只限钱挡不住无限等待。",
            "生产里把每步的 cost / steps / wall 落 trace（Day 15），按 thread 聚合做告警。",
        ],
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ============================================================
# 14. 接口实现：A/B 投影
# ============================================================


async def ab_demo(db: AsyncSession, payload: Day17AbRequest) -> dict[str, Any]:
    """课程 06 的对比表：四个杠杆逐档现算成本与延迟，正确率留给 Day 16。"""
    model = payload.model or await _model_name(db, None)
    report = project_levers(payload, model)
    _dump(AB_FILE, report)
    report["files"] = {"ab_report": str(AB_FILE)}
    report["notes"] = [
        "一次只改一个变量：每行相对上一行只动一个杠杆，收益才归因得清。",
        "正确率列故意留空（null）：它只能由 Day 16 的评估集实测，照抄课程数字等于自欺欺人。",
        "读表顺序：先看成本降幅够不够，再看延迟降幅，最后跑 Day 16 门禁确认没变坏。",
        "门禁不过就回退一档——这四档是「可回退的阶梯」，不是一次性全部上线。",
        "延迟是线性近似：prefill 6ms/1k tok、decode 20ms/token、编排 50ms/步，仅供相对比较。",
    ]
    return report


# ============================================================
# 15. 概念表
# ============================================================


def mapping() -> dict[str, Any]:
    """Day 17 静态概念数据：成本结构 / 延迟四段 / 四杠杆 / 缓存三层 / 六个坑 / 速查表 / 价格。"""
    return {
        "cost_structure": COST_STRUCTURE,
        "latency_breakdown": LATENCY_BREAKDOWN,
        "levers": LEVERS,
        "cache_layers": CACHE_LAYERS,
        "pitfalls": PITFALLS,
        "cheatsheet": CHEATSHEET,
        "checklist": CHECKLIST,
        "presets": PRESETS,
        "prices": PRICE_TABLE,
        "thresholds": {
            **BUDGET_DEFAULTS,
            "cost_trigger_ratio": 0.7,
            "semantic_cache_threshold": 0.95,
            "tool_cache_ttl": 300,
            "prefix_cache_min_tokens": 1024,
        },
        "tiers": TIER_MODEL,
        "default_calls": DEFAULT_CALLS,
        "readonly_tools": sorted(READONLY_TOOLS),
        "files": {
            "ledger": str(LEDGER_FILE),
            "ab_report": str(AB_FILE),
        },
        "install_hint": INSTALL_HINT,
    }
