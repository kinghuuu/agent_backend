"""Day 20 可靠性：当 Agent 服务在凌晨三点挂掉时。

课程 day20 的六段脚本在这里收成一个模块：

    故障分层    → ``retry_demo``     （01：先分类再行动 + 退避抖动 + run 级总预算）
    降级阶梯    → ``ladder_demo``    （02：L0 主模型 → L1 备模型 → L2 缓存 → L3 结构化道歉）
    优雅退出    → ``shutdown_demo``  （03：摘流量 → 拒新 → drain → 落盘 → flush）
    容量评估    → ``capacity_demo``  （04：Little's Law + 三道天花板 + 恒定速率压测读数）
    SLO 与告警  → ``slo_demo``       （05：窗口指标 + 烧毁率 + 最小样本量 + runbook）
    混沌演练    → ``chaos_demo``     （06：注入开关 + 预期声明表 + 盲区清单）
    概念数据    → ``mapping``

四条贯穿全篇的结论：

1. **单跳的「五个九」撑不起整条链路**：一个 10 步 run 要穿 LLM 五六次、工具三四次，
   每跳 99.9% 时整条只剩 ``0.999^9 ≈ 99.1%``——可靠性必须建立在「跳会失败」的假设上；
2. **失败必须先分类再行动**：瞬态重试、确定性失败立即上抛、逻辑性失败改 prompt 再试，
   未知错误 fail-closed（Day 18 原则）；
3. **认输也要体面**：阶梯每降一级都要显式标注 ``degraded``，而降级答案绝不回写缓存——
   缓存的是「已知正确」，不是「碰巧有货」；
4. **可靠性不是写出来的，是演练出来的**：没演练过的降级链，第一次真实启用时大概率发现
   备模型 key 没配、缓存指向测试环境、runbook 里的面板早已改名。

与课程示例的工程差异（都是为了在本仓库里可跑、可测、可复现）：

- 课程 01 的重试用 ``time.sleep`` 真等，一轮实验要几十秒；本模块只**模拟**等待，
  时间线与总耗时照常输出，接口毫秒级返回；
- 课程 02 的 ``call_llm`` 打真实模型、``supports_fc`` 是写死的白名单；本模块把
  「主路是否恢复 / 备路是否可用 / 缓存是否命中」变成请求参数，纯逻辑可复现（不花 token），
  但**能力对齐校验仍然保留**——给不支持 FC 的备模型发 tools 会 400，这条不能省；
- 课程 03 的 ``GracefulExit`` 依赖真实信号与事件循环；本模块把退出时序做成**确定性时间线
  模拟**（谁跑完、谁落盘可续跑、谁被撕碎），并额外校验 ``preStop`` 是否够长——课程里
  特意点出的那行 ``sleep 5``，漏了就是「优雅退出写得再好也白搭」；
- 课程 04 的压测脚本要先起服务；本模块用排队模拟器算出 p50/p95/503，把「读数字做决策」
  这一课单独拎出来（课程自己也在注释里承认：读数决策表比脚本更重要）；
  **顺带修正课程的一处算术失误**：`$3000/月 ÷ $0.009/run` 不是 12.9 run/s 而是 0.13 run/s
  （差 100 倍），默认预算因此设为 $200k 才复现出「上游 8.33 run/s 是瓶颈」那一幕；
- 课程 05 依赖 ``traces/2026-11-18.jsonl`` 落盘文件；本模块**合成**同构的 span 数据
  （并故意夹一行坏 JSON，演示半行日志跳过不崩溃），同样的 ``load_runs`` /
  ``window_metrics`` / ``burn_rate`` 纯函数在单测里直接使用；
- 课程 06 的 ``maybe_inject`` 用 ``time.sleep`` 与真 SIGTERM；本模块用可复现的随机注入
  模拟一次 run，并把「预期声明表」变成机器可核对的判定（判定落空即盲区）。
"""

from __future__ import annotations

import heapq
import json
import math
import random
import time
from importlib import metadata
from typing import Any

from app.modules.study.schemas.study import (
    Day20CapacityRequest,
    Day20ChaosRequest,
    Day20LadderRequest,
    Day20RetryRequest,
    Day20ShutdownRequest,
    Day20SloRequest,
)

# ============================================================
# 0. 静态概念数据（全部来自课程原文）
# ============================================================

# 0.1 错误四分类：先分类，再行动
ERROR_CLASSES: list[dict[str, str]] = [
    {
        "no": "1",
        "cls": "瞬态可重试",
        "action": "retry",
        "examples": "网络超时、429、502/503、连接重置",
        "why": "下一次大概率成功：指数退避 + 抖动重试",
    },
    {
        "no": "2",
        "cls": "确定性失败",
        "action": "fail",
        "examples": "400 参数错、401 鉴权失败、404、422",
        "why": "重试一万次结果一样，还烧钱：立即换路或认输",
    },
    {
        "no": "3",
        "cls": "可换路",
        "action": "fallback",
        "examples": "主模型持续 429、主工具进程挂了",
        "why": "换条路到达同一目的地：备模型 / 备工具 / 缓存",
    },
    {
        "no": "4",
        "cls": "逻辑性失败",
        "action": "fix_prompt",
        "examples": "答非所问、格式解析失败、假收敛（Day 05）",
        "why": "原样重试 = 复原样错误：必须把报错回填进 prompt",
    },
]

# 0.2 重试三要素
RETRY_RULES: list[dict[str, str]] = [
    {
        "no": "1",
        "element": "指数退避",
        "formula": "min(base × 2^(n-1), cap)",
        "why": "提供商一抖，立刻重试等于再撞一次墙；退避给上游恢复时间",
    },
    {
        "no": "2",
        "element": "抖动",
        "formula": "× (0.5 + random())",
        "why": "打散同步脉冲：否则 50 个并发 run 会一起在同一毫秒重试",
    },
    {
        "no": "3",
        "element": "硬上限",
        "formula": "单步 ≤ 3 次 + run 级总预算（Day 17 账本）",
        "why": "单步重试成功 ≠ run 安全：10 步 × 3 次 = 30 次调用",
    },
]

# 0.3 降级阶梯
LADDER_LEVELS: list[dict[str, str]] = [
    {
        "level": "L0",
        "model": "主模型（最强、最贵）",
        "trigger": "正常路径",
        "sacrifice": "—",
        "degraded": "false",
    },
    {
        "level": "L1",
        "model": "备用提供商的备模型",
        "trigger": "主模型重试 3 次耗尽",
        "sacrifice": "质量略降；能力必须对齐（FC / 结构化输出 / 上下文长度）",
        "degraded": "true",
    },
    {
        "level": "L2",
        "model": "语义缓存命中",
        "trigger": "备模型也失败",
        "sacrifice": "答案可能过期：必须标「参考历史回答」",
        "degraded": "true",
    },
    {
        "level": "L3",
        "model": "结构化道歉 + 后续路径",
        "trigger": "缓存也未命中",
        "sacrifice": "没有答案，但有体面：绝不出裸 500，绝不静默吞掉",
        "degraded": "true",
    },
]

# 0.4 优雅退出时序（SIGTERM 之后）
SHUTDOWN_STEPS: list[dict[str, str]] = [
    {"no": "①", "stage": "摘流量", "detail": "立即 stop accepting：新请求不再进来", "if_skipped": "退出瞬间仍收新请求 → 全部 502"},
    {"no": "②", "stage": "拒新请求", "detail": "已接收未开始的请求返回 503 + Retry-After", "if_skipped": "客户端白等到超时，还占着连接"},
    {"no": "③", "stage": "drain 在途", "detail": "有上限地等：短任务跑完；长 run 等下一个超步边界落盘", "if_skipped": "在途 run 被硬切，interrupt 审批凭空消失"},
    {"no": "④", "stage": "审批挂起", "detail": "靠 PostgresSaver 天然安全：状态在库里，只要求 resume 端点在别的副本可用", "if_skipped": "用户点「同意」时 404（Day 19 轨迹二）"},
    {"no": "⑤", "stage": "超时收尾", "detail": "宽限期将尽：对未完成 run 做最后落盘后退出", "if_skipped": "SIGKILL 撕碎，连日志都来不及写"},
    {"no": "⑥", "stage": "flush", "detail": "flush 日志 / trace（Day 15 的教训：忘 flush = 丢最后一段轨迹）", "if_skipped": "事故复盘时恰好缺了最关键的那几秒"},
]

# 0.5 容量三道天花板
CAPACITY_CEILINGS: list[dict[str, str]] = [
    {
        "name": "钱",
        "formula": "月预算 ÷ (30×86400) ÷ 单 run 成本",
        "source": "Day 17 账本",
        "why": "Agent 服务的特色：很多人只看机器，最后被账单叫醒",
    },
    {
        "name": "上游配额",
        "formula": "RPM ÷ 每 run 平均调用数",
        "source": "提供商控制台",
        "why": "你自己限流做得再好，也挡不住提供商先限你",
    },
    {
        "name": "机器资源",
        "formula": "副本数 × 单副本并发 ÷ 单 run 停留时间",
        "source": "CPU / 内存 / 连接池",
        "why": "唯一被传统容量规划覆盖的那道，也是最容易被拍脑袋的那道",
    },
]

# 0.6 最小 SLO 集
SLO_TABLE: list[dict[str, str]] = [
    {"sli": "run 成功率", "slo": "99%（5 分钟窗口）", "source": "status != ok 计数 / 总数", "why": "最基础的存活信号"},
    {"sli": "run p95 完成时长", "slo": "≤ 30s", "source": "按 trace_id 聚合首尾 span", "why": "用户体感的主指标"},
    {"sli": "降级率", "slo": "≤ 10%", "source": "response.degraded 计数 / 总数", "why": "比错误率更早叫（阶梯让人「不报错」）"},
    {"sli": "单 run 成本 p95", "slo": "≤ 预算 2 倍", "source": "cost 字段", "why": "抓成本失控（Day 17 放大效应）"},
    {"sli": "重试率", "slo": "≤ 15%", "source": "retry 事件计数", "why": "上游健康状况的先行指标"},
]

# 0.7 告警三原则
ALERT_PRINCIPLES: list[dict[str, str]] = [
    {
        "no": "1",
        "name": "烧毁率而非绝对阈值",
        "detail": "错误预算被消耗的速度：1 小时烧 14.4 倍 → 页面；6 小时烧 6 倍 → 工单",
        "why": "小流量时「错误率 > 5%」噪声巨大：10 个 run 错 1 个就 10%",
    },
    {
        "no": "2",
        "name": "先看分母",
        "detail": "规则必须带最小样本量（如 5 分钟 ≥ 50 个 run 才评估）",
        "why": "深夜流量低谷一个偶发错误就能触发假告警",
    },
    {
        "no": "3",
        "name": "每条告警自带 runbook",
        "detail": "指标怎么算 / 最可能的三个原因 / 第一步动作",
        "why": "凌晨三点被叫醒的人不该现场推理；没有 runbook 的告警会被学会忽略",
    },
]

# 0.8 混沌演练红线
CHAOS_REDLINES: list[str] = [
    "只在预发环境演练，或生产低峰期 + 灰度单副本；绝不全量开注。",
    "注入必须有总开关（CHAOS_OFF）和自动到期：演练开关忘了关就是真事故。",
    "演练前在群里宣布时间窗；演练后 24 小时内出复盘，修复项进 issue 追踪——价值全部兑现在复盘里。",
]

# 0.9 常见的坑
PITFALLS: list[dict[str, str]] = [
    {
        "no": "1",
        "title": "所有异常套同一个 except 无限重试",
        "symptom": "400 参数错也重试 10 次，账单涨 10 倍，错误一次没少",
        "fix": "先分类再行动：确定性失败立即上抛，未知错误 fail-closed",
    },
    {
        "no": "2",
        "title": "裸重试（无退避无抖动）",
        "symptom": "提供商一抖，50 个并发 run 同时重试，同步脉冲把上游彻底打死",
        "fix": "指数退避 + 0.5~1.5 倍抖动 + 硬上限",
    },
    {
        "no": "3",
        "title": "换路不做能力对齐",
        "symptom": "给不支持 function calling 的备模型发 tools，直接 400",
        "fix": "把瞬态错误换成了确定性错误：先校验能力，不满足就跳过这级",
    },
    {
        "no": "4",
        "title": "降级答案回写缓存",
        "symptom": "一次故障被放大成 N 次：之后每个命中缓存的用户都拿到同一个坏答案",
        "fix": "降级答案绝不回写；缓存「已知正确」，不缓存「碰巧有货」",
    },
    {
        "no": "5",
        "title": "只监控 error_rate",
        "symptom": "阶梯让系统「没有报错」，但用户拿到的是打折回答，团队对此一无所知",
        "fix": "degraded_rate 单独统计并告警：它比 error_rate 早叫 2 分钟",
    },
    {
        "no": "6",
        "title": "把优雅退出当成「等所有 run 跑完」",
        "symptom": "30 秒宽限期里等 3 分钟的 run，最后被 SIGKILL 撕得更碎",
        "fix": "本质是「没跑完的变成能续跑的」：超步边界落盘 + 强制收尾",
    },
    {
        "no": "7",
        "title": "Semaphore 拍脑袋 / 全并发压测",
        "symptom": "拍 8 只能跑出目标吞吐的 13%；全并发猛灌只测出自己的闸门",
        "fix": "三道天花板取最小值算并发；恒定速率慢压测 ≥ 60s 读 p95 与 503",
    },
]

# 0.10 速查表
CHEATSHEET: list[dict[str, str]] = [
    {"topic": "错误怎么分类", "content": "retry（瞬态/429）/ fail（400/401）/ fix_prompt（解析失败回填）/ fallback", "key": "未知错误 fail-closed（Day 18）"},
    {"topic": "重试三要素", "content": "指数退避 + 抖动 + 硬上限", "key": "抖动打散脉冲；run 级预算另算"},
    {"topic": "降级阶梯", "content": "L0 主模型 → L1 备模型 → L2 缓存 → L3 结构化道歉", "key": "L1 要能力对齐；降级答案不回写缓存"},
    {"topic": "换路防乒乓", "content": "探活窗口：连续 N 次成功才回主路", "key": "和 Day 05 的循环不收敛是同一类病"},
    {"topic": "降级可观测", "content": "响应带 degraded 标记；degraded_rate 单独告警", "key": "它比 error_rate 更早叫"},
    {"topic": "优雅退出", "content": "摘流量 → 503 拒新 → drain → 落盘 → flush", "key": "本质是「没跑完的变成能续跑的」"},
    {"topic": "编排配合", "content": "terminationGracePeriod > drain 窗口；preStop sleep 5", "key": "没 sleep 5，摘除瞬间流量照打"},
    {"topic": "Semaphore 定容", "content": "并发 = min(钱, 上游配额, 机器) × 延迟 ÷ 副本数", "key": "三道天花板取最小，别拍脑袋"},
    {"topic": "压测方式", "content": "恒定速率慢压测 ≥ 60s，读 p95 与状态码分布", "key": "全并发猛灌只能测出自己的闸门"},
    {"topic": "SLO 最小集", "content": "成功率 99% / p95 ≤ 30s / 降级率 ≤ 10% / 成本 p95 / 重试率", "key": "数据源就是 Day 15 的 trace JSONL"},
    {"topic": "告警原则", "content": "烧毁率 ≥ 14.4x 页面 / 6x 工单；最小样本量；自带 runbook", "key": "没 runbook 的告警等于没有告警"},
    {"topic": "混沌演练", "content": "注入开关 + 预期声明表 + 复盘；CHAOS_OFF 总开关", "key": "预期落空处 = 可靠性盲区"},
    {"topic": "缓存铁律", "content": "降级答案绝不回写缓存", "key": "缓存「已知正确」，不缓存「碰巧有货」"},
]

# 0.11 告警 runbook（告警消息必须自带）
RUNBOOK: dict[str, list[str]] = {
    "error_burn": [
        "1. 查 degraded_rate 是否同时升高 → 是：按提供商故障处理（看 L1 切换日志）",
        "2. 单工具错误率 > 50% → 该 MCP Server 崩了，重启该容器，主服务不受影响",
        "3. 全绿但错误率高 → 看 Day 15 回放树最近 10 条失败 trace，找共同节点",
    ],
    "degraded_burn": [
        "1. L1 备模型在用吗？→ 查探活窗口状态，主模型是否持续 429",
        "2. L2 缓存命中率骤降？→ 缓存服务挂了，修复后降级率会自然回落",
        "3. 都没异常但降级率高 → 可能是能力对齐白名单误杀，检查 supports_fc",
    ],
}

# 0.12 run 级总预算（单步重试成功 ≠ run 安全）
RUN_BUDGET: dict[str, float] = {"max_llm_calls": 30, "max_wall_seconds": 180}


def _version(pkg: str) -> str:
    """读取依赖版本；没装就返回「未安装」，不让接口因缺包 500。"""
    try:
        return metadata.version(pkg)
    except Exception:  # noqa: BLE001  缺包 / 元数据损坏都算未安装
        return "未安装"


# ============================================================
# 1. 故障分层：分类器 + 退避重试器
# ============================================================


class ReliabilityFault(Exception):
    """可靠性实验里所有异常的基类（便于统一分类）。"""


class TransientError(ReliabilityFault):
    """瞬态错误：网络超时、502/503、连接重置。"""


class RateLimitError(TransientError):
    """429：同时是「瞬态可重试」和「需要降速」的信号。"""


class PermanentError(ReliabilityFault):
    """确定性失败：400 参数错、401 鉴权失败、404、422。"""


class SchemaValidationError(ReliabilityFault):
    """模型输出不符合 Schema（课程里对应 pydantic.ValidationError）。"""


class UnknownFault(ReliabilityFault):
    """未分类异常：按 fail-closed 处理。"""


# 场景表：接口只收一个字符串，映射成异常类型 + 动作 + 解释
SCENARIOS: dict[str, dict[str, str]] = {
    "transient": {
        "type": "TransientError",
        "label": "网络超时 / 502 / 连接重置",
        "action": "retry",
        "why": "瞬态：下一次大概率成功，退避后重试",
    },
    "rate_limit": {
        "type": "RateLimitError",
        "label": "429 限流（响应带 Retry-After）",
        "action": "retry",
        "why": "可重试但必须降速：等待时间取 max(退避, Retry-After)",
    },
    "permanent": {
        "type": "PermanentError",
        "label": "400 参数错 / 401 鉴权失败",
        "action": "fail",
        "why": "确定性失败：重试一万次结果一样，立即上抛",
    },
    "invalid_output": {
        "type": "SchemaValidationError",
        "label": "模型输出不符合 Schema",
        "action": "fix_prompt",
        "why": "逻辑性失败：原样重试 = 复原样错误，必须回填报错再试",
    },
    "unknown": {
        "type": "UnknownFault",
        "label": "未分类异常",
        "action": "fail",
        "why": "未知错误 fail-closed：宁可认输，不要赌",
    },
}


def classify(exc: Exception) -> str:
    """把异常翻译成动作：retry / fix_prompt / fallback / fail。

    注意 ``RateLimitError`` 是 ``TransientError`` 的子类，必须**先判子类**——
    否则 isinstance 会先命中父类分支，虽然这里两者动作相同，但换路策略会不同。
    """
    if isinstance(exc, RateLimitError):
        return "retry"
    if isinstance(exc, TransientError):
        return "retry"
    if isinstance(exc, PermanentError):
        return "fail"
    if isinstance(exc, SchemaValidationError):
        return "fix_prompt"
    return "fail"


def build_fault(scenario: str) -> Exception:
    """按场景造一个异常实例（单测与接口共用，避免异常类型散落各处）。"""
    mapping: dict[str, type[Exception]] = {
        "transient": TransientError,
        "rate_limit": RateLimitError,
        "permanent": PermanentError,
        "invalid_output": SchemaValidationError,
        "unknown": UnknownFault,
    }
    return mapping.get(scenario, UnknownFault)("模拟故障")


def backoff_ms(
    attempt: int,
    base_ms: int = 500,
    cap_ms: int = 8000,
    rng: random.Random | None = None,
) -> int:
    """指数退避 + 抖动：``min(base × 2^(n-1), cap) × (0.5 ~ 1.5)``。

    attempt 从 1 开始（第一次重试前等的就是 base 量级）。抖动不可省：
    没有它，50 个并发 run 会在同一毫秒一起重试，形成同步脉冲。
    """
    attempt = max(1, attempt)
    raw = min(base_ms * (2 ** (attempt - 1)), cap_ms)
    jitter = (rng or random).uniform(0.5, 1.5)
    return int(raw * jitter)


def _pulse_stats(waits: list[int], bucket_ms: int = 250) -> dict[str, Any]:
    """把一组等待时间打成直方图，用来看「同步脉冲」有没有被打散。"""
    if not waits:
        return {"buckets": [], "spread_ms": 0, "peak_share": 0.0}
    buckets: dict[int, int] = {}
    for wait in waits:
        idx = wait // bucket_ms
        buckets[idx] = buckets.get(idx, 0) + 1
    rows = [
        {"range": f"{idx * bucket_ms}~{(idx + 1) * bucket_ms}ms", "count": count}
        for idx, count in sorted(buckets.items())
    ]
    return {
        "buckets": rows,
        "spread_ms": max(waits) - min(waits),
        "peak_share": round(max(buckets.values()) / len(waits), 3),
    }


def retry_demo(payload: Day20RetryRequest) -> dict[str, Any]:
    """模拟一次「调用 → 失败 → 分类 → 退避 → 再试」的完整时间线。

    等待是**模拟**的（只累加毫秒数，不真 sleep）：既能展示退避曲线，又不会让接口
    在一次实验里盯上几十秒。
    """
    rng = random.Random(payload.seed)
    meta = SCENARIOS[payload.scenario]
    action = meta["action"]

    timeline: list[dict[str, Any]] = []
    total_wait = 0
    success = False
    prompt_patch: str | None = None

    for attempt in range(1, payload.max_attempts + 1):
        if attempt > payload.fail_times:
            timeline.append(
                {
                    "attempt": attempt,
                    "outcome": "ok",
                    "action": action,
                    "wait_ms": 0,
                    "detail": "调用成功，循环结束",
                }
            )
            success = True
            break

        entry: dict[str, Any] = {
            "attempt": attempt,
            "outcome": "fail",
            "error": meta["type"],
            "action": action,
            "wait_ms": 0,
            "detail": meta["label"],
        }

        if action == "fail":
            entry["detail"] = "确定性失败：立即上抛，不重试"
            timeline.append(entry)
            break

        if action == "fix_prompt":
            prompt_patch = "把报错现场（字段名 / 期望类型 / 实际值）回填进 messages 再试"
            entry["prompt_patch"] = prompt_patch

        if attempt == payload.max_attempts:
            entry["detail"] = "重试次数耗尽：不抛原始异常，交给降级阶梯"
            timeline.append(entry)
            break

        wait = backoff_ms(attempt, payload.base_ms, payload.cap_ms, rng)
        if payload.scenario == "rate_limit":
            entry["retry_after_ms"] = payload.retry_after_ms
            wait = max(wait, payload.retry_after_ms)  # 不尊重 Retry-After 会让限流更狠
        entry["wait_ms"] = wait
        total_wait += wait
        timeline.append(entry)

    if success:
        final = "ok"
    elif action == "fail":
        final = "fail"
    else:
        final = "fallback"

    attempts_used = len(timeline)
    calls = payload.llm_calls + attempts_used
    wall = payload.elapsed_s + total_wait / 1000
    within = calls <= RUN_BUDGET["max_llm_calls"] and wall <= RUN_BUDGET["max_wall_seconds"]

    # 并发脉冲对照：50 个 run 同时进入「第二次重试」，有抖动 vs 无抖动
    sync_wait = min(payload.base_ms * 2, payload.cap_ms)
    sync_stats = _pulse_stats([sync_wait] * 50)
    jitter_stats = _pulse_stats(
        [backoff_ms(2, payload.base_ms, payload.cap_ms, rng) for _ in range(50)]
    )

    return {
        "scenario": payload.scenario,
        "classification": {
            "type": meta["type"],
            "action": action,
            "why": meta["why"],
        },
        "timeline": timeline,
        "summary": {
            "attempts": attempts_used,
            "succeeded": success,
            "total_wait_ms": total_wait,
            "final": final,
            "final_note": {
                "ok": "重试救回来了，用户无感",
                "fail": "确定性失败：立即换路或认输，别再烧钱",
                "fallback": "重试耗尽：不抛原始异常，走降级阶梯（模块 02）",
            }[final],
        },
        "budget": {
            "max_llm_calls": RUN_BUDGET["max_llm_calls"],
            "max_wall_seconds": RUN_BUDGET["max_wall_seconds"],
            "llm_calls_used": calls,
            "wall_seconds_used": round(wall, 2),
            "within_budget": within,
            "note": (
                "单步重试成功 ≠ run 安全：没有 run 级总闸，每步「合法」的重试加起来能拖到几分钟"
            ),
        },
        "pulse": {
            "concurrency": 50,
            "attempt": 2,
            "sync": sync_stats,
            "jittered": jitter_stats,
            "note": (
                f"裸重试：50 个 run 的等待全部落在 {sync_wait}ms，峰值占比 100%（同步脉冲打死上游）；"
                f"带抖动后峰值占比 {jitter_stats['peak_share']:.0%}，被摊到 {jitter_stats['spread_ms']}ms 宽度"
            ),
        },
        "notes": [
            "429 站在被限流的一侧：Day 19 你是限流发起方，今天才知道被限流时重试风暴有多致命。",
            "fix_prompt 是 Agent 特有的重试类别：传统服务重试就是原样再来，Agent 必须把错误现场回填。",
            "重试耗尽不该抛原始异常：用户不需要看到 HTTPError 429，他需要看到一个回答。",
            "单步有上限（3 次），run 也要有总闸（30 次调用 / 180 秒）：两道闸拦的是不同的事。",
        ],
    }


# ============================================================
# 2. 降级阶梯：L0 → L1 → L2 → L3
# ============================================================


def supports_fc(model: str) -> bool:
    """能力对齐白名单：备模型不支持 function calling 时发 tools 会直接 400。"""
    return model in {"gpt-4.1", "gpt-4.1-mini"}


def ladder_demo(payload: Day20LadderRequest) -> dict[str, Any]:
    """按阶梯逐级尝试，返回统一结构：answer / degraded / reason / level。

    纯逻辑（不调模型）：把「主路是否恢复 / 备路是否可用 / 缓存是否命中」作为输入，
    这样同一个降级链可以在单测和前端反复复现，不花一分钱 token。
    """
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    # ---- L0 主模型 ----
    if payload.primary_ok:
        attempts.append(
            {
                "level": 0,
                "model": "gpt-4.1",
                "result": "ok",
                "detail": f"主模型在第 {payload.primary_fail_times + 1} 次调用恢复",
                "degraded": False,
            }
        )
        final = {
            "answer": "（主模型回答，完整质量）",
            "degraded": False,
            "reason": None,
            "level": 0,
        }
    else:
        attempts.append(
            {
                "level": 0,
                "model": "gpt-4.1",
                "result": "failed",
                "detail": f"重试 {payload.primary_fail_times} 次耗尽，上抛",
                "degraded": False,
            }
        )

        # ---- L1 备模型（先过能力对齐）----
        if payload.needs_tools and not payload.fallback_supports_tools:
            attempts.append(
                {
                    "level": 1,
                    "model": "gpt-4.1-mini",
                    "result": "skipped",
                    "detail": "能力对齐失败：备模型不支持 function calling，发 tools 会 400（把瞬态错误换成了确定性错误）",
                    "degraded": False,
                }
            )
        elif payload.fallback_ok:
            attempts.append(
                {
                    "level": 1,
                    "model": "gpt-4.1-mini",
                    "result": "ok",
                    "detail": "备提供商接管（换 base_url，模型能力已对齐）",
                    "degraded": True,
                }
            )
            final = {
                "answer": "（备模型回答，质量略降）",
                "degraded": True,
                "reason": "fallback_to_gpt-4.1-mini",
                "level": 1,
            }
        else:
            attempts.append(
                {
                    "level": 1,
                    "model": "gpt-4.1-mini",
                    "result": "failed",
                    "detail": "备模型同样失败（首次演练最常见的现场：key 根本没配到生产）",
                    "degraded": False,
                }
            )

    # ---- L2 缓存 / L3 道歉 ----
    if final is None:
        if payload.cache_hit:
            attempts.append(
                {
                    "level": 2,
                    "model": "__cache__",
                    "result": "ok",
                    "detail": "语义缓存命中：历史回答 + 「参考历史回答」标注",
                    "degraded": True,
                }
            )
            final = {
                "answer": "（缓存里的历史回答，带「参考历史回答」标注）",
                "degraded": True,
                "reason": "primary_unavailable_cache_hit",
                "level": 2,
            }
        else:
            attempts.append(
                {
                    "level": 2,
                    "model": "__cache__",
                    "result": "failed",
                    "detail": "缓存未命中（或缓存里没有非降级的「已知正确」）",
                    "degraded": False,
                }
            )
            attempts.append(
                {
                    "level": 3,
                    "model": "__apology__",
                    "result": "ok",
                    "detail": "结构化道歉 + 明确后续路径：绝不出裸 500，绝不静默吞掉",
                    "degraded": True,
                }
            )
            final = {
                "answer": (
                    "抱歉，服务暂时不可用（已持续约 X 分钟）。您的问题已记录，"
                    "系统恢复后将优先处理；也可稍后在设置里点击「重试」。"
                ),
                "degraded": True,
                "reason": "all_levels_failed",
                "level": 3,
            }

    # ---- 缓存铁律：降级答案绝不回写 ----
    written = not final["degraded"]
    cache = {
        "rule": "降级答案绝不回写缓存：缓存的是「已知正确」，不是「碰巧有货」",
        "written": written,
        "store": [payload.query] if written else [],
        "second_query": (
            "缓存里存的是这次的非降级回答，后续相似提问可作参考；"
            "若当初把降级答案也写进去，一次故障会被放大成 N 次故障"
        )
        if written
        else (
            "缓存保持为空：这次故障不会污染后续用户；代价是下一个用户仍要自己走一遍阶梯。"
            "反过来，若降级时刻写进缓存，之后每个命中的用户都拿到同一个坏答案"
        ),
    }

    # ---- 探活窗口：连续 N 次成功才回主路，防乒乓 ----
    probe: list[dict[str, Any]] = []
    degraded = True  # 阶梯已经降级，从「不在主路」开始
    consec = 0
    flips = 0
    for step, ok in enumerate(payload.probe_sequence, 1):
        before = degraded
        if not ok:
            consec = 0
            degraded = True
        else:
            consec += 1
            if degraded and consec >= payload.recover_after:
                degraded = False
        if before != degraded:
            flips += 1
        probe.append(
            {
                "step": step,
                "primary_ok": ok,
                "consec_ok": consec,
                "degraded": degraded,
                "note": (
                    "连续成功达标：回到 L0"
                    if ok and not degraded and consec == payload.recover_after
                    else ("失败清零，重新计数" if not ok else "")
                ),
            }
        )

    return {
        "query": payload.query,
        "attempts": attempts,
        "final": final,
        "cache": cache,
        "probe": probe,
        "notes": [
            "每降一级都要显式标注 degraded + reason：前端展示角标、监控统计降级率、回放能区分正常/降级答案。",
            "L1 三个细节：能力对齐、备模型的工具描述要单独调优（弱模型更依赖说明书）、恢复后要能自动回主路。",
            f"探活窗口连续 {payload.recover_after} 次成功才切回，本次状态翻转 {flips} 次；"
            "没有窗口就会在两个都半死不活的模型之间乒乓（Day 05 的乒乓循环在基础设施层的镜像）。",
            "L3 不是失败：它的产物仍然是一个 HTTP 200 + 结构化说明，用户知道发生了什么、下一步做什么。",
        ],
    }


# ============================================================
# 3. 优雅退出：摘流量 → drain → 落盘 → flush
# ============================================================


def _k8s_snippet(grace: float, prestop: float) -> str:
    """生成可直接抄的编排器配置：宽限期必须比 drain 窗口长。"""
    tgp = int(max(grace + 15, 40))
    return (
        "# k8s deployment 片段\n"
        f"terminationGracePeriodSeconds: {tgp}   # 比 drain 窗口({grace:g}s)长，留出收尾余量\n"
        "preStop:\n"
        f'  exec: {{ command: ["sleep", "{int(prestop)}"] }}   # 等 iptables 规则传播完再摘流量\n'
        "# 没有这行 sleep，摘除瞬间的最后一波请求照样打到正在退出的副本上\n"
    )


def shutdown_demo(payload: Day20ShutdownRequest) -> dict[str, Any]:
    """模拟 SIGTERM 之后的完整退出时序（确定性，不依赖真实信号）。"""
    grace = payload.grace_seconds
    timeline: list[dict[str, Any]] = []
    counters: dict[str, Any] = {
        "accepted": len(payload.runs),
        "rejected_503": payload.new_requests,
        "completed": 0,
        "checkpointed": 0,
        "shattered": 0,
        "leaked": 0,
    }

    timeline.append(
        {"t": 0.0, "stage": "SIGTERM", "detail": "① 立即 stop accepting：新请求不再进来，开始 drain 倒计时"}
    )

    # preStop 与规则传播：课程里最常被遗漏的一行
    if payload.prestop_sleep_s < payload.lb_propagation_s:
        leaked = min(payload.new_requests, 3)
        counters["leaked"] = leaked
        timeline.append(
            {
                "t": 0.0,
                "stage": "preStop 不足",
                "detail": (
                    f"preStop sleep {payload.prestop_sleep_s:g}s < 规则传播 {payload.lb_propagation_s:g}s："
                    f"摘除瞬间仍有 {leaked} 个请求打到正在退出的副本"
                ),
            }
        )
    else:
        timeline.append(
            {
                "t": 0.0,
                "stage": "preStop",
                "detail": f"preStop sleep {payload.prestop_sleep_s:g}s ≥ 传播 {payload.lb_propagation_s:g}s：流量入口已摘干净",
            }
        )

    timeline.append(
        {
            "t": 0.1,
            "stage": "503",
            "detail": f"② 已接收未开始的 {payload.new_requests} 个请求返回 503 + Retry-After: 5（调度到别的副本）",
        }
    )

    # drain 窗口：0.5 秒一格，模拟在途 run 逐个结束
    remaining = {run.run_id: run for run in payload.runs}
    t = 0.0
    while t < grace and remaining:
        t = round(t + 0.5, 1)
        finished = [rid for rid, run in remaining.items() if run.remaining_s <= t]
        for rid in finished:
            timeline.append({"t": t, "stage": "drain", "detail": f"{rid} 跑完，正常结束"})
            counters["completed"] += 1
            del remaining[rid]

    if remaining:
        timeline.append(
            {
                "t": round(grace, 1),
                "stage": "宽限期将尽",
                "detail": f"⑤ 对剩余 {len(remaining)} 个 run 做最后处理：有 checkpointer 的落盘，没有的只能放弃",
            }
        )

    # 每个 run 的结局
    outcomes: list[dict[str, Any]] = []
    for run in payload.runs:
        if run.run_id not in remaining:
            outcomes.append(
                {
                    "run_id": run.run_id,
                    "remaining_s": run.remaining_s,
                    "checkpointed": run.checkpointed,
                    "outcome": "completed",
                    "resumable": False,
                    "detail": f"在 {run.remaining_s:g}s 时跑完，用户拿到了完整答案",
                }
            )
        elif run.checkpointed:
            counters["checkpointed"] += 1
            outcomes.append(
                {
                    "run_id": run.run_id,
                    "remaining_s": run.remaining_s,
                    "checkpointed": True,
                    "outcome": "checkpointed",
                    "resumable": True,
                    "detail": (
                        f"还要跑 {run.remaining_s:g}s，超过 {grace:g}s 宽限期："
                        "在最后一个完整的超步边界落盘，进程退出；客户端凭 thread_id 在别的副本 resume"
                    ),
                }
            )
        else:
            counters["shattered"] += 1
            outcomes.append(
                {
                    "run_id": run.run_id,
                    "remaining_s": run.remaining_s,
                    "checkpointed": False,
                    "outcome": "shattered",
                    "resumable": False,
                    "detail": "没有 checkpointer：被 SIGKILL 撕碎，进度全丢，客户端只能带幂等键从头重试",
                }
            )

    if payload.flush_trace:
        timeline.append(
            {"t": round(grace, 1), "stage": "flush", "detail": "⑥ flush trace / 日志：最后一段轨迹保住了"}
        )
    else:
        timeline.append(
            {
                "t": round(grace, 1),
                "stage": "flush 缺失",
                "detail": "⑥ 没 flush：最后一段 trace 随进程消失（Day 15 的坑在新代码里复发）",
            }
        )

    problems: list[str] = []
    if counters["shattered"]:
        problems.append(f"{counters['shattered']} 个无 checkpointer 的 run 被撕碎")
    if counters["leaked"]:
        problems.append("preStop 太短，摘流量有漏")
    if not payload.flush_trace:
        problems.append("trace 尾段丢失")
    verdict = "可通过：摘流量 → 拒新 → drain → 落盘 → flush 五步齐全" if not problems else "需修：" + "；".join(problems)

    return {
        "config": {
            "grace_seconds": grace,
            "prestop_sleep_s": payload.prestop_sleep_s,
            "lb_propagation_s": payload.lb_propagation_s,
            "termination_grace_seconds": int(max(grace + 15, 40)),
            "relation_ok": payload.prestop_sleep_s >= payload.lb_propagation_s,
        },
        "timeline": timeline,
        "runs": outcomes,
        "counters": counters,
        "k8s_snippet": _k8s_snippet(grace, payload.prestop_sleep_s),
        "verdict": verdict,
        "notes": [
            "优雅退出不等于「等所有 run 跑完」：本质是「把没跑完的，变成能续跑的」。",
            "有 checkpointer：状态随时在库里，drain 只需保证当前超步完整落盘；无 checkpointer：drain 就是最大可接受等待。",
            "interrupt 审批中的 run 靠 PostgresSaver 天然安全：进程死了状态还在，只要求 resume 端点在别的副本上可用。",
            "terminationGracePeriod 必须比 drain 窗口长，否则编排器在 drain 中途就 SIGKILL。",
        ],
    }


# ============================================================
# 4. 容量评估：Little's Law + 三道天花板 + 慢压测读数
# ============================================================


def _queue_sim(
    arrival_rps: float,
    latency_s: float,
    concurrency: int,
    duration_s: int,
    max_wait_s: float,
) -> dict[str, Any]:
    """确定性排队模拟：c 个执行位 + 恒定到达率，排队超过 ``max_wait_s`` 直接 503。

    为什么不用真压测：真压测要么需要起服务，要么只能测出自己设的闸门。
    这里把「排队数学」单独拿出来，用来教「读数字做决策」。
    """
    if arrival_rps <= 0 or concurrency <= 0:
        return {
            "arrival_rps": round(arrival_rps, 2),
            "concurrency": concurrency,
            "sent": 0,
            "completed": 0,
            "rejected_503": 0,
            "success_rate": 0.0,
            "throughput_rps": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "avg_wait": 0.0,
            "utilization": 0.0,
        }

    # 每个执行位的「下次空闲时间」
    free = [0.0] * concurrency
    heapq.heapify(free)
    lats: list[float] = []
    waits: list[float] = []
    rejected = 0
    total = int(arrival_rps * duration_s)

    for i in range(total):
        arrived = i / arrival_rps
        ready = heapq.heappop(free)
        wait = max(0.0, ready - arrived)
        if wait > max_wait_s:
            heapq.heappush(free, ready)  # 拒绝不占位：宁可快速失败，不要慢速排队
            rejected += 1
            continue
        heapq.heappush(free, arrived + wait + latency_s)
        waits.append(wait)
        lats.append(wait + latency_s)

    lats.sort()

    def pct(p: float) -> float:
        if not lats:
            return 0.0
        return round(lats[min(len(lats) - 1, int(len(lats) * p))], 2)

    completed = len(lats)
    return {
        "arrival_rps": round(arrival_rps, 2),
        "concurrency": concurrency,
        "sent": total,
        "completed": completed,
        "rejected_503": rejected,
        "success_rate": round(completed / total, 4) if total else 0.0,
        "throughput_rps": round(completed / duration_s, 2),
        "p50": pct(0.5),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "avg_wait": round(sum(waits) / len(waits), 2) if waits else 0.0,
        "utilization": round(arrival_rps * latency_s / max(concurrency, 1), 3),
    }


def capacity_demo(payload: Day20CapacityRequest) -> dict[str, Any]:
    """三道天花板取最小值 → 推导每副本 Semaphore → 恒定速率压测读数。"""
    money_rps = payload.monthly_budget / (30 * 86400) / payload.run_cost
    upstream_rps = payload.rpm / 60 / payload.calls_per_run
    machine_total = payload.replicas * payload.machine_conc_per_replica
    machine_rps = machine_total / payload.latency_s

    caps = {"money": money_rps, "upstream": upstream_rps, "machine": machine_rps}
    binding = min(caps, key=caps.get)
    throughput = caps[binding]
    concurrency = throughput * payload.latency_s
    # 减一个极小量：避免 31.000000000000004 被 ceil 成 32（浮点边界）
    per_replica = max(1, math.ceil(concurrency / payload.replicas * (1 + payload.headroom) - 1e-9))

    naive_throughput = payload.replicas * payload.naive_semaphore / payload.latency_s

    ceilings = {
        "rows": [
            {
                "name": "钱",
                "formula": f"{payload.monthly_budget:g} ÷ (30×86400) ÷ {payload.run_cost:g}",
                "value": round(money_rps, 3),
                "unit": "run/s",
            },
            {
                "name": "上游配额",
                "formula": f"{payload.rpm} RPM ÷ 60 ÷ {payload.calls_per_run:g} 次/run",
                "value": round(upstream_rps, 3),
                "unit": "run/s",
            },
            {
                "name": "机器资源",
                "formula": f"{payload.replicas} 副本 × {payload.machine_conc_per_replica} ÷ {payload.latency_s:g}s",
                "value": round(machine_rps, 3),
                "unit": "run/s",
            },
        ],
        "binding": binding,
        "throughput_rps": round(throughput, 2),
        "concurrency_needed": round(concurrency, 1),
    }

    derivation = [
        {"step": "①", "text": f"单 run 成本 ${payload.run_cost:g}、平均停留 {payload.latency_s:g}s（Day 17 账本）"},
        {"step": "②", "text": f"钱的天花板：${payload.monthly_budget:g}/月 → {money_rps:.2f} run/s"},
        {"step": "③", "text": f"上游的天花板：{payload.rpm} RPM ÷ {payload.calls_per_run:g} 次/run → {upstream_rps:.2f} run/s"},
        {"step": "④", "text": f"机器的天花板：{payload.replicas} 副本 × {payload.machine_conc_per_replica} 并发 → {machine_rps:.2f} run/s"},
        {"step": "⑤", "text": f"取 min(钱, 上游, 机器) = {throughput:.2f} run/s（瓶颈在「{binding}」）"},
        {"step": "⑥", "text": f"Little's Law：并发 = {throughput:.2f} run/s × {payload.latency_s:g}s ≈ {concurrency:.0f} 个在途 run"},
        {"step": "⑦", "text": f"按 {payload.replicas} 副本摊开 + {payload.headroom:.0%} 余量 → 每副本 Semaphore ≈ {per_replica}"},
    ]

    semaphore = {
        "per_replica": per_replica,
        "total": per_replica * payload.replicas,
        "naive": {
            "per_replica": payload.naive_semaphore,
            "throughput_rps": round(naive_throughput, 2),
            "share_of_target": f"{naive_throughput / throughput:.0%}",
        },
        "verdict": (
            f"拍脑袋设 {payload.naive_semaphore} 只能跑出目标吞吐的 {naive_throughput / throughput:.0%}："
            "Semaphore 不是保护闸，是掐脖子的闸"
            if naive_throughput < throughput * 0.9
            else (
                f"当前 Semaphore 允许的吞吐是目标吞吐的 {naive_throughput / throughput:.1f} 倍："
                "超额配置会把预算和上游配额一次性押上桌，提供商抖一下你就先撞限流"
                if naive_throughput > throughput * 2
                else "当前 Semaphore 与目标吞吐基本匹配"
            )
        ),
    }

    arrival = payload.arrival_rps if payload.arrival_rps else throughput
    sim = _queue_sim(
        arrival,
        payload.latency_s,
        per_replica * payload.replicas,
        payload.duration_s,
        payload.max_wait_s,
    )

    if sim["rejected_503"] > 0:
        decision = {
            "verdict": "出现 503",
            "action": "触到闸门：这才是真实容量上限，记下来作为 Semaphore 与告警阈值的依据",
        }
    elif sim["p95"] > 30:
        decision = {
            "verdict": "p95 超标但无 503",
            "action": "Semaphore 太大导致慢速排队：调小闸门，宁可快速拒绝，不要慢速排队（Day 19）",
        }
    else:
        decision = {
            "verdict": "p95 达标且无 503",
            "action": "容量有富余：到达率可以再 +20% 试探，直到出现 503 为止",
        }

    return {
        "ceilings": ceilings,
        "derivation": derivation,
        "semaphore": semaphore,
        "load_test": sim,
        "decision": decision,
        "notes": [
            "三道天花板取最小值：多数人只看机器资源，而 Agent 服务真正的天花板常在钱和上游配额。",
            (
                "课程原文「$3000/月 ÷ $0.009/run ≈ 12.9 run/s」是算术失误（差 100 倍）："
                "按公式只有 0.13 run/s（约 11k run/月）。要让上游 8.33 run/s 先成为瓶颈，月预算需 ≈ $200k，"
                "本实验默认值即按此设置；把预算调回 3000 就能看到「钱先成为天花板」——那也是真话。"
            ),
            "压测要恒定速率慢压，不要全并发猛灌：全开只会测出你自己设的 Semaphore。",
            "出现 503 不是坏事，是读数：它标出了真实容量；全程无 503 的「成功压测」什么都没测到。",
            "成功率 < 99% 且无 503 → 依赖在压测下崩了（提供商限流 / 工具超时）：这才是最有价值的发现，说明降级阶梯没接好。",
        ],
    }


# ============================================================
# 5. SLO 与告警：从 trace 到烧毁率
# ============================================================


def load_runs(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 trace_id 聚合 span → run 级记录（Day 15 的回放树思路）。

    缺少 ``trace_id`` 或字段类型不对的 span 直接跳过：可观测性数据的消费者
    必须假设输入是脏的，否则一条坏行就能让整次告警评估崩掉。
    """
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for span in spans:
        try:
            by_trace.setdefault(str(span["trace_id"]), []).append(span)
        except (KeyError, TypeError):
            continue

    runs: list[dict[str, Any]] = []
    for tid, group in by_trace.items():
        group.sort(key=lambda s: s["ts"])
        runs.append(
            {
                "trace_id": tid,
                "ts": group[-1]["ts"],
                "ok": all(span.get("status") == "ok" for span in group),
                "latency": group[-1]["ts"] - group[0]["ts"],
                "cost": sum(span.get("cost", 0) for span in group),
                "retries": sum(1 for span in group if span.get("event") == "retry"),
                "degraded": any(bool(span.get("degraded")) for span in group),
            }
        )
    return runs


def load_runs_jsonl(lines: list[str]) -> list[dict[str, Any]]:
    """从 JSONL 行读 run（半行日志：跳过不崩溃）。"""
    spans: list[dict[str, Any]] = []
    for line in lines:
        try:
            spans.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return load_runs(spans)


def window_metrics(
    runs: list[dict[str, Any]],
    window_s: int = 300,
    min_sample: int = 50,
) -> dict[str, Any]:
    """滑动窗口指标；样本不足直接返回 ``insufficient_sample``（假告警第一道防线）。

    窗口以数据里最新的 ``ts`` 为「现在」，因此同一份数据可以反复评估、单测可复现。
    """
    if not runs:
        return {"insufficient_sample": True, "n": 0, "window_s": window_s}
    now = max(run["ts"] for run in runs)
    recent = [run for run in runs if now - run["ts"] < window_s]
    if len(recent) < min_sample:
        return {"insufficient_sample": True, "n": len(recent), "window_s": window_s}

    lats = sorted(run["latency"] for run in recent)
    costs = sorted(run["cost"] for run in recent)
    return {
        "insufficient_sample": False,
        "n": len(recent),
        "window_s": window_s,
        "success_rate": round(sum(1 for run in recent if run["ok"]) / len(recent), 4),
        "p95_latency": lats[int(len(lats) * 0.95)],
        "cost_p95": round(costs[int(len(costs) * 0.95)], 6),
        "degraded_rate": round(sum(1 for run in recent if run["degraded"]) / len(recent), 4),
        "retry_rate": round(sum(run["retries"] for run in recent) / len(recent), 4),
    }


def burn_rate(
    metrics: dict[str, Any],
    slo_success: float = 0.99,
    slo_degraded: float = 0.10,
) -> dict[str, float] | None:
    """烧毁率 = 实际坏率 ÷ 允许坏率；> 1 就是在超额消耗错误预算。"""
    if metrics.get("insufficient_sample"):
        return None
    return {
        "error_burn": round((1 - metrics["success_rate"]) / (1 - slo_success), 2),
        "degraded_burn": round(metrics["degraded_rate"] / slo_degraded, 2),
    }


def should_page(burn: dict[str, float], long_burn: dict[str, float] | None = None) -> dict[str, Any]:
    """告警判定：短窗口烧毁率 ≥ 14.4x 立即 page；长窗口 ≥ 6x 开工单。

    只看一个窗口会出两种错：只看短窗口 → 小抖动也叫人；只看长窗口 → 正在燃烧的火
    被平均值稀释。所以「短窗口急 + 长窗口稳」才是 page 的形状，这里简化成两条阈值。
    """
    worst_key = max(burn, key=burn.get)
    worst = burn[worst_key]
    long_worst = max(long_burn.values()) if long_burn else 0.0

    if worst >= 14.4:
        return {
            "severity": "page",
            "reason": f"短窗口烧毁率 {worst:.1f}x ≥ 14.4x：错误预算 1 小时内耗尽",
            "runbook_key": worst_key,
        }
    if long_worst >= 6.0:
        return {
            "severity": "ticket",
            "reason": f"长窗口烧毁率 {long_worst:.1f}x ≥ 6x：错误预算 6 小时内耗尽，开工单",
            "runbook_key": worst_key,
        }
    return {"severity": "ok", "reason": f"烧毁率 {worst:.1f}x，在预算内", "runbook_key": worst_key}


def _synthesize_spans(payload: Day20SloRequest) -> list[dict[str, Any]]:
    """合成与 Day 15 JSONL 同构的 span（每个 run 一条 start + 一条 done，偶尔多一条 retry）。"""
    rng = random.Random(payload.seed)
    now = int(time.time())
    total_s = payload.duration_min * 60
    incident_s = payload.incident_min * 60
    spans: list[dict[str, Any]] = []

    for i in range(payload.runs):
        start = now - total_s + int(total_s * (i + 0.5) / payload.runs)
        in_incident = start >= now - incident_s
        bad = payload.bad_rate if in_incident else payload.baseline_bad
        degraded_rate = payload.degraded_rate if in_incident else payload.baseline_degraded
        p95 = payload.latency_p95_s if in_incident else payload.baseline_p95_s

        ok = rng.random() >= bad
        degraded = rng.random() < degraded_rate
        latency = max(0.5, rng.gauss(p95 * 0.55, p95 * 0.22))
        trace_id = f"{i:08x}{payload.seed:02x}"

        spans.append(
            {
                "trace_id": trace_id,
                "ts": start,
                "step": 0,
                "status": "ok",
                "latency_ms": 0,
                "tokens": 0,
                "cost": 0.0,
                "degraded": False,
                "event": "start",
                "node": "agent",
            }
        )
        if rng.random() < 0.15:
            spans.append(
                {
                    "trace_id": trace_id,
                    "ts": start + 1,
                    "step": 1,
                    "status": "ok",
                    "latency_ms": 900,
                    "tokens": rng.randint(180, 600),
                    "cost": round(rng.uniform(0.0008, 0.003), 6),
                    "degraded": degraded,
                    "event": "retry",
                    "node": "agent",
                }
            )
        spans.append(
            {
                "trace_id": trace_id,
                "ts": start + int(latency),
                "step": 2,
                "status": "ok" if ok else "error",
                "latency_ms": int(latency * 1000),
                "tokens": rng.randint(300, 1200),
                "cost": round(rng.uniform(0.002, 0.012), 6),
                "degraded": degraded,
                "event": "done",
                "node": "agent",
            }
        )

    spans.sort(key=lambda span: span["ts"])
    return spans


def slo_demo(payload: Day20SloRequest) -> dict[str, Any]:
    """从合成 trace 算出短/长两个窗口的指标与烧毁率，并给出告警判定 + runbook。"""
    spans = _synthesize_spans(payload)
    runs = load_runs(spans)

    short_s = 3600
    long_s = payload.duration_min * 60
    windows: list[dict[str, Any]] = []
    burns: dict[int, dict[str, float] | None] = {}
    for window_s in (short_s, long_s):
        metrics = window_metrics(runs, window_s, payload.min_sample)
        burn = burn_rate(metrics)
        burns[window_s] = burn
        windows.append(
            {
                "window_min": window_s // 60,
                "metrics": metrics,
                "burn": burn,
            }
        )

    short_burn = burns[short_s]
    long_burn = burns[long_s]
    if short_burn is None:
        alert = {
            "severity": "insufficient",
            "reason": "短窗口样本不足（最小样本量守门）：不评估、不告警，避免深夜假告警",
            "runbook": [],
        }
    else:
        decision = should_page(short_burn, long_burn)
        alert = {
            "severity": decision["severity"],
            "reason": decision["reason"],
            "runbook": RUNBOOK.get(decision["runbook_key"], []),
        }

    sample_lines = [
        json.dumps(spans[0], ensure_ascii=False),
        json.dumps(spans[1], ensure_ascii=False),
        '{"trace_id": "broken", "ts": 0',  # 故意留半行：load_runs_jsonl 必须跳过
    ]

    return {
        "sample_lines": sample_lines,
        "windows": windows,
        "alert": alert,
        "slo": {
            "success_rate": 0.99,
            "p95_latency_s": 30,
            "degraded_rate": 0.10,
            "note": "阈值、规则都要像 prompt 一样进 git 走评审：换主模型后 p95 SLO 该跟着改",
        },
        "notes": [
            "同一份 trace 的第三次变现：Day 15 用来回放，Day 16 用来评估，今天用来告警。",
            "最值钱的告警是 degraded_rate 而不是 error_rate：阶梯让系统「没报错」，只盯错误率的团队是瞎的。",
            "烧毁率把小流量噪声压下去：深夜 10 个 run 错 1 个不是 10% 事故，而是样本不足。",
            "告警消息必须带 runbook：凌晨三点被叫醒的人不该现场推理。",
        ],
    }


# ============================================================
# 6. 混沌演练：注入 + 预期声明表
# ============================================================


def chaos_demo(payload: Day20ChaosRequest) -> dict[str, Any]:
    """模拟一次注入故障的 run，并把五条预期逐条核对（落空即盲区）。"""
    rng = random.Random(payload.seed)
    llm_calls = max(3, round(payload.steps * 0.6))
    tool_calls = max(1, payload.steps - llm_calls)

    retries = 0
    rate_limited = 0
    timeouts = 0
    degraded_steps = 0
    fallback_used = 0
    apology = 0
    bare_500 = 0

    max_attempts = 4  # 首次调用 + 3 次重试（课程：重试 3 次耗尽）
    for _ in range(llm_calls):
        if payload.chaos_off:
            continue
        failed = False
        for attempt in range(max_attempts):
            if rng.random() < payload.llm_429_prob:
                rate_limited += 1
                failed = True
                if attempt < max_attempts - 1:
                    retries += 1
                continue
            if rng.random() < payload.llm_timeout_prob:
                timeouts += 1
                failed = True
                if attempt < max_attempts - 1:
                    retries += 1
                continue
            failed = False
            break
        if failed:
            degraded_steps += 1
            if payload.fallback_configured:
                fallback_used += 1
            else:
                apology += 1
                # 阶梯保证「没拿到答案」也是个体面的 200，而不是裸 500

    tool_errors = 0
    same_tool_attempts = 0
    if payload.mcp_down and not payload.chaos_off:
        tool_errors = tool_calls
        # 没有短路就连续 3 轮死磕同一个死工具；有短路则第 2 次就放弃
        same_tool_attempts = 2 if payload.tool_short_circuit else 3

    degraded_rate = degraded_steps / llm_calls if llm_calls else 0.0

    simulation = {
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "retries": retries,
        "rate_limited": rate_limited,
        "timeouts": timeouts,
        "fallback_used": fallback_used,
        "apology": apology,
        "degraded_rate": round(degraded_rate, 3),
        "bare_500": bare_500,
        "tool_errors": tool_errors,
        "same_tool_attempts": same_tool_attempts,
        "final_status": "ok" if bare_500 == 0 else "error",
        "sigterm_at_s": payload.kill_at_s,
        "trace_complete": payload.trace_flush,
        "resumed_after_restart": True,
    }

    expectations: list[dict[str, Any]] = [
        {
            "no": 1,
            "inject": "LLM 50% 429，持续 5 分钟",
            "expected": "退避重试消化大部分；degraded_rate ≤ 10%；无裸 500",
            "actual": f"429 {rate_limited} 次 / 重试 {retries} 次 / 降级率 {degraded_rate:.0%} / 裸 500 {bare_500} 个",
            "pass": bare_500 == 0 and degraded_rate <= 0.10,
            "finding": "重试日志没带 trace_id，回放困难 → 修",
        },
        {
            "no": 2,
            "inject": "LLM 持续 429（重试也必失败）",
            "expected": "3 次重试耗尽 → L1 备模型接管；degraded 响应带角标",
            "actual": (
                f"持续失败时 L1 接管全部 {llm_calls} 次调用，响应均带 degraded 标记"
                if payload.fallback_configured
                else f"L1 全程不可用：{llm_calls} 次调用全部落到 L3 结构化道歉（本次 50% 抖动场景里落 L3 的是 {apology} 次）"
            ),
            "pass": payload.fallback_configured,
            "finding": "L1 备模型的 API key 没配到生产环境 → 降级链第一次启用即失败，这就是不演练的下场",
        },
        {
            "no": 3,
            "inject": "拔掉 MCP Server（工具全挂）",
            "expected": "工具报错以错误信息回填（Day 14）；Agent 改走其他路径或告知用户；主服务存活",
            "actual": f"工具错误 {tool_errors} 次；同一死工具连续尝试 {same_tool_attempts} 轮",
            "pass": payload.tool_short_circuit,
            "finding": "Agent 连续 3 轮尝试同一死工具才放弃 → 终止条件里加「同工具连续失败 2 次」短路（Day 05 的硬自旋在基础设施层的镜像）",
        },
        {
            "no": 4,
            "inject": f"{payload.kill_at_s}s 对自己发 SIGTERM",
            "expected": "新请求 503；在途 run 落盘；重启后 resume 成功；trace 完整",
            "actual": f"trace {'完整' if payload.trace_flush else '尾段丢失'}；重启后 resume 成功",
            "pass": payload.trace_flush,
            "finding": "trace 尾段丢失（忘了 flush，Day 15 的坑在新代码里复发）→ 修",
        },
        {
            "no": 5,
            "inject": "恢复主模型",
            "expected": "探活窗口 3 次成功后自动回 L0；无乒乓",
            "actual": "连续 3 次主路成功 → 自动回 L0；期间无反复切换",
            "pass": True,
            "finding": "—",
        },
    ]

    blind_spots = [row["finding"] for row in expectations if not row["pass"] and row["finding"] != "—"]

    return {
        "injection": {
            "llm_429_prob": payload.llm_429_prob,
            "llm_timeout_prob": payload.llm_timeout_prob,
            "mcp_down": payload.mcp_down,
            "kill_at_s": payload.kill_at_s,
            "chaos_off": payload.chaos_off,
            "note": "CHAOS_OFF 打开后注入全路径短路：演练结束必须置 True，并有自动到期",
        },
        "simulation": simulation,
        "expectations": expectations,
        "blind_spots": blind_spots,
        "notes": [
            "演练的核心不是注入，是**预期**：先写下「我预期系统会怎样」，跑完逐条核对。",
            (
                "注意 degraded_rate 的粒度：一次 run 只有 6 次 LLM 调用，1 次降级就是 17%，"
                "直接越过 10% 的 SLO——小样本下要用更长窗口或多个 run 聚合，别拿单次 run 下结论。"
            ),
            "首次演练 5 项预期常有 2~3 项落空（课程原表是 2 项 ❌ + 2 条「通过但发现」）——每一项落空都是在演习场里堵住的线上事故。",
            "频率建议：新机制上线必演练，之后每月一次 + 每次大版本前一次。",
            "演练价值全部兑现在复盘里：24 小时内出复盘，修复项进 issue 追踪。",
        ],
    }


# ============================================================
# 7. 概念数据出口
# ============================================================


def mapping() -> dict[str, Any]:
    """返回 Day 20 静态概念数据（不调模型、不写文件）。"""
    return {
        "error_classes": ERROR_CLASSES,
        "retry_rules": RETRY_RULES,
        "ladder_levels": LADDER_LEVELS,
        "shutdown_steps": SHUTDOWN_STEPS,
        "capacity_ceilings": CAPACITY_CEILINGS,
        "slo_table": SLO_TABLE,
        "alert_principles": ALERT_PRINCIPLES,
        "chaos_redlines": CHAOS_REDLINES,
        "pitfalls": PITFALLS,
        "cheatsheet": CHEATSHEET,
        "runbooks": RUNBOOK,
        "versions": {
            "python": "3.11",
            "fastapi": _version("fastapi"),
            "langgraph": _version("langgraph"),
            "httpx": _version("httpx"),
        },
    }
