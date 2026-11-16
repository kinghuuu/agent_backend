"""终止策略（Day 05）：Agent 循环的刹车系统。

设计原则：**纯逻辑、零 I/O、可单测**——不调模型、不读库，时钟由外部注入。
终止逻辑跑在异常路径上，手工测不到，必须能构造任意场景反复验证，
所以它单独成模块，而不是散落在主循环的 if 里。

四层结构（课程 1.3）：
1. 语义终止：模型不再返回 tool_calls（主循环判定，正常出口）；
2. 结构终止：``LoopGuard`` —— 动作指纹抓硬自旋，结果指纹抓变体自旋/乒乓；
3. 资源熔断：``Budget`` —— 步数 / token / 时延，按任务类型分档；
4. 结果校验：``looks_grounded`` —— 最终答案有没有引用工具事实（防假收敛）。

两个关键认知：
- **进展 ≠ 调用了工具，进展 = 上下文里的信息量增加了**，所以判定要看结果指纹；
- 空结果是合法答案（``total: 0``），不是错误——工具返回里的语义提示配合终止逻辑
  才能避免"换个写法再查"的自旋。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable


class StopReason(str, Enum):
    """循环的停止原因：每个值都对应一种可定位的病因，而不是一个数字。"""

    COMPLETED = "completed"          # 模型不再请求工具（正常出口）
    REPEATED_ACTION = "repeated_action"  # 同一动作反复调用（硬自旋）
    NO_PROGRESS = "no_progress"      # 连续多步结果无增量（变体自旋 / 乒乓）
    MAX_STEPS = "max_steps"          # 跑满步数上限
    BUDGET = "budget"                # token / 时延超预算
    UNGROUNDED = "ungrounded"        # 打回后答案仍无事实支撑（假收敛）


@dataclass
class Budget:
    """资源熔断配置（第 3 层）：不管有没有进展，预算花完就不许再花。"""

    max_steps: int = 12
    max_tokens: int = 60_000
    max_seconds: float = 90.0

    def describe(self) -> str:
        """给前端/日志看的档位描述。"""
        return f"steps<={self.max_steps}, tokens<={self.max_tokens}, {self.max_seconds:.0f}s"


# 练习 2：按任务类型分档——简单任务 6 步足够，复杂任务才给 20 步。
# 阈值的正确来源是历史 trace 的 P95（Day 16 评估体系），这里的数字是起点
TASK_BUDGETS: dict[str, Budget] = {
    "simple": Budget(max_steps=6, max_tokens=30_000, max_seconds=60.0),
    "complex": Budget(max_steps=20, max_tokens=80_000, max_seconds=120.0),
}


def _canonical(obj: Any) -> str:
    """稳定序列化：键排序 + 紧凑分隔符，保证同语义同哈希。"""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def action_fp(name: str, args: dict[str, Any]) -> str:
    """动作指纹：工具名 + 规范化参数（同一意图不同写法 → 不同指纹）。"""
    return hashlib.sha1(f"{name}:{_canonical(args)}".encode("utf-8")).hexdigest()[:12]


def progress_fp(results: list[dict[str, Any]]) -> str:
    """进展指纹：**只看结果，不看动作**，排序消除顺序差异。

    这是拦住"变体自旋"的关键：``"上周"``/``"最近7天"``/``"近一周"``
    参数各不相同（动作指纹不同），但返回都是 ``{"total": 0}``（结果指纹相同）。
    """
    norm = sorted(_canonical(result) for result in results)
    return hashlib.sha1("|".join(norm).encode("utf-8")).hexdigest()[:12]


@dataclass
class StepRecord:
    """循环的一步：动作、结果、累计 token 与耗时（轨迹可直接序列化给前端）。"""

    index: int
    actions: list[dict[str, Any]]
    results: list[dict[str, Any]]
    tokens: int
    elapsed: float


@dataclass
class LoopGuard:
    """第 2 层：结构终止——在浪费发生前两三步就停，而不是等步数上限。

    - ``max_same_action``：同一动作（工具+参数指纹）最多调用次数，超过即停；
    - ``max_no_progress``：连续多少步结果指纹不变即判定无进展（实际步数 = +1）；
    - ``repeatable``：允许重复的只读工具白名单（轮询类），重复不加计数、改为退避。
    """

    max_same_action: int = 3
    max_no_progress: int = 2
    repeatable: set[str] = field(default_factory=set)

    _counts: dict[str, int] = field(default_factory=dict)
    _hist: list[str] = field(default_factory=list)
    _backoff: dict[str, int] = field(default_factory=dict)

    def check(
        self, actions: list[dict[str, Any]], results: list[dict[str, Any]]
    ) -> StopReason | None:
        """检查一步是否触发结构终止；返回 None 表示还有信息增量，可以继续。"""
        fps = [action["fp"] for action in actions]

        # ① 单步内完全重复：模型一次吐了两个一样的调用，纯浪费
        if len(fps) != len(set(fps)):
            return StopReason.REPEATED_ACTION

        # ② 跨步动作计数（轮询白名单除外）
        for name, fp in zip([action["name"] for action in actions], fps):
            if name in self.repeatable:
                continue
            self._counts[fp] = self._counts.get(fp, 0) + 1
            if self._counts[fp] > self.max_same_action:
                return StopReason.REPEATED_ACTION

        # ③ 无进展：连续 max_no_progress + 1 步结果指纹完全相同
        # 纯轮询步骤（动作全在白名单）不参与无进展判定：任务连续返回 running 是常态，
        # 真正的兜底是第 3 层预算——否则一个稍慢的导出任务就会被误杀
        all_repeatable = bool(actions) and all(
            action["name"] in self.repeatable for action in actions
        )
        if not all_repeatable:
            self._hist.append(progress_fp(results))
            window = self._hist[-(self.max_no_progress + 1):]
            if len(window) == self.max_no_progress + 1 and len(set(window)) == 1:
                return StopReason.NO_PROGRESS

        return None

    def backoff_seconds(self, name: str) -> float:
        """轮询类工具的重复间隔：1s、2s、4s、8s 封顶（指数退避）。"""
        count = self._backoff.get(name, 0)
        self._backoff[name] = count + 1
        return min(2.0 ** count, 8.0)

    def diagnose(self) -> str:
        """病因摘要：给日志和用户看"为什么停"，而不是一个光秃秃的步数。"""
        parts: list[str] = []
        if self._counts:
            worst_fp, worst_count = max(self._counts.items(), key=lambda kv: kv[1])
            parts.append(f"最频繁动作 {worst_fp} × {worst_count} 次")
        if self._hist:
            recent = self._hist[-3:]
            parts.append(f"最近 {len(recent)} 步结果指纹 {len(set(recent))} 种")
        return "；".join(parts) or "无动作记录"


# ============================================================
# 第 4 层：结果校验（识别假收敛）
# ============================================================

# 可引用的"硬实体"：订单号 / 用户号 / 退款号 / 导出任务号 / 日期。
# 故意不用金额数字（"3" 这种到处都是，误判率高），也不用语义相似度——
# 字符串包含简单、可解释、零成本，误判率足够低
_ENTITY_PATTERNS = (
    re.compile(r"\bO\d{8}\b"),
    re.compile(r"\bU\d{6}\b"),
    re.compile(r"\bR\d{8}\b"),
    re.compile(r"\bT\d{4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
)


def extract_entities(results: Iterable[dict[str, Any]]) -> list[str]:
    """从工具结果里抽取可引用的实体（去重，保持出现顺序）。"""
    text = " ".join(_canonical(result) for result in results)
    found: list[str] = []
    for pattern in _ENTITY_PATTERNS:
        for match in pattern.findall(text):
            if match not in found:
                found.append(match)
    return found


def looks_grounded(answer: str, results: Iterable[dict[str, Any]]) -> bool:
    """第 4 层：最终答案是否引用了工具返回的事实。

    三种信号里选最廉价的一种——"未引用最新结果"：
    - 工具结果里没有任何硬实体 → 不判（纯闲聊不该被打回）；
    - 有实体但答案里一个都没出现 → 大概率在编，打回一次。
    """
    entities = extract_entities(results)
    if not entities:
        return True
    return any(entity in answer for entity in entities)
