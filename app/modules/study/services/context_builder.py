"""上下文工程（Day 08）：把七种原料按四层结构编排进有限预算。

四层结构（从上到下，对应课程第 1 节）：

1. 身份与指令层：system prompt（角色 / 硬约束 / 格式）+ 工具说明书——**永不裁**；
2. 状态层：短期摘要（Day 06）+ 长期画像（Day 07）——按"条"裁，画像按 updated_at 取最近 N 条；
3. 知识层：RAG 检索块 + 工具实时数据——先去重、再按 U 型曲线摆放、最后按 token 裁剪；
4. 对话层：近期对话历史——按"轮"裁剪，吃剩余的最大份额。

与课程代码的一处刻意差异：课程把"硬约束复述"挂在知识层尾部，这里放到上下文**最末尾**
（对话层之后、当前问题之前）。复述的全部意义是紧贴当前问题，中间隔着几千 token 历史
就会被稀释——「头尾放最关键的」这条原则要贯彻到底。

设计原则：所有"放什么 / 放多少 / 放哪里"的判断都是纯函数、零 I/O，便于单测与
黄金文件回归（课程练习 3）；真实画像与检索由 ``day08_context.py`` 拉取后注入本模块。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from app.modules.study.services import long_memory, rag
from app.modules.study.services.memory import estimate_tokens as est_tokens

# 单个块至少要有这么多 token 才值得占位置（连来源标注都放不下就干脆不装）
MIN_BLOCK_TOKENS = 80
# 约束复述条数上限：复述太多反而互相稀释
MAX_REMINDERS = 3
# 截断标记本身的 token 开销（给 _clip 预留）
CLIP_MARK_TOKENS = 8


# ============================================================
# 1. 预算配置（课程 1.2 的表格直接落地）
# ============================================================


@dataclass
class Budget:
    """四层预算表：比例之和 + reserve = 1。

    预算不是均分的，而是按"信息不可再生性"分配：指令丢了整个 Agent 失忆，
    检索块丢了这一题答错，都不可替代；对话历史可以被摘要压缩（Day 06 已做），
    所以它吃最大的份额。
    """

    total: int = 16000
    reserve: float = 0.09        # 给模型输出留的余量，先扣
    instruction: float = 0.06    # 身份与指令（含 1% 约束复述）
    tools: float = 0.15          # 工具说明书（物理上在指令层，预算单独算）
    state: float = 0.15          # 短期记忆 + 长期画像
    knowledge: float = 0.20      # RAG 15% + 工具实时数据 5%
    history: float = 0.35        # 对话历史

    @property
    def usable(self) -> int:
        """扣掉输出预留后的可用预算。"""
        return int(self.total * (1 - self.reserve))

    def budget_of(self, ratio: float) -> int:
        """按比例折算成 token 预算。"""
        return int(self.usable * ratio)

    def ratios(self) -> list[dict[str, Any]]:
        """预算表快照（前端展示用）。"""
        return [
            {"name": "身份与指令", "ratio": self.instruction, "note": "永不裁"},
            {"name": "工具说明书", "ratio": self.tools, "note": "工具多时做工具检索"},
            {"name": "短期记忆 + 长期画像", "ratio": self.state, "note": "按条裁"},
            {"name": "检索块 + 工具数据", "ratio": self.knowledge, "note": "去重 + 位置学 + 截断"},
            {"name": "对话历史", "ratio": self.history, "note": "按轮裁"},
            {"name": "输出预留", "ratio": self.reserve, "note": "先扣"},
        ]


# ============================================================
# 2. 原料与产出结构
# ============================================================


@dataclass
class ContextRequest:
    """一次编排的全部原料（Day 02~07 的产物都从这里进来）。"""

    system_prompt: str
    user_query: str
    hard_constraints: list[str] = field(default_factory=list)
    short_memory: str = ""
    profile_facts: list[dict[str, Any]] = field(default_factory=list)
    profile_text: str = ""
    rag_blocks: list[dict[str, Any]] = field(default_factory=list)
    tool_data: str = ""
    tools_schema: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Part:
    """层内一个分区（如指令层里的"角色与约束"和"工具说明书"）。"""

    name: str
    tokens: int
    budget: int
    note: str = ""


@dataclass
class Layer:
    """一层编排结果：预算、实际占用、最终文本、层内分区。"""

    key: str
    name: str
    position: str
    budget: int
    used: int
    content: str
    parts: list[Part] = field(default_factory=list)


@dataclass
class BlockTrace:
    """一个检索块的命运：保留 / 去重丢弃 / 裁剪装填 / 预算不足丢弃，以及最终摆放位置。"""

    title: str
    chunk_index: int
    score: float
    action: str
    tokens: int
    text: str
    position: int | None = None
    detail: str = ""


@dataclass
class Check:
    """退化自检清单的一项：问题 + 机器可查信号 + 结论。"""

    no: int
    question: str
    signal: str
    ok: bool
    detail: str


@dataclass
class BuildResult:
    """编排产出：给模型的上下文 + 给人类审查的分层账单与自检报告。"""

    context: str
    total: int
    used: int
    layers: list[Layer] = field(default_factory=list)
    blocks: list[BlockTrace] = field(default_factory=list)
    checks: list[Check] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dump: str = ""

    @property
    def usage_ratio(self) -> float:
        """总用量占比（相对硬上限）。"""
        return round(self.used / self.total, 4) if self.total else 0.0

    def as_dict(self) -> dict[str, Any]:
        """转成接口响应结构（与 schemas 的 ContextBuildResult 对齐）。"""
        return {
            "context": self.context,
            "total_budget": self.total,
            "used_tokens": self.used,
            "usage_ratio": self.usage_ratio,
            "layers": [
                {
                    "key": layer.key,
                    "name": layer.name,
                    "position": layer.position,
                    "budget": layer.budget,
                    "used": layer.used,
                    "content": layer.content,
                    "parts": [
                        {"name": part.name, "tokens": part.tokens, "budget": part.budget, "note": part.note}
                        for part in layer.parts
                    ],
                }
                for layer in self.layers
            ],
            "blocks": [
                {
                    "title": block.title,
                    "chunk_index": block.chunk_index,
                    "score": block.score,
                    "action": block.action,
                    "tokens": block.tokens,
                    "position": block.position,
                    "text": block.text,
                    "detail": block.detail,
                }
                for block in self.blocks
            ],
            "checks": [
                {
                    "no": check.no,
                    "question": check.question,
                    "signal": check.signal,
                    "ok": check.ok,
                    "detail": check.detail,
                }
                for check in self.checks
            ],
            "warnings": list(self.warnings),
            "dump": self.dump,
        }


# ============================================================
# 3. 检索块的预处理：归一化 / 去重 / 位置学 / 裁剪
# ============================================================


def normalize_block(raw: dict[str, Any]) -> dict[str, Any]:
    """把 RAG 命中 / 手填块统一成编排器内部结构。

    RAG 命中用 ``content`` + ``doc_title`` + ``rrf``，手填块用 ``text`` + ``title``
    + ``score``——两套字段在这里收敛，后面的算法只认一套。
    """
    score = raw.get("score")
    if score is None:
        score = raw.get("rrf") or 0.0
    return {
        "text": str(raw.get("text") or raw.get("content") or "").strip(),
        "title": str(raw.get("title") or raw.get("doc_title") or "未命名资料"),
        "chunk_index": int(raw.get("chunk_index") or 0),
        "score": float(score),
        "section": raw.get("section"),
        "version": raw.get("version") or raw.get("doc_version"),
        "style": "clause" if raw.get("style") == "clause" else "narrative",
    }


def _ngrams(text: str, n: int = 3) -> set[str]:
    """字符 n-gram 集合（去掉空白，中文免分词）。"""
    compact = "".join(text.split())
    if not compact:
        return set()
    if len(compact) <= n:
        return {compact}
    return {compact[index : index + n] for index in range(len(compact) - n + 1)}


def similarity(left: str, right: str) -> float:
    """Jaccard 重合度：两块是不是在讲同一件事。"""
    a, b = _ngrams(left), _ngrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(left: str, right: str) -> float:
    """包含度：短块被长块完整覆盖时接近 1.0（只除以较小的那个集合）。"""
    a, b = _ngrams(left), _ngrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def dedup_blocks(
    blocks: list[dict[str, Any]], threshold: float = 0.9
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """极简 MMR：与已保留块高度重合的后来者丢弃，返回 ``(保留, 丢弃)``。

    ``blocks`` 需已按相关度降序——先到先得，保住高分的那一块。

    为什么同时看 Jaccard 和包含度：课程示例里块 6「保修期为两年，电池保修一年。」
    是块 5 的子集，Jaccard 只有 0.6（分母被长块的额外内容撑大），只看 Jaccard
    会漏掉"短块复述长块"这种最常见的重复；包含度对子集关系敏感，能抓住它。
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for block in blocks:
        best_score, best_hit = 0.0, None
        for kept_block in kept:
            score = max(
                similarity(block["text"], kept_block["text"]),
                containment(block["text"], kept_block["text"]),
            )
            if score > best_score:
                best_score, best_hit = score, kept_block
        if best_hit is not None and best_score > threshold:
            dropped.append(
                {**block, "reason": f"与块{best_hit['chunk_index']}重合度 {best_score:.2f}，只留高分那块"}
            )
        else:
            kept.append(block)
    return kept, dropped


def order_blocks_by_position(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 U 型曲线摆放：最重要的放头，次重要的放尾，垫底的填中间。

    ``blocks`` 需已按相关度降序。Lost in the Middle 的结论是模型对开头和结尾
    利用率最高，所以把"留哪几个"（按分数）和"放在哪"（按 U 型）拆成两步：
    先按分数决定留谁，再按位置学决定怎么排。
    """
    if len(blocks) <= 2:
        return list(blocks)
    head = blocks[0]
    tail = blocks[1]
    middle = sorted(blocks[2:], key=lambda block: block["score"])
    return [head, *middle, tail]


def _clip(text: str, max_tokens: int, keep_tail: bool = False) -> str:
    """按 token 预算截断文本，``keep_tail=True`` 时保尾（条款类的例外常在末尾）。

    逐次收缩而不是按字符数直接切：中文≈1 token/字、英文≈0.3，混排时字符数
    和 token 数并不等比，切完再回测一次最稳。
    """
    if not text or max_tokens <= 0:
        return ""
    if est_tokens(text) <= max_tokens:
        return text
    mark_head, mark_tail = "……（已截断）", "（前文已截断）……"
    body = text
    while body and est_tokens(mark_head + body) > max_tokens:
        keep = max(1, int(len(body) * 0.85))
        body = body[-keep:] if keep_tail else body[:keep]
    if not body:
        return ""
    return (mark_tail + body) if keep_tail else (body + mark_head)


def crop_block(block: dict[str, Any], max_tokens: int) -> str:
    """单块装填：先保来源标注，再截正文——信息打折，来源不打折。

    截断方向由 ``style`` 决定：narrative（总-分结构）保头，clause（条款）
    保尾，因为例外条款往往写在最后。
    """
    version = f"（{block['version']}）" if block.get("version") else ""
    section = f"§{block['section']}" if block.get("section") else ""
    source = f"[来源: {block['title']}{version}{section} · 块{block['chunk_index']}] "
    body_budget = max_tokens - est_tokens(source)
    if body_budget <= CLIP_MARK_TOKENS:
        return source.strip()  # 连正文都放不下：只留来源，至少答案有据可查
    body = _clip(block["text"], body_budget, keep_tail=block.get("style") == "clause")
    return source + body


# ============================================================
# 4. 状态层与对话层的裁剪
# ============================================================


def _recency_key(fact: dict[str, Any]) -> tuple[int, dt.datetime]:
    """画像事实的新旧排序键：时间解析失败的老条目排后面（宁可裁它）。"""
    raw = str(fact.get("updated_at") or "")
    for pattern in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return (1, dt.datetime.strptime(raw, pattern))
        except ValueError:
            continue
    return (0, dt.datetime.min)


def prune_profile(
    facts: list[dict[str, Any]], max_tokens: int
) -> tuple[str, list[str]]:
    """画像超预算时按 ``updated_at`` 保留最近 N 条，返回 ``(画像文本, 被裁 key)``。

    课程练习 1：裁画像不是随机丢，也不是按 key 字典序——用户最近说的话最可能
    还是有效的，老事实先让位。输出保持原列表顺序，避免画像变成倒序让人读不懂。
    """
    if not facts:
        return "", []
    header = "【用户长期画像】"
    used = est_tokens(header)
    kept_keys: set[str] = set()
    dropped: list[str] = []
    for fact in sorted(facts, key=_recency_key, reverse=True):
        line = long_memory.fact_line(fact)
        cost = est_tokens(line) + 1
        if used + cost > max_tokens:
            dropped.append(str(fact.get("key", "?")))
            continue
        kept_keys.add(str(fact.get("key", "")))
        used += cost
    lines = [
        long_memory.fact_line(fact)
        for fact in facts
        if str(fact.get("key", "")) in kept_keys
    ]
    if not lines:
        return "", dropped
    return header + "\n" + "\n".join(lines), dropped


def build_reminder(constraints: list[str]) -> str:
    """硬约束复述文本：只复述前 3 条，多了反而互相稀释。"""
    items = [item.strip() for item in constraints if item.strip()]
    if not items:
        return ""
    return "【再次提醒（硬约束，必须遵守）】" + "；".join(items[:MAX_REMINDERS])


def split_rounds(history: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """把消息序列切成"轮"：一条 user + 它后面的全部 assistant/tool 消息。

    裁剪的最小单位是轮而不是条——按条裁最容易留下孤儿的 tool 消息（Day 06 的坑）。
    """
    rounds: list[list[dict[str, Any]]] = []
    for message in history:
        if message.get("role") == "user" or not rounds:
            rounds.append([message])
        else:
            rounds[-1].append(message)
    return rounds


def fit_history(
    history: list[dict[str, Any]], max_tokens: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """从最新一轮往回装，返回 ``(保留消息, 被裁轮次摘要)``。"""
    rounds = split_rounds(history)
    kept_rounds: list[list[dict[str, Any]]] = []
    evicted: list[dict[str, Any]] = []
    used = 0
    for messages in reversed(rounds):
        cost = sum(est_tokens(str(message.get("content") or "")) for message in messages)
        if used + cost > max_tokens:
            preview = ""
            for message in messages:
                if message.get("role") == "user":
                    preview = str(message.get("content") or "")[:40]
                    break
            evicted.append({"messages": len(messages), "preview": preview})
            continue
        kept_rounds.insert(0, messages)
        used += cost
    kept = [message for messages in kept_rounds for message in messages]
    return kept, list(reversed(evicted))


def check_tool_pairing(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """自检 #5 的落地点：丢掉找不到 assistant.tool_calls 的孤儿 tool 消息。

    裁剪留下孤儿 tool 消息，下一轮请求会被 API 直接 400（Day 03 的坑），
    所以在交给模型之前就清掉；反向也检查"有 tool_calls 却没等到响应"的半轮。
    """
    known_ids: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    warnings: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            known_ids |= {call.get("id") for call in (message.get("tool_calls") or [])}
            cleaned.append(message)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id in known_ids:
                cleaned.append(message)
                known_ids.discard(call_id)
            else:
                warnings.append(f"tool 消息 {call_id} 找不到配对的 assistant.tool_calls，已丢弃")
        else:
            cleaned.append(message)
    answered = {message.get("tool_call_id") for message in cleaned if message.get("role") == "tool"}
    for message in cleaned:
        ids = {call.get("id") for call in (message.get("tool_calls") or [])}
        missing = ids - answered
        if ids and missing:
            warnings.append(f"assistant 的 tool_calls {sorted(missing)} 缺少工具响应，下一轮可能 400")
    return cleaned, warnings


# ============================================================
# 5. 编排器主体
# ============================================================


class ContextBuilder:
    """四层编排器：预算审批 → 分层装填 → 七问自检 → dump。

    所有注入都必须走这里（课程"坑 1"：禁止绕过编排器直接拼字符串），
    这样预算才有人管，退化才有信号可查。
    """

    def __init__(self, budget: Budget | None = None) -> None:
        self.budget = budget or Budget()
        self.warnings: list[str] = []

    # ---------- 主入口 ----------

    def build(self, req: ContextRequest, debug: bool = False) -> BuildResult:
        """执行一次编排，返回上下文与分层账单。

        ``debug=True`` 时把最终上下文整体 dump 进结果——上下文 bug 不抛异常，
        只会让回答悄悄变差，肉眼审查 dump 是最有效的听诊器。
        """
        self.warnings = []
        budget = self.budget
        layers: list[Layer] = []

        instruction_layer = self._build_instruction(req)
        layers.append(instruction_layer)

        state_layer = self._build_state(req)
        layers.append(state_layer)

        knowledge_layer, block_traces, placed_blocks, kept_blocks = self._build_knowledge(req)
        layers.append(knowledge_layer)

        # 会话开头注入前先去重：同一句问题既在 history 里又在 user_query 里会被问两遍
        history = list(req.history)
        if history and history[-1].get("role") == "user":
            if str(history[-1].get("content") or "").strip() == req.user_query.strip():
                history.pop()

        used_before_history = (
            instruction_layer.used + state_layer.used + knowledge_layer.used
        )
        reminder = self._build_reminder(req)
        remaining = max(0, budget.usable - used_before_history - est_tokens(reminder))
        history_budget = min(budget.budget_of(budget.history), remaining)
        history_layer, kept_history = self._build_history(history, history_budget)
        layers.append(history_layer)

        # 物理顺序：指令 → 状态 → 知识 → 对话 → 约束复述 → 当前问题
        context = "\n\n".join(
            part
            for part in (
                instruction_layer.content,
                state_layer.content,
                knowledge_layer.content,
                history_layer.content,
                reminder,
            )
            if part
        )
        context += f"\n\nuser: {req.user_query}"

        used_tokens = est_tokens(context)
        if used_tokens > budget.total * 0.9:
            self.warnings.append(
                f"上下文 {used_tokens} 已超过硬上限的 90%（{budget.total}），触发降级"
            )

        checks = self._self_check(
            req, instruction_layer, placed_blocks, kept_blocks, kept_history, used_tokens
        )
        result = BuildResult(
            context=context,
            total=budget.total,
            used=used_tokens,
            layers=layers,
            blocks=block_traces,
            checks=checks,
            warnings=list(self.warnings),
        )
        result.dump = self._dump(result) if debug else ""
        return result

    # ---------- 第 1 层：身份与指令 ----------

    def _build_instruction(self, req: ContextRequest) -> Layer:
        """角色规则 + 工具说明书：这一层永远不裁，超预算只能改写精简。"""
        budget = self.budget
        system_text = req.system_prompt.strip()
        tools_text = req.tools_schema.strip()
        system_budget = budget.budget_of(budget.instruction)
        tools_budget = budget.budget_of(budget.tools)

        parts = [
            Part("角色与硬约束", est_tokens(system_text), system_budget, "永不裁：丢了整个 Agent 失忆")
        ]
        if est_tokens(system_text) > system_budget:
            self.warnings.append(
                f"指令层 {est_tokens(system_text)}/{system_budget} 超预算：请改写精简，不要删条款"
            )

        content = system_text
        if tools_text:
            parts.append(
                Part(
                    "工具说明书",
                    est_tokens(tools_text),
                    tools_budget,
                    "每轮重发的隐性成本；工具多时做工具检索只放 Top-N（Day 22）",
                )
            )
            content += "\n\n【可用工具】\n" + tools_text
            if est_tokens(tools_text) > tools_budget:
                self.warnings.append(
                    f"工具说明书 {est_tokens(tools_text)}/{tools_budget} 超预算：优先精简最长的 description"
                )

        return Layer(
            key="instruction",
            name="第 1 层 · 身份与指令层",
            position="最顶部，永不可裁",
            budget=system_budget + tools_budget,
            used=est_tokens(content),
            content=content,
            parts=parts,
        )

    # ---------- 第 2 层：状态 ----------

    def _build_state(self, req: ContextRequest) -> Layer:
        """短期摘要 + 长期画像：按条裁，画像超预算按 updated_at 保留最近的。"""
        state_budget = self.budget.budget_of(self.budget.state)
        summary_budget = int(state_budget * 0.55)
        profile_budget = state_budget - summary_budget

        short_memory = req.short_memory.strip()
        if short_memory and est_tokens(short_memory) > summary_budget:
            self.warnings.append(
                f"短期摘要 {est_tokens(short_memory)}/{summary_budget} 超预算：生产应回 Day 06 增量压缩，这里只做硬截"
            )
            short_memory = _clip(short_memory, summary_budget)

        profile_text, dropped_keys = prune_profile(req.profile_facts, profile_budget)
        if not profile_text and req.profile_text.strip():
            # 只给了拼好的画像文本（没给结构化 facts）时退化为整体裁剪
            profile_text = _clip(req.profile_text.strip(), profile_budget)
        if dropped_keys:
            self.warnings.append(
                f"画像超预算，按 updated_at 裁掉 {len(dropped_keys)} 条：{'、'.join(dropped_keys)}"
            )

        parts = [
            Part("短期记忆（摘要 + 事实槽位）", est_tokens(short_memory), summary_budget, "Day 06 的产物"),
            Part("长期画像（recall_prompt）", est_tokens(profile_text), profile_budget, "Day 07 的产物；按最近更新优先"),
        ]
        content = "\n\n".join(text for text in (short_memory, profile_text) if text)
        return Layer(
            key="state",
            name="第 2 层 · 状态层",
            position="指令之后、对话之前",
            budget=state_budget,
            used=est_tokens(content),
            content=content,
            parts=parts,
        )

    # ---------- 第 3 层：知识 ----------

    def _build_knowledge(
        self, req: ContextRequest
    ) -> tuple[Layer, list[BlockTrace], list[dict[str, Any]], list[dict[str, Any]]]:
        """检索块：去重 → 按重要度装填 → 按位置学摆放 → 工具数据。

        返回 ``(层, 块轨迹, 已摆放的块, 去重后的候选块)``：后两者给自检 #3/#4 用。
        """
        knowledge_budget = self.budget.budget_of(self.budget.knowledge)
        rag_budget = int(knowledge_budget * 0.75)
        tool_budget = knowledge_budget - rag_budget
        traces: list[BlockTrace] = []

        blocks = [
            block
            for block in (normalize_block(raw) for raw in req.rag_blocks)
            if block["text"]
        ]
        blocks.sort(key=lambda block: block["score"], reverse=True)
        kept, dropped = dedup_blocks(blocks)
        for block in dropped:
            traces.append(
                BlockTrace(
                    title=block["title"],
                    chunk_index=block["chunk_index"],
                    score=round(block["score"], 4),
                    action="去重丢弃",
                    tokens=est_tokens(block["text"]),
                    text=block["text"],
                    detail=block.get("reason", ""),
                )
            )

        # 先按重要度决定"留哪几个"（预算不够时先砍低分块），再按 U 型摆放决定"放哪里"
        chosen: list[dict[str, Any]] = []
        used = 0
        per_block = max(MIN_BLOCK_TOKENS, rag_budget // max(1, len(kept))) if kept else 0
        cropped_count = 0
        for block in kept:
            remaining = rag_budget - used
            if remaining < MIN_BLOCK_TOKENS:
                traces.append(
                    BlockTrace(
                        title=block["title"],
                        chunk_index=block["chunk_index"],
                        score=round(block["score"], 4),
                        action="预算不足丢弃",
                        tokens=est_tokens(block["text"]),
                        text=block["text"],
                        detail="检索块预算已用完，低分块让位",
                    )
                )
                continue
            full = crop_block(block, 10**6)
            allowed = min(per_block, remaining)
            rendered = crop_block(block, allowed)
            was_cropped = est_tokens(full) > allowed
            if was_cropped:
                cropped_count += 1
            chosen.append(
                {
                    **block,
                    "rendered": rendered,
                    "tokens": est_tokens(rendered),
                    "action": "裁剪装填" if was_cropped else "保留",
                    "detail": f"预算 {allowed} tok，原文 {est_tokens(full)} tok" if was_cropped else "",
                }
            )
            used += est_tokens(rendered)

        ordered = order_blocks_by_position(chosen)
        parts_text: list[str] = []
        for position, block in enumerate(ordered, start=1):
            parts_text.append(block["rendered"])
            if position == 1:
                block["slot"] = "头"
            elif position == len(ordered) and len(ordered) > 1:
                block["slot"] = "尾"
            else:
                block["slot"] = "中"
            traces.append(
                BlockTrace(
                    title=block["title"],
                    chunk_index=block["chunk_index"],
                    score=round(block["score"], 4),
                    action=block["action"],
                    tokens=block["tokens"],
                    text=block["text"],
                    position=position,
                    detail=block["detail"] or f"按 U 型曲线摆放（第 {position} 位，槽位：{block['slot']}）",
                )
            )

        if cropped_count:
            self.warnings.append(f"{cropped_count} 个检索块超单块预算被裁剪（详见块轨迹）")

        content = "【检索到的资料】\n" + "\n---\n".join(parts_text) if parts_text else ""

        # 工具实时数据：优先级最高，紧跟在检索块后面（知识层尾部）
        tool_data = req.tool_data.strip()
        if tool_data:
            if est_tokens(tool_data) > tool_budget:
                self.warnings.append(
                    f"工具实时数据 {est_tokens(tool_data)}/{tool_budget} 超预算：应在工具层就 digest（Day 06 二级裁剪）"
                )
            content += "\n\n【工具实时数据】\n" + _clip(tool_data, tool_budget)

        parts = [
            Part(
                "RAG 检索块",
                sum(block["tokens"] for block in chosen),
                rag_budget,
                f"候选 {len(blocks)} 块 → 去重后 {len(kept)} 块 → 最终摆放 {len(chosen)} 块",
            ),
            Part("工具实时数据", est_tokens(tool_data), tool_budget, "实时事实不放进检索索引"),
        ]
        return (
            Layer(
                key="knowledge",
                name="第 3 层 · 知识层",
                position="靠近当前问题（尾部），带来源标注",
                budget=knowledge_budget,
                used=est_tokens(content),
                content=content,
                parts=parts,
            ),
            traces,
            ordered,
            kept,
        )

    # ---------- 约束复述 ----------

    def _build_reminder(self, req: ContextRequest) -> str:
        """硬约束复述：头尾呼应，让 >8K 上下文里的约束不被稀释。"""
        constraints = [item.strip() for item in req.hard_constraints if item.strip()]
        if len(constraints) > MAX_REMINDERS:
            self.warnings.append(
                f"硬约束复述给了 {len(constraints)} 条，超过 {MAX_REMINDERS} 条上限：复述太多反而互相稀释"
            )
        return build_reminder(constraints)

    # ---------- 第 4 层：对话历史 ----------

    def _build_history(
        self, history: list[dict[str, Any]], max_tokens: int
    ) -> tuple[Layer, list[dict[str, Any]]]:
        """按轮从新到旧装填，再跑一次工具配对自检；返回 ``(层, 保留消息)``。"""
        budget = self.budget
        kept, evicted = fit_history(history, max_tokens) if max_tokens > 0 else (history, [])
        if max_tokens <= 0 and history:
            self.warnings.append("前面几层已吃满预算，对话历史被整层裁掉（首轮无历史可裁则属正常）")
            kept, evicted = [], [
                {"messages": len(history), "preview": "预算不足，全部淘汰"}
            ]
        kept, pairing_warnings = check_tool_pairing(kept)
        self.warnings.extend(pairing_warnings)

        if evicted:
            self.warnings.append(
                f"对话历史裁掉 {len(evicted)} 轮（按轮淘汰，不做条级切割）"
            )

        content = "\n".join(
            f"{message.get('role')}: {message.get('content') or ''}" for message in kept
        )
        layer = Layer(
            key="history",
            name="第 4 层 · 对话层",
            position="最底部，紧贴用户当前输入",
            budget=budget.budget_of(budget.history),
            used=est_tokens(content),
            content=content,
            parts=[
                Part(
                    "近期对话原文",
                    est_tokens(content),
                    budget.budget_of(budget.history),
                    f"保留 {len(split_rounds(kept))} 轮 / 淘汰 {len(evicted)} 轮",
                )
            ],
        )
        return layer, kept

    # ---------- 七问自检 ----------

    def _self_check(
        self,
        req: ContextRequest,
        instruction_layer: Layer,
        placed_blocks: list[dict[str, Any]],
        kept_blocks: list[dict[str, Any]],
        kept_history: list[dict[str, Any]],
        used_tokens: int,
    ) -> list[Check]:
        """课程第 3 节的七问自检清单，每问都给一个机器可查的信号。"""
        checks: list[Check] = []
        constraints = [item.strip() for item in req.hard_constraints if item.strip()]

        missing = [item for item in constraints if item not in instruction_layer.content]
        reminder_text = build_reminder(constraints)
        reminder_missing = [
            item for item in constraints[:MAX_REMINDERS] if item not in reminder_text
        ]
        checks.append(
            Check(
                no=1,
                question="硬约束还在上下文里吗？",
                signal="约束关键句同时出现在头部指令层与尾部复述区",
                ok=not missing and not reminder_missing,
                detail=(
                    f"头部缺失：{'；'.join(missing)}" if missing
                    else f"复述区缺失：{'；'.join(reminder_missing)}" if reminder_missing
                    else f"{len(constraints)} 条约束头部写全，尾部复述 {min(len(constraints), MAX_REMINDERS)} 条"
                ),
            )
        )

        rounds = split_rounds(kept_history)
        first_role = rounds[0][0].get("role") if rounds else "user"
        checks.append(
            Check(
                no=2,
                question="有没有裁剪把一轮对话拦腰斩断？",
                signal="保留下来的首条消息必须是 user（按轮裁剪的必然结果）",
                ok=first_role == "user",
                detail=f"首条消息 role={first_role}，保留 {len(rounds)} 轮 / {len(kept_history)} 条消息",
            )
        )

        # 自检 #3 看最终装进上下文的块（去重之后应当没有高度重合的）
        worst, worst_pair = -1.0, ""
        for index, left in enumerate(placed_blocks):
            for right in placed_blocks[index + 1 :]:
                score = max(
                    similarity(left["text"], right["text"]),
                    containment(left["text"], right["text"]),
                )
                if score > worst:
                    worst = score
                    worst_pair = f"块{left['chunk_index']}↔块{right['chunk_index']}"
        checks.append(
            Check(
                no=3,
                question="检索块有没有互相重复？",
                signal="两两重合度报警阈值 0.9",
                ok=worst < 0.9,
                detail=(
                    f"最高重合度 {max(worst, 0.0):.2f}（{worst_pair}）"
                    if worst_pair
                    else "块数不足两块，无对比"
                ),
            )
        )

        # 自检 #4：最高分块必须落在第 1 位（头）；末位应当是次高分块（尾）
        slots = [block.get("slot", "?") for block in placed_blocks]
        if not kept_blocks:
            ok_four, detail_four = True, "本轮没有检索块，无从摆放"
        else:
            top = max(kept_blocks, key=lambda block: block["score"])
            head_chunk = placed_blocks[0]["chunk_index"] if placed_blocks else None
            if head_chunk == top["chunk_index"]:
                ok_four = len(placed_blocks) < 2 or slots[-1] == "尾"
                detail_four = f"槽位：{'、'.join(slots)}；top1=块{top['chunk_index']} 在头部"
            elif all(top["chunk_index"] != block["chunk_index"] for block in placed_blocks):
                ok_four, detail_four = True, f"top1=块{top['chunk_index']} 被预算裁掉，已记入块轨迹"
            else:
                ok_four, detail_four = False, f"top1=块{top['chunk_index']} 没有放在头部（槽位：{'、'.join(slots)}）"
        checks.append(
            Check(
                no=4,
                question="最重要的块在头 / 尾吗？",
                signal="top1 块的实际摆放位置与末位块槽位标记",
                ok=ok_four,
                detail=detail_four,
            )
        )

        pairing_warnings = [item for item in self.warnings if "tool" in item]
        checks.append(
            Check(
                no=5,
                question="tool 消息与 assistant.tool_calls 配对完整吗？",
                signal="逐条校验 tool_call_id",
                ok=not pairing_warnings,
                detail="配对完整" if not pairing_warnings else "；".join(pairing_warnings),
            )
        )

        checks.append(
            Check(
                no=6,
                question="实际 token 数离硬上限还有多少？",
                signal="用量超上限 90% 报警并触发降级",
                ok=used_tokens <= self.budget.total * 0.9,
                detail=f"{used_tokens}/{self.budget.total} tok（{used_tokens / self.budget.total:.1%}）",
            )
        )

        profile_conflicts = long_memory.detect_conflicts(req.profile_facts)
        numeric_conflicts = rag.detect_numeric_conflicts(
            req.user_query,
            [
                {
                    "content": block["text"],
                    "doc_title": block["title"],
                    "chunk_index": block["chunk_index"],
                }
                for block in placed_blocks
            ],
        )
        checks.append(
            Check(
                no=7,
                question="注入的记忆 / 检索内容里有没有互相矛盾的事实？",
                signal="同义键冲突扫描 + 问题金额与资料的数值比对",
                ok=not profile_conflicts and not numeric_conflicts,
                detail=(
                    f"画像冲突 {len(profile_conflicts)} 组、数值冲突 {len(numeric_conflicts)} 处"
                    if profile_conflicts or numeric_conflicts
                    else "未发现自相矛盾"
                ),
            )
        )

        failed = [check.no for check in checks if not check.ok]
        if failed:
            self.warnings.append(f"自检未通过：第 {'、'.join(str(no) for no in failed)} 项")
        return checks

    # ---------- dump ----------

    def _dump(self, result: BuildResult) -> str:
        """把最终上下文 + 分层账单 + 自检结论整体打印出来（开发模式的听诊器）。"""
        lines = [
            f"===== 上下文 dump（{result.used} tok / {result.total}，用量 {result.usage_ratio:.1%}）=====",
            result.context,
            "",
            "===== 分层预算 =====",
        ]
        for layer in result.layers:
            lines.append(
                f"{layer.name}：{layer.used}/{layer.budget} tok（{layer.position}）"
            )
            for part in layer.parts:
                lines.append(f"  - {part.name}: {part.tokens}/{part.budget} tok")
        passed = sum(1 for check in result.checks if check.ok)
        lines.append("")
        lines.append(f"===== 七问自检（{passed}/{len(result.checks)} 通过）=====")
        for check in result.checks:
            flag = "✔" if check.ok else "✘"
            lines.append(f"{check.no} {flag} {check.question} — {check.detail}")
        lines.append("")
        lines.append(f"===== 自检报警 {len(result.warnings)} 条 =====")
        for warning in result.warnings:
            lines.append(f" ⚠ {warning}")
        return "\n".join(lines)
