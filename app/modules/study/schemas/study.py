"""智能体开发学习接口的 Pydantic 结构。

设计要点：Agent 循环的返回值不是"一个答案"，而是一条**执行轨迹**（steps）——
前端把轨迹渲染成时间线，才能可视化地观察模型每一轮"决策 → 调工具 → 观察"的过程。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolOut(BaseModel):
    """单个工具的说明书（给前端展示用）。"""

    name: str
    description: str
    parameters: dict[str, Any]
    side_effect: str = Field(default="read", description="副作用分级（Day 04）：read/write/destructive")
    require_confirm: bool = Field(default=False, description="是否需要用户确认后才能执行（Day 04 确认门）")


class ToolCallTrace(BaseModel):
    """一次工具调用的记录：模型想调谁、传了什么参数、真实执行结果是什么。"""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str
    duration_ms: int | None = Field(default=None, description="工具执行耗时（毫秒），Day 03 练习 2 的观测指标")
    is_error: bool = Field(default=False, description="工具返回了错误（ok=false，含校验/业务/确认门拦截）")
    error_type: str | None = Field(
        default=None,
        description="错误类型（Day 04）：bad_json/invalid_args/business_rule/need_confirmation/tool_crash/unknown_tool",
    )
    self_heal: bool = Field(
        default=False,
        description="True 表示该工具上一轮报错后模型本轮重新发起调用（自我纠正，Day 03 练习 3）",
    )


class ToolCallLogEntry(BaseModel):
    """工具调用日志条目（Day 03 练习 2）：观察一轮对话真实触发了几次工具执行。"""

    time: str = Field(description="调用时刻（时:分:秒）")
    name: str = Field(description="工具名")
    arguments: dict[str, Any] = Field(default_factory=dict, description="解析后的调用参数")
    duration_ms: int = Field(description="执行耗时（毫秒）")
    is_error: bool = Field(description="是否失败（ok=false）")
    error_type: str | None = Field(default=None, description="错误类型（Day 04 错误契约）")


class ToolSpecCost(BaseModel):
    """单个工具 description 的字符成本（Day 04 练习 2）。"""

    name: str = Field(description="工具名")
    chars: int = Field(description="description 字符数")


class ToolSpecBudget(BaseModel):
    """工具说明书预算（Day 04 练习 2）：说明书随每轮请求重复发送，是隐性 token 成本。"""

    total_chars: int = Field(description="tools 清单序列化后的总字符数")
    limit: int = Field(description="预算上限（字符）")
    over_limit: bool = Field(description="是否超预算")
    top_tools: list[ToolSpecCost] = Field(description="description 最长的 3 个工具")


class AgentStepTrace(BaseModel):
    """Agent 循环的一轮：要么调了工具（tool_calls），要么给出最终回答（content）。"""

    step: int
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    content: str | None = None
    progress_fp: str | None = Field(
        default=None,
        description="该步结果的进展指纹（Day 05）：只看结果不看动作，连续几步相同即判定无进展",
    )


class AgentRunRequest(BaseModel):
    """运行 Function Calling Agent 循环的请求。"""

    question: str = Field(min_length=1, description="用户问题")
    budget_profile: Literal["auto", "simple", "complex"] = Field(
        default="auto",
        description=(
            "Day 05 预算档位：auto（先让模型判任务复杂度）/ simple（6 步）/ complex（20 步）。"
            "档位决定步数、token 与时延三道上限"
        ),
    )
    max_steps: int | None = Field(
        default=None, ge=1, le=30,
        description="显式覆盖档位步数上限（课堂实验用，如设 1 观察循环被砍掉）；不传按档位",
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")
    user_confirmed: bool = Field(
        default=False,
        description=(
            "Day 04 确认门：资金操作（退款）的二次确认。"
            "模型请求确认后，用户在前端勾选并重新发送同一问题即可放行执行"
        ),
    )


class AgentRunResult(BaseModel):
    """Agent 循环的完整轨迹。"""

    steps: list[AgentStepTrace]
    final_answer: str
    session_messages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="本轮实际产生的 assistant/tool 消息（Day 06 会话记忆回写与审计用）",
    )
    stop_reason: str = Field(
        description=(
            "停止原因（Day 05）：completed / repeated_action / no_progress / "
            "max_steps / budget / ungrounded"
        )
    )
    stopped_by_limit: bool = Field(description="True 表示跑满步数上限（stop_reason=max_steps）")
    degraded: bool = Field(
        default=False,
        description="True 表示触发了强制收敛：熔断后用 tools=None 逼模型用已有信息交卷（降级交付）",
    )
    escalated: bool = Field(
        default=False,
        description="True 表示自愈上限熔断（同一工具同类错误连续失败），已升级为人工处理",
    )
    grounded: bool = Field(
        default=True,
        description="False 表示最终答案没有引用任何工具返回的事实（疑似假收敛）",
    )
    grounding_note: str | None = Field(default=None, description="结果校验说明（如已打回重写一次）")
    guard_diagnosis: str | None = Field(
        default=None, description="结构终止的病因摘要（最频繁动作、最近几步结果指纹数量）"
    )
    budget_profile: str = Field(description="实际使用的预算档位：simple / complex")
    budget_desc: str = Field(description="档位对应的三道上限描述")
    tokens: int = Field(description="本次运行累计消耗 token（含分类与强制收敛调用）")
    elapsed_ms: int = Field(description="总耗时（毫秒）")
    model: str = Field(description="实际使用的模型名")


class ReactStepTrace(BaseModel):
    """ReAct 文本协议的一轮：模型原文 + 解析出的动作 + 工具观察结果。"""

    step: int
    raw_text: str = Field(description="模型本轮输出的原文（Thought/Action/...）")
    action_name: str | None = None
    action_args: dict[str, Any] | None = None
    observation: str | None = None
    is_error: bool = Field(default=False, description="True 表示解析失败（格式错误，已要求重试）")


class ReactRunRequest(BaseModel):
    """运行手写 ReAct 循环的请求。"""

    question: str = Field(min_length=1, description="用户问题")
    max_steps: int = Field(default=6, ge=1, le=12, description="最大循环步数")
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")
    session_id: str | None = Field(
        default=None,
        description=(
            "会话 ID：多次请求传同一个 ID 可保留 history 实现'接着上一句继续问'；"
            "不传则后端自动新建会话（返回值里会给出 ID，可拿它继续追问）"
        ),
    )


class ReactRunResult(BaseModel):
    """ReAct 循环的完整轨迹。"""

    steps: list[ReactStepTrace]
    final_answer: str
    stopped_by_limit: bool
    model: str
    session_id: str = Field(description="本次会话 ID，下次请求带上即可继续追问")
    history: str = Field(description="该会话累计的纯文本记忆（用户问题 + 模型输出 + Observation）")
    turns: int = Field(description="该会话已完成的提问轮数")


class ExtractAttemptTrace(BaseModel):
    """结构化抽取的一次尝试：模型原文、解析结果、失败原因、本轮实际使用的输出约束。"""

    attempt: int
    raw_content: str = Field(description="模型本轮输出的原文")
    parsed: dict[str, Any] | None = Field(default=None, description="解析/校验通过的结果")
    error: str | None = Field(default=None, description="解析或 Schema 校验失败的原因（会回填给模型重试）")
    response_format: str = Field(
        description="本轮实际使用的输出约束：none（口头要求）/ json_object / json_schema"
    )
    note: str | None = Field(default=None, description="降级/兜底说明（如服务不支持 json_schema 自动降级）")


class ExtractRunRequest(BaseModel):
    """Day 02 结构化输出实验请求：客户意图提取。"""

    text: str = Field(min_length=1, description="待抽取的用户原话")
    mode: Literal["naive", "json_mode", "json_schema"] = Field(
        default="json_schema",
        description="三级方案：naive 口头要求（反面教材）/ json_mode JSON Mode / json_schema Pydantic Schema",
    )
    max_retries: int = Field(
        default=2, ge=0, le=3,
        description="解析失败后带错误现场重试的次数（naive 模式是反面教材，固定不重试）",
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")


class ExtractRunResult(BaseModel):
    """结构化抽取结果：尝试轨迹 + 最终工单。"""

    mode: str = Field(description="本次使用的抽取方案")
    attempts: list[ExtractAttemptTrace] = Field(description="每次尝试的轨迹（含失败现场）")
    ticket: dict[str, Any] | None = Field(
        default=None, description="最终通过校验的工单；连续失败时为 null，绝不编造"
    )
    success: bool = Field(description="是否成功拿到通过校验的工单")
    model: str = Field(description="实际使用的模型名")


# ---------- Day 06 多轮会话（放在 AgentRunResult 之后，避免前向引用） ----------


class SessionChatRequest(BaseModel):
    """Day 06 多轮会话请求：同一 session_id 的多次调用共享一份记忆。"""

    message: str = Field(min_length=1, description="本轮用户输入")
    session_id: str | None = Field(
        default=None,
        description="会话 ID；传同一个 ID 可继续追问，不传则后端自动新建（返回值里可拿到）",
    )
    budget_profile: Literal["auto", "simple", "complex"] = Field(
        default="auto", description="复用 Day 05 的任务预算档位"
    )
    max_steps: int | None = Field(
        default=None, ge=1, le=30, description="显式覆盖档位步数；不传按档位"
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")
    user_confirmed: bool = Field(default=False, description="Day 04 资金操作确认放行")


class SessionStateOut(BaseModel):
    """短期记忆状态快照（约束锚点 / 事实槽位 / 摘要 / 水位 / 淘汰记录）。"""

    session_id: str
    system: str = Field(description="原始 system prompt（宪法区，永不裁剪）")
    constraints: list[str] = Field(description="用户硬约束锚点（永不裁剪、每轮重放）")
    facts: dict[str, Any] = Field(description="结构化事实槽位（事实不靠摘要）")
    anchors: list[str] = Field(description="摘要质检补回的硬事实")
    summary: str = Field(description="滚动摘要（增量合并）")
    turn: int = Field(description="已完成的对话轮数")
    total_tokens: int = Field(description="会话累计消耗 token")
    recent_rounds: int = Field(description="仍在近轮原文区的轮数")
    dropped: list[dict[str, Any]] = Field(description="淘汰记录（审计用）")
    water_events: list[dict[str, Any]] = Field(description="水位线事件（60/80/95%）")


class SessionChatResult(BaseModel):
    """多轮会话结果：本轮 Agent 执行 + 会话记忆投影后的状态。"""

    session_id: str
    turn: int
    context_tokens: int = Field(description="本轮渲染出的上下文 token 估算（含工具 Schema）")
    water_level: str = Field(description="当前水位：normal/watch/summarize/force_trim")
    constraints: list[str]
    facts: dict[str, Any]
    anchors: list[str]
    summary: str
    recent_rounds: int
    dropped: list[dict[str, Any]]
    water_events: list[dict[str, Any]]
    run: AgentRunResult = Field(description="本轮 Day 05 循环的完整轨迹")


# ---------- Day 07 长期记忆 + RAG（放在最后，Day07ChatResult 引用 AgentRunResult） ----------


class MemoryWriteRequest(BaseModel):
    """写一条长期记忆：同 key 覆盖，绝不追加。"""

    key: str = Field(min_length=1, max_length=100, description="事实键，如 dietary")
    value: str = Field(min_length=1, description="事实值，如 不吃香菜")
    source_turn: int | None = Field(default=None, ge=1, description="来源对话轮次（可追溯）")


class MemoryFactOut(BaseModel):
    """一条长期记忆事实。"""

    key: str
    value: str
    source_turn: int | None = None
    updated_at: str | None = Field(default=None, description="最后更新时间")


class MemoryWriteResult(BaseModel):
    """写入结果：created=True 为新建，False 表示覆盖了旧值。"""

    created: bool
    fact: MemoryFactOut


class MemoryConflictOut(BaseModel):
    """同义键组内的疑似冲突（课程练习 1）。"""

    group: str
    keys: list[str]
    values: list[str]
    suggestion: str


class MemoryProfileOut(BaseModel):
    """某用户的完整画像快照。"""

    user_id: str
    facts: list[MemoryFactOut]
    prompt: str = Field(description="会话开始时注入 system 的画像文本")
    conflicts: list[MemoryConflictOut]


class KnowledgeDocIn(BaseModel):
    """录入 / 重建一篇知识文档。"""

    doc_id: str = Field(min_length=1, max_length=100, description="文档 ID；重复提交则整篇重建")
    title: str = Field(min_length=1, max_length=200, description="文档标题（检索来源标注用）")
    content: str = Field(min_length=1, description="文档正文（Markdown，按标题分节切块）")
    version: str | None = Field(default=None, max_length=50, description="版本 / 时效，如 2026-10 版")
    chunk_size: int = Field(default=400, ge=100, le=2000, description="目标块大小（字）")
    overlap: int = Field(default=60, ge=0, le=500, description="相邻块重叠字数")


class KnowledgeDocOut(BaseModel):
    """文档索引概览（按 doc_id 聚合）。"""

    doc_id: str
    doc_title: str
    doc_version: str | None = None
    chunks: int
    updated_at: Any = Field(default=None, description="最近更新时间")


class KnowledgeIndexResult(BaseModel):
    """建索引结果。"""

    doc_id: str
    chunks: int
    embedding_source: str = Field(description="向量来源：api（真实模型）/ hash-fallback（本地回退）")


class KnowledgeSearchRequest(BaseModel):
    """混合检索请求（只检索，不生成回答）。"""

    query: str = Field(min_length=1, description="查询语句")
    top_k: int = Field(default=3, ge=1, le=20)
    alpha: float = Field(default=0.5, ge=0, le=1, description="向量路权重，1-alpha 给 BM25")
    doc_id: str | None = Field(default=None, description="元数据过滤：只检索该文档")


class KnowledgeHit(BaseModel):
    """一个命中块（含两路排名与 RRF 得分）。"""

    doc_id: str
    doc_title: str
    doc_version: str | None = None
    chunk_index: int
    section: str | None = None
    content: str
    rrf: float
    vec_score: float
    bm25_score: float
    vec_rank: int
    bm25_rank: int
    embedding_source: str | None = None


class Day07ChatRequest(BaseModel):
    """Day07 问答：画像 + RAG + 工具三条管道一起上。"""

    user_id: str = Field(min_length=1, max_length=100, description="用户 ID（长期记忆按它组织）")
    message: str = Field(min_length=1, description="本轮用户输入")
    history: list[dict[str, str]] = Field(
        default_factory=list, description="多轮历史（仅用于 history-aware 查询改写）"
    )
    doc_id: str | None = Field(default=None, description="只检索该文档（元数据过滤）")
    top_k: int = Field(default=3, ge=1, le=10)
    alpha: float = Field(default=0.5, ge=0, le=1)
    budget_profile: Literal["auto", "simple", "complex"] = Field(default="auto")
    max_steps: int | None = Field(default=None, ge=1, le=30)
    model_config_id: int | None = None
    user_confirmed: bool = False


class Day07ChatResult(BaseModel):
    """Day07 问答结果：最终回答 + 三条管道的执行摘要。"""

    answer: str
    user_id: str
    profile: str
    recalled_facts: list[MemoryFactOut]
    rewritten_query: str
    hits: list[KnowledgeHit]
    conflicts: list[dict[str, Any]]
    run: AgentRunResult
    pipeline: dict[str, Any]
