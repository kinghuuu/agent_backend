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


# ---------- Day 08 上下文工程（四层编排器） ----------


class ContextBlockIn(BaseModel):
    """一块手工检索原料（没接真实知识库时用它做编排实验）。"""

    text: str = Field(min_length=1, description="块正文")
    title: str = Field(default="手工资料", max_length=200, description="来源标题（注入时标注）")
    chunk_index: int = Field(default=0, ge=0, description="块序号（来源标注用）")
    score: float = Field(default=0.5, description="相关度分数：编排按它决定重要性排序")
    section: str | None = Field(default=None, max_length=200, description="所属章节")
    style: Literal["narrative", "clause"] = Field(
        default="narrative",
        description="文档风格：narrative 按总-分结构保头截尾；clause 条款类保尾截头（例外在末尾）",
    )


class ContextBuildRequest(BaseModel):
    """一次上下文编排请求：手填原料 + 可选的真实数据来源。"""

    user_query: str = Field(min_length=1, description="当前用户问题（永远在上下文最末尾）")
    user_id: str | None = Field(
        default=None, description="传了就读真实长期画像当状态层；不传用手填 profile_text 或演示事实"
    )
    profile_text: str = Field(default="", description="手填画像文本（没有 user_id 时使用）")
    short_memory: str = Field(default="", description="Day 06 的短期摘要（不传用示例摘要）")
    rag_query: str | None = Field(
        default=None, description="传了就真实混合检索知识库；不传用手填 rag_blocks 或课程演示块"
    )
    doc_id: str | None = Field(default=None, description="限定只检索该文档")
    top_k: int = Field(default=4, ge=1, le=10, description="真实检索的候选块数")
    alpha: float = Field(default=0.5, ge=0, le=1, description="检索的向量路权重")
    rag_blocks: list[ContextBlockIn] = Field(
        default_factory=list, description="手填检索块（含重复块可观察去重）"
    )
    tool_data: str = Field(default="", description="工具实时数据（已 digest）")
    history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="历史消息 [{role, content, tool_calls?, tool_call_id?}]；可塞孤儿 tool 消息观察自检",
    )
    hard_constraints: list[str] = Field(
        default_factory=list, description="需要在尾部复述的硬约束（超过 3 条会报警）"
    )
    system_prompt: str | None = Field(default=None, description="不传用课程默认客服规则")
    with_tools: bool = Field(default=True, description="是否注入真实工具说明书（观察隐性成本）")
    total_budget: int = Field(default=16000, ge=1000, le=128000, description="上下文硬上限（token）")
    debug: bool = Field(default=True, description="是否产出整体 dump（开发模式的听诊器）")


class ContextPartOut(BaseModel):
    """层内一个分区的预算账单。"""

    name: str
    tokens: int = Field(description="实际占用 token")
    budget: int = Field(description="该分区预算 token")
    note: str = ""


class ContextLayerOut(BaseModel):
    """一层编排结果：预算、占用、最终文本。"""

    key: str = Field(description="instruction / state / knowledge / history")
    name: str
    position: str = Field(description="该层在上下文里的位置说明")
    budget: int
    used: int
    content: str
    parts: list[ContextPartOut] = Field(default_factory=list)


class ContextBlockTraceOut(BaseModel):
    """一个检索块的命运（去重 / 裁剪 / 摆放位置）。"""

    title: str
    chunk_index: int
    score: float
    action: str = Field(description="保留 / 裁剪装填 / 去重丢弃 / 预算不足丢弃")
    tokens: int
    text: str
    position: int | None = Field(default=None, description="在知识层的摆放位次（1 = 头）")
    detail: str = ""


class ContextCheckOut(BaseModel):
    """退化自检清单的一项。"""

    no: int
    question: str
    signal: str = Field(description="机器可查的信号")
    ok: bool
    detail: str


class ContextBuildResult(BaseModel):
    """编排结果：给模型的上下文 + 给人类审查的账单与自检报告。"""

    context: str
    total_budget: int
    used_tokens: int
    usage_ratio: float
    layers: list[ContextLayerOut]
    blocks: list[ContextBlockTraceOut] = Field(description="检索块的裁剪 / 摆放轨迹")
    checks: list[ContextCheckOut] = Field(description="七问自检结果")
    warnings: list[str] = Field(description="预算超限 / 裁剪 / 配对异常等报警")
    dump: str = Field(description="可直接肉眼审查的整体 dump（debug=true 时非空）")
    sources: dict[str, str] = Field(
        default_factory=dict, description="每样原料的真实来源说明（真实库 / 手填 / 演示）"
    )
    profile_text: str = Field(default="", description="本次使用的完整画像文本（未裁剪，便于对比）")


# ---------- Day 09 LangChain 初探 ----------


class Day09CustomerReply(BaseModel):
    """结构化抽取的合同（练习 1）：with_structured_output 直接吃这个模型。

    每个 ``Field(description=...)`` 都会被转成 JSON Schema 下发——
    它就是 Day 04 说的"参数说明书"，写得越具体抽取越稳。
    """

    reply: str = Field(description="给用户的回复，一句话给出关键结论")
    sentiment: Literal["positive", "neutral", "negative"] = Field(description="用户情绪")
    needs_escalation: bool = Field(description="是否需要转人工（投诉/赔偿/监管风险时为 true）")


class Day09PromptRequest(BaseModel):
    """Prompt 模板演示请求。"""

    company: str = Field(default="极星科技", max_length=50, description="partial 冻结的公司名")
    question: str = Field(default="XR-500 Pro 退款要运费吗？", min_length=1, description="模板变量 question")
    history: list[dict[str, Any]] = Field(
        default_factory=list, description="占位到 placeholder 的历史消息 [{role, content}]"
    )
    omit_variable: bool = Field(
        default=False, description="故意漏填 history，观察模板的 fail-fast（课程轨迹二）"
    )


class Day09StructuredRequest(BaseModel):
    """结构化输出请求：可重复跑同一输入，观察字段抖动（练习 1）。"""

    text: str = Field(
        min_length=1,
        description="待抽取的用户原话",
        examples=["你们客服态度太差了，我要投诉！订单 SO20261101001 到现在还没发货"],
    )
    repeat: int = Field(default=1, ge=1, le=3, description="重复次数：>1 可对比同一输入的字段稳定性")
    temperature: float = Field(default=0.0, ge=0, le=2, description="坑 4：>0 时结构化抽取会抖")
    method: Literal["auto", "default", "function_calling", "json_mode"] = Field(
        default="auto",
        description="结构化输出的下发方式：auto 按 json_schema → function_calling → json_mode 自动降级",
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")


class Day09LcelRequest(BaseModel):
    """LCEL 管道演示请求。"""

    question: str = Field(default="什么是 Function Calling？", min_length=1)
    mode: Literal["invoke", "stream", "batch", "retry", "lambda"] = Field(
        default="invoke",
        description="invoke 单次 / stream 流式 / batch 并发批量 / retry 重试（纯本地） / lambda 管道插函数",
    )
    batch_questions: list[str] = Field(
        default_factory=list, description="batch 模式的输入列表（留空用课程三条）"
    )
    compare_serial: bool = Field(
        default=False, description="batch 模式下再串行跑一遍，对比并发与串行耗时"
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")


class Day09AgentRequest(BaseModel):
    """框架版 Agent 循环请求。"""

    question: str = Field(
        default="订单 SO20261101001 发货了吗？", min_length=1, description="用户问题"
    )
    variant: Literal["full", "vague", "missing"] = Field(
        default="full",
        description="工具 description 变体：full 写全 / vague 只剩『查询。』 / missing 无 docstring（定义即报错）",
    )
    max_steps: int = Field(
        default=6, ge=1, le=15, description="步数保险丝：框架不自带终止条件，必须自己装"
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")


class Day09MessagesResult(BaseModel):
    """消息抽象演示结果。"""

    messages: list[dict[str, Any]] = Field(description="四类消息：type / openai_role / content / note")
    openai_payload: list[dict[str, Any]] = Field(description="convert_to_openai_messages 的输出")
    roundtrip_note: str
    dict_vs_typed: list[dict[str, str]] = Field(description="裸 dict 与类型对象的差别对照")
    raw_typo: dict[str, Any] = Field(description="反例：role 拼成 assitant 仍能构造")
    typed_error: str = Field(description="正例：ToolMessage 缺 tool_call_id 在构造时就校验失败")
    extra_ignored: str = Field(description="补充：多传的未知 kwargs 会被 pydantic 静默忽略")
    part_mapping: list[dict[str, str]] = Field(description="手写零件 ↔ 框架零件对照表")


class Day09PromptResult(BaseModel):
    """Prompt 模板演示结果。"""

    input_variables: list[str] = Field(description="模板要求的变量（漏填会 KeyError）")
    optional_variables: list[str] = Field(
        default_factory=list, description="placeholder 变量：缺失即跳过，不参与必填校验"
    )
    partial_frozen: list[str] = Field(description="被 partial 冻结的变量")
    messages: list[dict[str, Any]] = Field(description="填充后的消息（type / content / source）")
    error: str | None = Field(default=None, description="漏填变量时捕获到的错误原文")
    error_hint: str
    part_mapping: list[dict[str, str]]


class Day09StructuredResult(BaseModel):
    """结构化输出结果：合同 + 每次尝试 + 抖动字段。"""

    schema_sent: dict[str, Any] = Field(description="下发给模型的 JSON Schema（Field description 就在里面）")
    attempts: list[dict[str, Any]] = Field(description="每次尝试：三字段 + 实际方式 + 耗时 + 错误")
    unstable_fields: list[str] = Field(description="重复跑时取值不一致的字段（temperature>0 才会出现）")
    method_used: str = Field(description="最终生效的下发方式：default / function_calling / json_mode")
    methods_tried: list[str] = Field(description="降级轨迹（失败现场）")
    model: str
    elapsed_ms: int
    note: str
    part_mapping: list[dict[str, str]]


class Day09LcelResult(BaseModel):
    """LCEL 管道结果。"""

    mode: str
    chain: list[dict[str, Any]] = Field(description="管道上的每个组件与职责（竖线的左边到右边）")
    answer: str
    chunks: list[str] = Field(default_factory=list, description="stream 模式的分片")
    chunk_count: int = 0
    elapsed_ms: int
    batch: list[dict[str, Any]] = Field(default_factory=list, description="batch 模式每个问题的回答")
    batch_ms: int | None = Field(default=None, description="并发批量耗时")
    serial_ms: int | None = Field(default=None, description="串行对照耗时")
    retry_trace: list[dict[str, Any]] = Field(default_factory=list, description="with_retry 的重试轨迹")
    note: str
    part_mapping: list[dict[str, str]]


class Day09AgentResult(BaseModel):
    """框架版 Agent 循环轨迹。"""

    steps: list[dict[str, Any]] = Field(description="每步：tool_calls（id/name/args/result/is_error）或 content")
    final_answer: str
    stop_reason: str = Field(description="completed / max_steps / tool_schema_error")
    tool_schemas: list[dict[str, Any]] = Field(description="bind_tools 真实下发的 schema（description 权重看这里）")
    variant_note: str
    messages_count: int = Field(description="messages 累计条数（含 ToolMessage 回填）")
    model: str
    elapsed_ms: int


class Day09MappingResult(BaseModel):
    """Day 09 静态概念数据（对照表 / 预设问题 / 依赖版本）。"""

    part_mapping: list[dict[str, str]]
    dict_vs_typed: list[dict[str, str]]
    agent_presets: list[str]
    install_hint: str
    langchain_version: str


# ---------- Day 10 LangGraph 状态图 ----------


class Day10Intent(BaseModel):
    """意图分类合同（练习 1）：直接交给 with_structured_output。

    ``confidence`` 是关键：路由函数用它在"答得动"和"转人工"之间做选择。
    ``Literal`` 锁死取值范围，是为了让条件边的路由表不会漏 key——类型层面的约束
    比运行时兜底便宜。
    """

    intent: Literal["order", "policy", "chat"] = Field(
        description="order=查订单物流；policy=问退款/保修政策；chat=闲聊"
    )
    confidence: float = Field(ge=0, le=1, description="对本次判断的把握，0~1；拿不准就给低分")
    order_id: str = Field(default="", description="若为 order，提取订单号，形如 SO20261101001")


class Day10MinGraphRequest(BaseModel):
    """最小线性图请求。"""

    question: str = Field(
        default="XR-500 Pro 保修多久？",
        min_length=1,
        description="用户问题（知识库里只有『保修』『退款』两个关键词能命中）",
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")


class Day10ConditionalRequest(BaseModel):
    """条件边路由请求。"""

    questions: list[str] = Field(
        default_factory=lambda: ["SO20261101001 到哪了？", "能退款吗？", "你们几点上班？"],
        description="依次跑的问题列表，每个问题独立走一遍图",
    )
    threshold: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="练习 1 的开关：置信度低于该值就走『转人工』分支（调高它就能看到 handoff）",
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")


class Day10ReactRequest(BaseModel):
    """完整 ReAct 环请求。"""

    question: str = Field(
        default="帮我查下 SO20261101001 到哪了，顺便说下 XR-500 Pro 的退款政策",
        min_length=1,
    )
    recursion_limit: int = Field(
        default=12,
        ge=2,
        le=40,
        description="物理保险丝：数的是**超步**（agent→tools→agent 一次往返吃 3 个），默认 25",
    )
    step_limit: int = Field(
        default=6,
        ge=1,
        le=20,
        description="语义刹车（练习 2）：agent 步数到顶就转 degrade 节点，别等 recursion_limit 硬停",
    )
    tool_mode: Literal["ok", "broken"] = Field(
        default="ok",
        description="broken 让工具永远返回错误——用它验证『6 步后是否优雅收尾』",
    )
    as_subgraph: bool = Field(
        default=False,
        description="练习 3：把 agent+tools 打包成子图，外层图用条件边判断子图是否收敛",
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")


class Day10ParallelRequest(BaseModel):
    """并行扇出请求（纯本地假检索，不花 token）。"""

    query: str = Field(
        default="退款；订单物流；发票政策",
        min_length=1,
        description="检索词；用分号分隔的子查询决定 Send 的分支数量",
    )


class Day10GraphStructure(BaseModel):
    """可渲染的图结构：节点、边（含条件边标注）、mermaid 源码、ascii 图。"""

    nodes: list[str]
    edges: list[dict[str, Any]] = Field(description="source / target / conditional / label")
    mermaid: str = Field(description="draw_mermaid() 源码，可直接贴进文档或 PR")
    ascii: str = Field(default="", description="draw_ascii() 结果（需要额外装 grandalf）")
    ascii_hint: str = Field(default="", description="ascii 画不出来时的原因/安装提示")


class Day10StateExperimentOut(BaseModel):
    """一个 reducer 对照实验的完整现场。"""

    key: str
    title: str
    state_def: str = Field(description="这个实验的 State 字段声明（唯一的差别就在这一行）")
    patches: list[dict[str, Any]] = Field(description="逐节点：received（读到的状态）/ patch（交的补丁）")
    result: Any = Field(description="图跑完的最终状态")
    verdict: str = Field(description="结论：覆盖 / 追加 / 同 id 更新 / 重复追加")


class Day10StateResult(BaseModel):
    """01 状态与 reducer 实验（不调模型）。"""

    experiments: list[Day10StateExperimentOut]
    reducer_table: list[dict[str, str]]
    triple_pieces: list[dict[str, str]]
    takeaway: str


class Day10MinGraphResult(BaseModel):
    """02 最小线性图结果。"""

    graph: Day10GraphStructure
    events: list[dict[str, Any]] = Field(description="updates（补丁）与 values（完整状态）交替的多条事件")
    trace: list[str] = Field(description="靠 operator.add 累积的审计字段")
    docs: list[str]
    answer: str
    model: str
    elapsed_ms: int
    note: str
    while_mapping: list[dict[str, str]] = Field(description="手写 while ↔ 图 的逐行对照表")


class Day10ConditionalResult(BaseModel):
    """03 条件边路由结果。"""

    graph: Day10GraphStructure
    runs: list[dict[str, Any]] = Field(description="每个问题：intent / confidence / path / answer")
    threshold: float
    model: str
    elapsed_ms: int
    note: str


class Day10ReactResult(BaseModel):
    """04 完整 ReAct 环结果。"""

    graph: Day10GraphStructure
    events: list[dict[str, Any]] = Field(
        description="每个节点提交了什么（scope 为子图命名空间，as_subgraph=true 时才有值）"
    )
    final_answer: str
    stop_reason: str = Field(description="completed / degraded / recursion_limit / unfinished")
    degraded: bool
    recursion_limit: int
    step_limit: int
    outer_steps: int = Field(description="外层超步数：recursion_limit 数的就是它")
    as_subgraph: bool
    tool_mode: str
    messages_count: int
    model: str
    elapsed_ms: int
    note: str
    while_mapping: list[dict[str, str]]


class Day10ParallelResult(BaseModel):
    """05 并行扇出 / 汇聚结果（静态 + Send 动态 + 子图）。"""

    static: dict[str, Any]
    dynamic: dict[str, Any]
    subgraph: dict[str, Any]
    note: str
    send_rules: list[str]


class Day10MappingResult(BaseModel):
    """Day 10 静态概念数据（三件套 / reducer / 对照表 / 六个坑 / 版本）。"""

    triple_pieces: list[dict[str, str]]
    reducer_table: list[dict[str, str]]
    while_vs_graph: list[dict[str, str]]
    stream_modes: list[dict[str, str]]
    pitfalls: list[dict[str, str]]
    send_rules: list[str]
    presets: dict[str, list[str]]
    install_hint: str
    langgraph_version: str


# ---------- Day 11 Checkpointer 与人机协同 ----------


class Day11BasicRequest(BaseModel):
    """课程 01 的三个实验：续跑 / 隔离 / 快照。"""

    first_message: str = Field(default="我叫小胡，最喜欢的数字是 7", min_length=1)
    second_message: str = Field(default="我叫什么？最喜欢的数字是多少？", min_length=1)
    isolation_question: str = Field(
        default="我叫什么？", min_length=1, description="换一个 thread_id 后问同一问题，验证隔离"
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")


class Day11BasicResult(BaseModel):
    """三个实验的完整现场：两轮回答 + 隔离回答 + 两边的快照 + 历史帧。"""

    thread_id: str
    other_thread_id: str
    first_reply: str
    second_reply: str = Field(description="同 thread 第二轮：应当记得第一轮的名字与数字")
    isolated_reply: str = Field(description="另一个 thread：应当表示不知道")
    turns: int = Field(description="第二轮结束后的 turns：2 说明 state 没被重置")
    messages_count: int = Field(description="第二轮结束后的消息条数：4 说明历史没丢")
    snapshot: dict[str, Any] = Field(description="最近一帧：values / next / tasks / created_at")
    other_snapshot: dict[str, Any]
    frames: list[dict[str, Any]] = Field(description="thread A 的历史帧（新的在前）")
    graph: Day10GraphStructure
    notes: list[str]
    model: str
    elapsed_ms: int


class Day11ChatRequest(BaseModel):
    """交互式续聊：同一 thread_id 的多次请求共享存档。"""

    message: str = Field(min_length=1, description="本轮用户输入")
    thread_id: str | None = Field(
        default=None, description="会话 ID；不传则新建（返回值里给出，可继续追问）"
    )
    reset: bool = Field(default=False, description="先删掉该 thread 的全部存档再跑（重置会话）")
    model_config_id: int | None = None


class Day11ChatResult(BaseModel):
    """一轮续聊结果：回答 + 计数 + 快照 + 历史帧。"""

    thread_id: str
    reply: str
    turns: int
    messages_count: int
    is_new_thread: bool
    snapshot: dict[str, Any]
    frames: list[dict[str, Any]]
    model: str
    elapsed_ms: int


class Day11ApprovalStartRequest(BaseModel):
    """发起一次需要人工审批的操作。"""

    question: str = Field(
        default="订单 A1001 不想要了，请直接帮我退款 299 元",
        min_length=1,
        description="要明确要求退款，模型才会调用敏感工具 refund_order",
    )
    thread_id: str | None = Field(default=None, description="不传则新建；恢复时必须用同一个")
    model_config_id: int | None = None


class Day11ApprovalStartResult(BaseModel):
    """第一次调用：图停在 approval（或没有敏感调用直接跑完）。"""

    thread_id: str
    paused: bool = Field(description="True 表示命中 interrupt，图已冻结并存档")
    next: list[str] = Field(description="接下来要跑的节点：暂停时是 ('approval',)")
    interrupt: dict[str, Any] | None = Field(default=None, description="interrupt 发出的待审批载荷")
    answer: str = Field(default="", description="未暂停时的最终回答")
    snapshot: dict[str, Any]
    graph: Day10GraphStructure
    note: str
    model: str
    elapsed_ms: int


class Day11ApprovalResumeRequest(BaseModel):
    """人工审批结果：true 放行 / false 拦截 / 对象表示改参数后放行（练习 2）。"""

    thread_id: str = Field(min_length=1)
    decision: bool | dict[str, Any] = Field(
        default=True,
        description="true / false / {'amount': 199}——对象会覆盖模型原始参数",
    )
    model_config_id: int | None = None


class Day11ApprovalResumeResult(BaseModel):
    """恢复后的结果：最终回答 + 工具是否执行 + 被改过的参数。"""

    thread_id: str
    approved: bool
    patched_args: dict[str, Any] = Field(description="人工改过的参数（空 dict 表示没改）")
    final_answer: str
    tool_results: list[dict[str, Any]] = Field(description="每个 ToolMessage：工具名 / 内容 / 是否被拦截")
    next: list[str] = Field(description="恢复后应当为空（图跑完了）")
    messages_count: int
    note: str
    model: str
    elapsed_ms: int


class Day11PendingOut(BaseModel):
    """一条待审批记录（练习 3 的审批队列）。"""

    thread_id: str
    question: str
    tool: str
    args: dict[str, Any]
    wait_seconds: float = Field(description="已等待秒数：超过 86400 即为超时")
    created_at: str | None = None


class Day11ExpireRequest(BaseModel):
    """模拟定时任务扫描超时审批并自动驳回。"""

    thread_id: str | None = Field(default=None, description="只处理该会话；不传处理全部符合条件者")
    max_age_seconds: float = Field(
        default=0,
        ge=0,
        description="等待超过该秒数即视为超时；0 = 课堂演示（队列里的都当已超时），生产填 86400",
    )
    model_config_id: int | None = None


class Day11ExpireResult(BaseModel):
    """自动驳回结果。"""

    expired: list[dict[str, Any]] = Field(description="每个会话：等待时长 + 自动驳回后的回答")
    remaining: int = Field(description="处理完后仍在等待审批的会话数")
    note: str


class Day11ReplayResult(BaseModel):
    """节点重放与副作用对照实验（纯本地）。"""

    experiments: list[dict[str, Any]] = Field(description="反面 / 正面：暂停前后副作用执行次数")
    rules: list[str]
    note: str


class Day11TimeTravelRequest(BaseModel):
    """时间旅行：换主题后从 answer 之前的那一帧重跑。"""

    topic: str = Field(default="RAG", min_length=1, description="第一次运行的主题")
    new_topic: str = Field(default="MCP 协议", min_length=1, description="改写后重跑的主题")
    model_config_id: int | None = None


class Day11TimeTravelResult(BaseModel):
    """时间旅行结果：两个回答 + 分叉前后的帧列表。"""

    thread_id: str
    original_answer: str
    new_answer: str
    frames_before: list[dict[str, Any]]
    frames_after: list[dict[str, Any]] = Field(description="分叉后：新旧两条链的帧都在")
    fork_checkpoint_id: str = Field(description="从哪一帧分叉（answer 之前）")
    original_preserved: bool = Field(description="原历史的最后一帧是否仍在（分叉不是覆盖）")
    note: str
    model: str
    elapsed_ms: int


class Day11SqliteRequest(BaseModel):
    """SQLite 落盘实验：两次请求等价于两次进程。"""

    mode: Literal["first", "resume"] = Field(
        default="first", description="first 写入一轮 / resume 只带新消息，历史从文件读"
    )
    thread_id: str = Field(default="sqlite-user-42", min_length=1)
    message: str | None = Field(default=None, description="不传用课程预设的两句话")
    reset: bool = Field(default=False, description="先删除该 thread 的落盘存档（重新演示）")
    model_config_id: int | None = None


class Day11SqliteResult(BaseModel):
    """SQLite 实验的返回：落盘位置 + 回答 + 快照 + 历史帧。"""

    mode: str
    thread_id: str
    db_path: str
    db_size_kb: float
    reply: str
    turns: int
    messages_count: int
    snapshot: dict[str, Any]
    frames: list[dict[str, Any]]
    note: str
    model: str
    elapsed_ms: int


class Day11MappingResult(BaseModel):
    """Day 11 静态概念数据（快照四要素 / 两套暂停机制 / 三档存档器 / 六个坑 / 版本）。"""

    snapshot_fields: list[dict[str, str]]
    pause_mechanisms: list[dict[str, str]]
    savers: list[dict[str, str]]
    pitfalls: list[dict[str, str]]
    time_travel_steps: list[str]
    rules: list[str]
    presets: dict[str, list[str]]
    install_hint: str
    langgraph_version: str
    checkpoint_sqlite_version: str


# ---------- Day 12 多智能体 Supervisor ----------


class Day12SingleRequest(BaseModel):
    """课程 01：一个"什么都会"的 Agent 怎么选错工具。"""

    question: str = Field(
        default="订单 A1001 签收几天了？按政策能不能退？帮我退掉，并给客户写一段说明。",
        min_length=1,
    )
    toolset: Literal["wide", "narrow"] = Field(
        default="wide",
        description="wide 给全部 5 个工具（含退款）/ narrow 只给 3 个只读工具，用来对比越权与假收敛",
    )
    max_steps: int = Field(default=4, ge=1, le=8, description="循环步数上限")
    model_config_id: int | None = None


class Day12SingleResult(BaseModel):
    """单 Agent 的执行现场：决策 / 工具调用 / 观察结果 + 复现出的失败点。"""

    question: str
    toolset: str
    tools_available: list[str] = Field(description="这次交给模型的工具清单")
    steps: list[dict[str, Any]] = Field(description="每步：decision（含 tool_calls）/ tool（观察结果）")
    tool_counts: dict[str, int] = Field(description="各工具被真实调用的次数（refund_order=0 就是没退）")
    final_answer: str
    issues: list[str] = Field(description="三堵墙的可查证据：假收敛 / 越权 / 工具返回值被丢弃")
    llm_calls: int
    note: str
    model: str
    elapsed_ms: int


class Day12SupervisorRequest(BaseModel):
    """课程 02：最小可跑的 Supervisor（1 主管 + 2 专家 + FINISH）。"""

    question: str = Field(
        default="订单 A1001 签收几天了？按政策能不能退？给客户写一段说明。", min_length=1
    )
    max_handoff: int = Field(default=6, ge=1, le=20, description="交接预算：超过就强制收尾")
    dedupe_guard: bool = Field(
        default=True, description="练习 1：连续两次派给同一人且任务高度相似时跳过本次派单"
    )
    recursion_limit: int = Field(default=24, ge=6, le=80, description="超步数保险丝")
    model_config_id: int | None = None


class Day12SupervisorResult(BaseModel):
    """主管模式的完整现场：派单时间线 / 结论账本 / 带 name 的消息流 / 预算与去重。"""

    question: str
    final_answer: str
    findings: list[str] = Field(description="账本层：每个专家回传的一句话结论")
    handoffs: list[dict[str, Any]] = Field(description="派单记录：role / task / reason")
    events: list[dict[str, Any]] = Field(description="审计事件流：dispatch / expert / dedupe / budget / finish")
    messages: list[dict[str, Any]] = Field(description="共享层消息流（专家的 AIMessage 带 name）")
    steps: int = Field(description="已消耗的交接次数")
    dedupe_skips: int = Field(description="去重守卫跳过的派单次数（练习 1）")
    budget_hit: bool = Field(description="是否因交接预算耗尽而强制收尾")
    llm_calls: int = Field(description="真实 LLM 调用次数（主管决策 + 专家子图步数）")
    graph: Day10GraphStructure
    note: str
    model: str
    elapsed_ms: int


class Day12FullRequest(BaseModel):
    """课程 03：加审批 + 加预算 + 加存档（生产形态）。"""

    question: str = Field(
        default="订单 A1001 签收3天，能不能退？退 299，顺便算 299*0.85", min_length=1
    )
    thread_id: str | None = Field(default=None, description="不传则新建；恢复时必须用同一个")
    max_handoff: int = Field(default=6, ge=1, le=20)
    recursion_limit: int = Field(default=30, ge=8, le=80)
    model_config_id: int | None = None


class Day12FullStartResult(BaseModel):
    """第一次调用：图跑到退款审批处冻结（或没有写操作直接跑完）。"""

    thread_id: str
    paused: bool = Field(description="True 表示命中 interrupt，图已冻结并存档")
    next: list[str] = Field(description="接下来要跑的节点：暂停时是 ('coder',)")
    interrupt: dict[str, Any] | None = Field(default=None, description="interrupt 发出的待审批载荷")
    answer: str = Field(default="", description="未暂停时的最终回答")
    findings: list[str]
    handoffs: list[dict[str, Any]]
    events: list[dict[str, Any]]
    steps: int
    budget_hit: bool
    snapshot: dict[str, Any]
    graph: Day10GraphStructure
    note: str
    model: str
    elapsed_ms: int


class Day12FullResumeRequest(BaseModel):
    """人工审批结果：true / false / {'approved': True, 'amount': 199}。"""

    thread_id: str = Field(min_length=1)
    decision: bool | dict[str, Any] = Field(
        default=True,
        description="true 放行 / false 驳回 / 对象表示改参数后放行（amount 覆盖退款金额）",
    )
    max_handoff: int = Field(default=6, ge=1, le=20, description="要与 start 时一致")
    recursion_limit: int = Field(default=30, ge=8, le=80)
    model_config_id: int | None = None


class Day12FullResumeResult(BaseModel):
    """恢复审批后的结果：最终答复 + 真实退款流水 + 审批是否放行。"""

    thread_id: str
    approved: bool
    final_answer: str
    refunds: list[dict[str, Any]] = Field(description="进程内退款流水：thread_id / 订单 / 金额")
    findings: list[str]
    handoffs: list[dict[str, Any]]
    events: list[dict[str, Any]]
    steps: int
    budget_hit: bool
    next: list[str] = Field(description="恢复后应当为空（图跑完了）")
    note: str
    model: str
    elapsed_ms: int


class Day12ParallelRequest(BaseModel):
    """课程 04 / 练习 3：两条调研分支并行，再汇聚给 writer。"""

    question: str = Field(
        default="订单 A1001 签收几天了？按政策能不能退？给客户写一段说明。", min_length=1
    )
    model_config_id: int | None = None


class Day12ParallelResult(BaseModel):
    """并行结果：两条分支各自的耗时与结论 + writer 调用次数。"""

    question: str
    final_answer: str
    branches: list[dict[str, Any]] = Field(description="每条分支：node / detail / elapsed_ms")
    findings: list[str] = Field(description="join 处由 operator.add 合并后的结论")
    writer_calls: int = Field(description="writer 被调用次数（要求恰好 1 次）")
    events: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    llm_calls: int
    note: str
    graph: Day10GraphStructure
    model: str
    elapsed_ms: int


class Day12MappingResult(BaseModel):
    """Day 12 静态概念数据（三堵墙 / 三角色 / 交接方式 / 状态三层 / 三层终止 / 五个坑 / 选型 / 成本）。"""

    walls: list[dict[str, str]]
    roles: list[dict[str, str]]
    handoff_modes: list[dict[str, str]]
    state_layers: list[dict[str, str]]
    terminations: list[dict[str, str]]
    pitfalls: list[dict[str, str]]
    decision_table: list[dict[str, str]]
    cost_table: list[dict[str, str]]
    checklist: list[str]
    presets: dict[str, list[str]]
    install_hint: str
    langgraph_version: str


# ---------- Day 13 流式输出 ----------


class Day13ModesRequest(BaseModel):
    """课程 01：同一张图分别按 values / updates / messages 订阅。"""

    question: str = Field(default="用一句话解释什么是 RAG", min_length=1)
    model_config_id: int | None = None


class Day13ModesResult(BaseModel):
    """四模式对照现场：各模式的事件列表 + 统计 + 多模式二元组序列。"""

    question: str
    values: list[dict[str, Any]] = Field(description="values 事件：index / 消息数 / next / 最后一条摘要")
    updates: list[dict[str, Any]] = Field(description="updates 事件：节点 / 增量键 / 一句话摘要")
    messages: list[dict[str, Any]] = Field(description="messages 事件：节点 / token / 超步 / 相对时间")
    multi: list[dict[str, Any]] = Field(description="多模式订阅的 (mode, payload) 序列，一眼看出分层")
    mode_stats: list[dict[str, Any]] = Field(description="每种模式的事件数与耗时")
    bubbles: list[dict[str, Any]] = Field(description="token 按节点聚合后的气泡（过滤主管 JSON 之后）")
    node_timing: list[dict[str, Any]] = Field(description="每个节点的 token 数与首 token 偏移")
    filtered_tokens: int = Field(description="被过滤掉的主管 JSON token 条数")
    takeaways: list[str]
    graph: Day10GraphStructure
    note: str
    model: str
    elapsed_ms: int


class Day13TokenRequest(BaseModel):
    """课程 02：多 Agent 的 token 按节点分流。"""

    question: str = Field(default="对比一下向量检索和 BM25 的优劣", min_length=1)
    model_config_id: int | None = None


class Day13TokenResult(BaseModel):
    """token 分流结果：每个专家一个气泡 + 被过滤的主管 JSON 明细。"""

    question: str
    bubbles: list[dict[str, Any]] = Field(description="气泡：node / text / tokens / 首末偏移")
    node_timing: list[dict[str, Any]] = Field(description="每节点 token 数 / 字符数 / 首 token 偏移")
    first_token_ms: int = Field(description="首个可见 token 的偏移（感知延迟的核心指标）")
    filtered: dict[str, Any] = Field(description="被过滤的主管 token：节点 / 条数 / 字符 / 样例")
    dispatches: list[dict[str, Any]] = Field(description="主管派单序列（来自 updates 流）")
    filter_rule: str
    note: str
    graph: Day10GraphStructure
    model: str
    elapsed_ms: int


class Day13CustomRequest(BaseModel):
    """课程 03：custom 模式——工具内部上报进度。"""

    question: str = Field(default="LangGraph 的断点续跑是怎么实现的？", min_length=1)
    model_config_id: int | None = None


class Day13CustomResult(BaseModel):
    """工具进度与 token 交错的时间线。"""

    question: str
    progress: list[dict[str, Any]] = Field(description="custom 事件：writer() 写进去的 dict + 偏移")
    stages: list[str] = Field(description="进度文案序列（前端可当进度条节点）")
    tokens: list[dict[str, Any]] = Field(description="messages 事件：节点 / token / 偏移")
    bubbles: list[dict[str, Any]]
    tool_results: list[dict[str, Any]] = Field(description="ToolMessage：工具返回的观察结果")
    final_answer: str
    first_token_ms: int
    protocol_note: str
    note: str
    graph: Day10GraphStructure
    model: str
    elapsed_ms: int


class Day13InterruptRequest(BaseModel):
    """课程 04：interrupt + 流式同开（三次订阅 + 一次续跑，不调模型）。"""

    amount: float = Field(default=199, gt=0, le=100000, description="退款金额（确定性演示，不经模型）")
    decision: bool | dict[str, Any] = Field(
        default=True,
        description="续跑时给 interrupt 的回答：true / false / {'approved': True, 'amount': 150}",
    )
    with_resume: bool = Field(default=True, description="是否演示 Command(resume=) 续跑")


class Day13InterruptResult(BaseModel):
    """三种订阅方式的对照现场：错误的只订 messages / 正确的双订阅 / 续跑。"""

    messages_only: dict[str, Any] = Field(description="只订 messages：事件 0 条但图已暂停（坑）")
    with_updates: dict[str, Any] = Field(description="双订阅：收到 __interrupt__ 与审批载荷")
    resume: dict[str, Any] | None = Field(default=None, description="Command(resume=) 续跑的事件流")
    refunds: list[dict[str, Any]] = Field(description="本次 thread 的退款流水（重放安全证据）")
    rules: list[str]
    note: str
    elapsed_ms: int


class Day13MappingResult(BaseModel):
    """Day 13 静态概念数据（四模式 / API 选型 / interrupt 规则 / SSE 三件套 / 事件协议 / 五个坑）。"""

    stream_modes: list[dict[str, str]]
    api_choices: list[dict[str, str]]
    interrupt_rules: list[dict[str, str]]
    sse_rules: list[dict[str, str]]
    event_protocol: list[dict[str, str]]
    pitfalls: list[dict[str, str]]
    checklist: list[str]
    presets: dict[str, list[str]]
    install_hint: str
    langgraph_version: str


# ---------- Day 14 MCP 协议 ----------


class Day14MappingResult(BaseModel):
    """Day 14 静态概念数据（M×N / FC vs MCP / 三角色 / 两传输 / 三原语 / 迁移对照 / 五个坑）。"""

    mxn: dict[str, Any]
    fc_vs_mcp: list[dict[str, str]]
    roles: list[dict[str, str]]
    transports: list[dict[str, str]]
    primitives: list[dict[str, str]]
    migration: list[dict[str, str]]
    pitfalls: list[dict[str, str]]
    cheatsheet: list[dict[str, str]]
    checklist: list[str]
    presets: dict[str, list[str]]
    servers: list[dict[str, str]]
    install_hint: str
    versions: dict[str, str]


class Day14DiscoverResult(BaseModel):
    """工具发现现场：每个 Server 的协议视角 + LangChain 视角，以及 spawn 了几个子进程。"""

    servers: list[dict[str, Any]] = Field(description="每个 Server：tools / resources / prompts / adapted")
    spawn_count: int = Field(description="本次请求 spawn 的子进程数（stdio 下每开一个会话就是一个进程）")
    total_tools: int
    handshake: list[dict[str, str]] = Field(description="握手五步：spawn → initialize → tools/list → resources → get_tools")
    bridge_note: str
    note: str
    elapsed_ms: int


class Day14CallRequest(BaseModel):
    """直接发 tools/call：不经过模型，用来验证 Server 端校验。"""

    server: Literal["docs", "crm"] = Field(default="crm", description="演示 Server：docs（只读检索）/ crm（读写）")
    tool: str = Field(default="deduct_balance", min_length=1)
    arguments: dict[str, Any] = Field(
        default_factory=lambda: {"customer_id": "C001", "amount": 100},
        description="工具参数（原样进 tools/call 的 arguments）",
    )


class Day14CallResult(BaseModel):
    """一次 tools/call 的结果：isError + 文本内容 + 耗时。"""

    server: str
    tool: str
    arguments: dict[str, Any]
    is_error: bool = Field(description="Server 端把失败变成 isError + 文本，而不是抛异常")
    content: list[str]
    structured: Any = None
    note: str
    elapsed_ms: int


class Day14AgentRequest(BaseModel):
    """课程 03：单 Agent 横跨两个 MCP Server。"""

    question: str = Field(
        default="查一下客户 C001 的信息，然后看看文档库里 Checkpointer 是什么", min_length=1
    )
    model_config_id: int | None = None
    max_steps: int = Field(default=8, ge=2, le=20, description="工具调用步数上限（换算成 recursion_limit）")


class Day14AgentResult(BaseModel):
    """端到端结果：工具清单（带来源 Server）+ 执行轨迹 + 最终回答。"""

    question: str
    tools: list[dict[str, Any]] = Field(description="get_tools() 转换后的工具：name / server / params")
    steps: list[dict[str, Any]] = Field(description="轨迹：tool_call / tool_result / answer")
    tool_call_count: int
    answer: str
    discovery_ms: int = Field(description="spawn + tools/list + 转换耗时（每个请求都要付一次）")
    elapsed_ms: int
    model: str
    note: str


class Day14MixedRequest(BaseModel):
    """课程 05：MCP 工具 + 本地工具混装（余额不足 → 转人工）。"""

    question: str = Field(default="给客户 C001 扣款 5000 元", min_length=1)
    model_config_id: int | None = None


class Day14MixedResult(BaseModel):
    """混装演练现场：气泡（token 分流）+ updates + 完整轨迹 + 是否真的转了人工。"""

    question: str
    tools: list[dict[str, Any]]
    bubbles: list[dict[str, Any]]
    updates: list[dict[str, Any]]
    steps: list[dict[str, Any]]
    escalated: bool = Field(description="是否调用了本地 escalate_to_human")
    final_answer: str
    note: str
    model: str
    elapsed_ms: int


class Day14HttpRequest(BaseModel):
    """课程 04：streamable-http 共享服务（临时 uvicorn，随机端口）。"""

    topic: str = Field(default="本周", min_length=1, description="传给 team_announcement 的参数")


class Day14HttpResult(BaseModel):
    """临时 HTTP Server 的现场：URL、启动耗时、工具清单、调用结果、是否已关闭。"""

    url: str
    startup_ms: int
    stopped: bool
    tools: list[dict[str, Any]]
    call_arguments: dict[str, Any]
    content: list[str]
    is_error: bool
    notes: list[str]
    elapsed_ms: int


class Day14NoisyRequest(BaseModel):
    """坑 1 现场：stdout 污染的三种形态。"""

    mode: Literal["all", "print", "fd_line", "fd"] = Field(
        default="all", description="print=写 sys.stdout / fd_line=整行写 fd1 / fd=半行写 fd1"
    )
    timeout_seconds: float = Field(default=6, ge=2, le=30, description="单次的等待上限（fd 形态会挂死到超时）")
    repeat: int = Field(default=1, ge=1, le=3, description="每种形态跑几次（用来观察整行垃圾的抖动）")


class Day14NoisyResult(BaseModel):
    """三种污染形态的实测结论。"""

    cases: list[dict[str, Any]]
    explain: str
    sdk_note: str
    elapsed_ms: int


# ---------- Day 15 可观测性 ----------


class Day15MappingResult(BaseModel):
    """Day 15 静态概念数据（失败形态 / 三大件 / 两条路线 / 八字段 / 指标定义 / 五个坑 / 速查表）。"""

    failure_modes: list[dict[str, str]]
    pillars: list[dict[str, str]]
    routes: list[dict[str, str]]
    trace_fields: list[dict[str, str]]
    metric_defs: list[dict[str, str]]
    pitfalls: list[dict[str, str]]
    cheatsheet: list[dict[str, str]]
    checklist: list[str]
    presets: dict[str, list[str]]
    platform: dict[str, Any] = Field(description="LangSmith / Langfuse 的就绪状态（缺哪些环境变量、装没装）")
    files: dict[str, str] = Field(description="自建 trace / 业务日志 / 服务端 span / 反馈的落盘路径")
    versions: dict[str, str]
    install_hint: str


class Day15CallbackRequest(BaseModel):
    """课程 02：自建 CallbackHandler 跑一次真实 Agent。"""

    question: str = Field(default="帮我报销昨天的午餐 45 元", min_length=1)
    user_id: str = Field(default="u_889", min_length=1, description="会随 metadata 上报，作为筛选维度")
    model_config_id: int | None = None


class Day15CallbackResult(BaseModel):
    """回调现场：树 + 时间轴 + 原始事件 + token 统计。"""

    question: str
    answer: str
    tree: list[dict[str, Any]] = Field(description="按 parent_run_id 还原的调用树")
    timeline: list[dict[str, Any]] = Field(description="按 offset_ms 排序的甘特条数据")
    events: list[dict[str, Any]] = Field(description="原始事件：LLM 输入预览 / token / 工具参数与返回")
    llm_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    tags: list[str]
    metadata: dict[str, Any]
    model: str
    note: str
    elapsed_ms: int


class Day15PlatformRequest(BaseModel):
    """课程 01/03：平台接入（没配 key 时只返回配置说明，不跑模型）。"""

    provider: Literal["langsmith", "langfuse"] = Field(default="langsmith")
    question: str = Field(default="帮我报销昨天的午餐 45 元", min_length=1)
    user_id: str = Field(default="u_889", min_length=1)
    model_config_id: int | None = None
    feedback: bool = Field(default=False, description="Langfuse：跑完回填一条点赞 score（Day 16 评估集原料）")


class Day15PlatformResult(BaseModel):
    """平台接入现场：配置状态 + 标签 + trace id + 下一步指引。"""

    provider: str
    configured: bool = Field(description="false 时不会跑模型，只返回配置说明")
    missing: list[str]
    answer: str
    trace_id: str = Field(description="Langfuse 可直接拿到；LangSmith 由平台分配，UI 里按 tags 筛选")
    tags: list[str]
    metadata: dict[str, Any]
    dashboard: str
    instructions: list[str]
    notes: list[str]
    model: str
    elapsed_ms: int
    run_info: dict[str, Any] | None = Field(default=None, description="尽力读取的最近一条 LangSmith run")


class Day15OwnTraceRequest(BaseModel):
    """课程 04：自建 span + JSONL（真实调模型，但结构完全可复现）。"""

    question: str = Field(default="报销高铁票 237 元", min_length=1)
    user_id: str = Field(default="u_889", min_length=1)
    model_config_id: int | None = None
    max_steps: int = Field(default=5, ge=2, le=10, description="手写循环的 MAX_STEPS 保险丝")
    concurrency_runs: int = Field(default=3, ge=1, le=5, description="并发不串线实验的协程数")


class Day15OwnTraceResult(BaseModel):
    """自建追踪现场：本次 trace 的树与指标 + 全局指标 + 并发隔离证据。"""

    trace_id: str
    tree: list[dict[str, Any]]
    spans: list[dict[str, Any]] = Field(description="扁平八字段记录（与 JSONL 行一致，已脱敏）")
    metrics: dict[str, Any] = Field(description="本次 trace 聚合：p50/p95 / token / 成本 / 错误率")
    global_metrics: dict[str, Any] = Field(description="traces.jsonl 全量聚合")
    concurrency: dict[str, Any] = Field(description="ContextVar 并发隔离实验（纯本地，不调模型）")
    answer: str
    stop_reason: str
    model: str
    file: str
    note: str
    elapsed_ms: int


class Day15FullStackRequest(BaseModel):
    """课程 05：LangGraph + MCP 全链路（trace_id 跨进程传播）。"""

    question: str = Field(default="报销昨天的晚餐 78 元", min_length=1)
    model_config_id: int | None = None


class Day15FullStackResult(BaseModel):
    """全链路现场：客户端事件 + 服务端 span + 合并时间轴。"""

    trace_id: str
    answer: str
    terminated_normally: bool
    termination: str
    steps: int
    tool_calls: list[dict[str, Any]]
    business: dict[str, Any] | None
    server_spans: list[dict[str, Any]]
    merged: list[dict[str, Any]]
    matched: bool = Field(description="服务端是否收到同一个 trace_id（模型有没有把参数传过去）")
    input_tokens: int
    output_tokens: int
    files: dict[str, str]
    notes: list[str]
    model: str
    elapsed_ms: int


class Day15FeedbackRequest(BaseModel):
    """自建版 score：把用户反馈落到某条 trace 上。"""

    trace_id: str = Field(min_length=1, description="要打分的 trace（自建 traces.jsonl 里的 id）")
    value: int = Field(default=1, ge=-1, le=1, description="1=赞 / -1=踩 / 0=中性")
    comment: str = Field(default="", max_length=500)


class Day15FeedbackResult(BaseModel):
    """反馈落库结果 + 点赞率聚合。"""

    trace_id: str
    value: int
    comment: str
    summary: dict[str, Any]


class Day15TracesResult(BaseModel):
    """trace 回放与指标看板：最近 N 条 trace + 全局指标 + 反馈聚合。"""

    traces: list[dict[str, Any]] = Field(description="每条 trace：question / termination / tree / metrics")
    metrics: dict[str, Any]
    feedback: dict[str, Any]
    files: dict[str, str]


# ---------- Day 16 评估体系 ----------


class Day16MappingResult(BaseModel):
    """Day 16 静态概念数据（三个困难 / 三层评估 / 指标体系 / 数据来源 / rubric / 五个坑 / 速查表）。"""

    difficulties: list[dict[str, str]]
    levels: list[dict[str, str]]
    metric_groups: list[dict[str, str]]
    dataset_sources: list[dict[str, str]]
    case_fields: list[dict[str, str]]
    rubric_rules: list[dict[str, str]]
    pitfalls: list[dict[str, str]]
    cheatsheet: list[dict[str, str]]
    checklist: list[str]
    presets: dict[str, list[str]]
    behavior_categories: list[str] = Field(description="有行为断言的类别（越界 / 能力边界 / 歧义 / 安全）")
    gates: dict[str, Any] = Field(description="默认门禁阈值与 judge 校准口径")
    prompts: dict[str, str] = Field(description="被对比的两版系统提示词")
    files: dict[str, str]
    dataset: dict[str, Any] = Field(description="当前数据集画像（类别 / 难度 / 来源分布）")
    install_hint: str


class Day16BuildRequest(BaseModel):
    """构建评估集：三个来源可分别开关。"""

    include_feedback: bool = Field(default=True, description="把 Day 15 点踩 trace 收进回归集（真实失败）")
    include_seed: bool = Field(default=True, description="内置人工边界 case（越界 / 能力边界 / 歧义 / 安全）")
    include_synthesized: bool = Field(default=True, description="内置合成变体（等价于 LLM 仿写扩量，需人工抽查）")
    replace: bool = Field(default=True, description="True=覆盖重写；False=与已有数据集按 input 合并去重")


class Day16DatasetResult(BaseModel):
    """数据集现场：总数 + 三个维度的分布 + 全部 case。"""

    total: int
    distribution: dict[str, int] = Field(description="按 category 的分布")
    difficulties: dict[str, int]
    sources: dict[str, int] = Field(description="seed / synthesized / online 三类来源")
    online_found: int = Field(default=0, description="本次从 Day 15 反馈里捞到的点踩条数")
    cases: list[dict[str, Any]]
    files: dict[str, str]
    notes: list[str]


class Day16RunRequest(BaseModel):
    """跑评估：每 case 采样 N 次取多数，可按类别过滤与截断。"""

    limit: int = Field(default=5, ge=1, le=50, description="本次最多跑几条 case（控成本）")
    runs: int = Field(default=3, ge=1, le=5, description="每条 case 跑几次（多数投票，课程建议 3）")
    category: str | None = Field(default=None, description="只跑某一类（如「越界」）")
    model_config_id: int | None = None


class Day16RunResult(BaseModel):
    """评估报告：总体正确率 + 按类别切分 + 逐 case 明细 + 失败清单。"""

    total: int
    accuracy: float = Field(description="多数投票后通过的 case 占比")
    avg_pass_rate: float = Field(description="逐 case 通过率的均值（比 accuracy 更平滑）")
    avg_tokens: int
    avg_steps: float
    by_category: dict[str, dict[str, Any]]
    runs_per_case: int
    model: str
    cases: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    files: dict[str, str]
    notes: list[str]
    elapsed_ms: int


class Day16JudgeRequest(BaseModel):
    """LLM-as-judge：单点 rubric 打分 + 可选 pairwise 换序对比。"""

    question: str = Field(default="报销昨天的晚餐 78 元", min_length=1)
    answer: str = Field(
        default="已为您提交报销：餐饮 78 元，单号 EXP-2046。",
        min_length=1,
        description="被评的回答（单点 rubric 用）",
    )
    old_answer: str = Field(default="已提交。", description="pairwise 的旧版回答；留空则跳过对比")
    new_answer: str = Field(
        default="已提交餐饮报销 78 元（单号 EXP-2046）。注意：餐饮报销需附发票，每日上限 100 元。",
        description="pairwise 的新版回答",
    )
    model_config_id: int | None = None
    judge_model_config_id: int | None = Field(
        default=None, description="judge 单独指定更强的模型；不填即与被评模型同源（会有自评偏差）"
    )


class Day16JudgeResult(BaseModel):
    """judge 现场：单点结论 + 降级轨迹 + pairwise 换序结论 + 自评偏差提示。"""

    verdict: dict[str, Any] = Field(description="score / passed / reason")
    method: str = Field(description="default / function_calling / json_mode / heuristic")
    attempts: list[str]
    pairwise: dict[str, Any] | None
    pair_attempts: list[str]
    self_judging: bool
    model: str
    notes: list[str]
    elapsed_ms: int


class Day16RegressionRequest(BaseModel):
    """回归对比：baseline vs candidate 双跑同一数据集，带门禁。"""

    limit: int = Field(default=4, ge=1, le=20, description="双跑很贵，默认只取前 4 条")
    runs: int = Field(default=2, ge=1, le=5, description="每 case 采样次数（双跑 × runs 次调用）")
    category: str | None = None
    accuracy_drop_max: float = Field(default=0.02, ge=0, le=1, description="正确率最大降幅（0.02 = 2 个百分点）")
    cost_increase_max: float = Field(default=0.15, ge=0, le=2, description="平均 token 最大涨幅")
    candidate_prompt: str | None = Field(default=None, description="留空用内置 candidate 提示词")
    model_config_id: int | None = None


class Day16RegressionResult(BaseModel):
    """回归现场：两版摘要 + 逐 case diff + 门禁判定。"""

    baseline: dict[str, Any]
    candidate: dict[str, Any]
    diff: dict[str, Any]
    gate: dict[str, Any]
    model: str
    runs_per_case: int
    files: dict[str, str]
    notes: list[str]
    elapsed_ms: int


class Day16CiRequest(BaseModel):
    """CI 冒烟门禁：跑冒烟子集，与归档基线比较。"""

    smoke_size: int = Field(default=10, ge=1, le=50, description="冒烟子集大小（课程建议 20）")
    runs: int = Field(default=1, ge=1, le=3, description="冒烟为了快，默认每条只跑一次")
    tolerance: float = Field(default=0.02, ge=0, le=1, description="允许低于基线的幅度")
    refresh_baseline: bool = Field(default=False, description="跑完把本次成绩归档为新基线（夜跑用）")
    candidate_prompt: str | None = None
    model_config_id: int | None = None


class Day16CiResult(BaseModel):
    """冒烟门禁现场：passed / exit_code / 基线对比 / 失败清单。"""

    passed: bool
    exit_code: int = Field(description="0=允许合并；1=CI 标红")
    accuracy: float
    baseline_accuracy: float
    tolerance: float
    smoke_cases: int
    total_cases: int
    archived: bool = Field(description="本次是否把成绩归档为新基线（首次运行或勾选刷新）")
    baseline: dict[str, Any] | None
    results: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    by_category: dict[str, dict[str, Any]]
    model: str
    files: dict[str, str]
    notes: list[str]
    elapsed_ms: int
