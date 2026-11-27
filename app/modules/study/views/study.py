"""智能体开发学习接口：工具清单、工具调用日志、Agent 循环、ReAct、结构化输出、多轮会话。

除 Day 07 的长期记忆与知识库（以及 Day 13 的 SSE 流）外，其余接口都不落库——
学习页是"试验台"，会话状态只活在进程内存里，重启即清。

Day 15 的可观测性接口是例外中的例外：它把 trace 写进 ``agent_backend/var/day15/*.jsonl``
（文件而不是数据库），这样重启后仍能回放历史轨迹——这正是"可观测性数据要留存"的本意。
"""

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.deps import get_db
from app.modules.study.crud import knowledge as knowledge_crud
from app.modules.study.schemas.study import (
    AgentRunRequest,
    AgentRunResult,
    ContextBuildRequest,
    ContextBuildResult,
    Day07ChatRequest,
    Day07ChatResult,
    Day09AgentRequest,
    Day09AgentResult,
    Day09LcelRequest,
    Day09LcelResult,
    Day09MappingResult,
    Day09MessagesResult,
    Day09PromptRequest,
    Day09PromptResult,
    Day09StructuredRequest,
    Day09StructuredResult,
    Day10ConditionalRequest,
    Day10ConditionalResult,
    Day10MappingResult,
    Day10MinGraphRequest,
    Day10MinGraphResult,
    Day10ParallelRequest,
    Day10ParallelResult,
    Day10ReactRequest,
    Day10ReactResult,
    Day10StateResult,
    Day11ApprovalResumeRequest,
    Day11ApprovalResumeResult,
    Day11ApprovalStartRequest,
    Day11ApprovalStartResult,
    Day11BasicRequest,
    Day11BasicResult,
    Day11ChatRequest,
    Day11ChatResult,
    Day11ExpireRequest,
    Day11ExpireResult,
    Day11MappingResult,
    Day11PendingOut,
    Day11ReplayResult,
    Day11SqliteRequest,
    Day11SqliteResult,
    Day11TimeTravelRequest,
    Day11TimeTravelResult,
    Day12FullRequest,
    Day12FullResumeRequest,
    Day12FullResumeResult,
    Day12FullStartResult,
    Day12MappingResult,
    Day12ParallelRequest,
    Day12ParallelResult,
    Day12SingleRequest,
    Day12SingleResult,
    Day12SupervisorRequest,
    Day12SupervisorResult,
    Day13CustomRequest,
    Day13CustomResult,
    Day13InterruptRequest,
    Day13InterruptResult,
    Day13MappingResult,
    Day13ModesRequest,
    Day13ModesResult,
    Day13TokenRequest,
    Day13TokenResult,
    Day14AgentRequest,
    Day14AgentResult,
    Day14CallRequest,
    Day14CallResult,
    Day14DiscoverResult,
    Day14HttpRequest,
    Day14HttpResult,
    Day14MappingResult,
    Day14MixedRequest,
    Day14MixedResult,
    Day14NoisyRequest,
    Day14NoisyResult,
    Day15CallbackRequest,
    Day15CallbackResult,
    Day15FeedbackRequest,
    Day15FeedbackResult,
    Day15FullStackRequest,
    Day15FullStackResult,
    Day15MappingResult,
    Day15OwnTraceRequest,
    Day15OwnTraceResult,
    Day15PlatformRequest,
    Day15PlatformResult,
    Day15TracesResult,
    Day16BuildRequest,
    Day16CiRequest,
    Day16CiResult,
    Day16DatasetResult,
    Day16JudgeRequest,
    Day16JudgeResult,
    Day16MappingResult,
    Day16RegressionRequest,
    Day16RegressionResult,
    Day16RunRequest,
    Day16RunResult,
    Day17AbRequest,
    Day17AbResult,
    Day17BudgetRequest,
    Day17BudgetResult,
    Day17CacheRequest,
    Day17CacheResult,
    Day17CompressRequest,
    Day17CompressResult,
    Day17LedgerRequest,
    Day17LedgerResult,
    Day17MappingResult,
    Day17ParallelRequest,
    Day17ParallelResult,
    Day17RouterRequest,
    Day17RouterResult,
    Day18ApprovalRequest,
    Day18ApprovalResult,
    Day18AuthzRequest,
    Day18AuthzResult,
    Day18ExfilRequest,
    Day18ExfilResult,
    Day18FenceRequest,
    Day18FenceResult,
    Day18InjectionRequest,
    Day18InjectionResult,
    Day18MappingResult,
    Day18PipelineRequest,
    Day18PipelineResult,
    Day18SandboxRequest,
    Day18SandboxResult,
    Day18SuiteResult,
    Day19ConcurrencyRequest,
    Day19ConcurrencyResult,
    Day19MappingResult,
    Day19RateLimitRequest,
    Day19RateLimitResult,
    Day19RolloutRequest,
    Day19RolloutResult,
    Day19RunsRequest,
    Day19RunsResult,
    Day19SessionRequest,
    Day19SessionResult,
    Day19SkeletonResult,
    Day19SsePreviewRequest,
    Day19SsePreviewResult,
    Day20CapacityRequest,
    Day20CapacityResult,
    Day20ChaosRequest,
    Day20ChaosResult,
    Day20LadderRequest,
    Day20LadderResult,
    Day20MappingResult,
    Day20RetryRequest,
    Day20RetryResult,
    Day20ShutdownRequest,
    Day20ShutdownResult,
    Day20SloRequest,
    Day20SloResult,
    Day21EvalRequest,
    Day21EvalResult,
    Day21ExecuteRequest,
    Day21ExecuteResult,
    Day21FailuresRequest,
    Day21FailuresResult,
    Day21GraphRequest,
    Day21GraphResult,
    Day21MappingResult,
    Day21ReflexionRequest,
    Day21ReflexionResult,
    Day21ReplanRequest,
    Day21ReplanResult,
    Day21SchemaRequest,
    Day21SchemaResult,
    Day22AstRequest,
    Day22AstResult,
    Day22BridgeRequest,
    Day22BridgeResult,
    Day22ContainerRequest,
    Day22ContainerResult,
    Day22L1Request,
    Day22L1Result,
    Day22LoopRequest,
    Day22LoopResult,
    Day22MappingResult,
    Day22NetfsRequest,
    Day22NetfsResult,
    Day22PressureRequest,
    Day22PressureResult,
    Day22ProductionRequest,
    Day22ProductionResult,
    Day22WhyRequest,
    Day22WhyResult,
    ExtractRunRequest,
    ExtractRunResult,
    KnowledgeDocIn,
    KnowledgeDocOut,
    KnowledgeHit,
    KnowledgeIndexResult,
    KnowledgeSearchRequest,
    MemoryProfileOut,
    MemoryWriteRequest,
    MemoryWriteResult,
    ReactRunRequest,
    ReactRunResult,
    SessionChatRequest,
    SessionChatResult,
    SessionStateOut,
    ToolCallLogEntry,
    ToolOut,
    ToolSpecBudget,
)
from app.modules.study.services import (
    agent_loop,
    day07_agent,
    day08_context,
    day09_langchain,
    day10_langgraph,
    day11_checkpointer,
    day12_supervisor,
    day13_stream,
    day14_mcp,
    day15_observability,
    day16_eval,
    day17_cost,
    day18_security,
    day19_service,
    day20_reliability,
    day21_planning,
    day22_sandbox,
    long_memory,
    rag,
    react_loop,
    session_agent,
    structured_extract,
    tools,
)
from app.modules.study.services.tools import TOOL_SCHEMAS

router = APIRouter(tags=["智能体开发学习"])


@router.get(
    "/tools",
    response_model=APIResponse[list[ToolOut]],
    summary="工具清单（Day 01 定义 + Day 04 元数据）",
)
async def list_tools() -> APIResponse[list[ToolOut]]:
    """返回当前注册的全部工具与 JSON Schema 说明书（展示"模型眼里看到的工具长什么样"）。"""
    data = [
        ToolOut(
            name=schema["function"]["name"],
            description=schema["function"]["description"],
            parameters=schema["function"]["parameters"],
            side_effect=tools.REGISTRY[schema["function"]["name"]]["side_effect"],
            require_confirm=tools.REGISTRY[schema["function"]["name"]]["require_confirm"],
        )
        for schema in TOOL_SCHEMAS
    ]
    return APIResponse(data=data)


@router.get(
    "/tools/calls",
    response_model=APIResponse[list[ToolCallLogEntry]],
    summary="Day 03 工具调用日志",
)
async def list_tool_calls() -> APIResponse[list[ToolCallLogEntry]]:
    """返回最近的工具调用记录（内存环形队列，最多 200 条，重启即清）。

    用来观察"一轮 Agent 对话到底真实执行了几次工具、耗时多少、
    哪些报错后模型自我纠正了"——即 dispatch 分发器（练习 2）的落点。
    """
    return APIResponse(data=[ToolCallLogEntry(**entry) for entry in tools.TOOL_CALL_LOG])


@router.get(
    "/tools/budget",
    response_model=APIResponse[ToolSpecBudget],
    summary="Day 04 练习 2：工具说明书预算",
)
async def get_tool_budget() -> APIResponse[ToolSpecBudget]:
    """统计 tools 清单序列化后的总字符数与最长的 3 个 description。

    说明书随每轮请求重复发送，是最容易被忽视的 token 成本项；
    超过预算（2000 字符）时前端给出警告，提示精简最长的 description。
    """
    return APIResponse(data=tools.spec_budget())


@router.post(
    "/agent/run",
    response_model=APIResponse[AgentRunResult],
    summary="运行 Function Calling Agent 循环",
)
async def run_agent(
    payload: AgentRunRequest,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[AgentRunResult]:
    """跑一次完整的"决策 → 调工具 → 观察"循环，返回每一步的执行轨迹。"""
    try:
        result = await agent_loop.run(db, payload)
    except ValueError as exc:  # 业务校验失败（如无可用模型）
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001  上游模型服务失败
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/react/run",
    response_model=APIResponse[ReactRunResult],
    summary="运行手写 ReAct 文本协议循环",
)
async def run_react(
    payload: ReactRunRequest,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ReactRunResult]:
    """不借助 Function Calling，用纯文本协议跑 ReAct 循环，返回每一步原文与观察结果。"""
    try:
        result = await react_loop.run(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/extract/run",
    response_model=APIResponse[ExtractRunResult],
    summary="Day 02 结构化输出实验：客户意图提取（三级方案）",
)
async def run_extract(
    payload: ExtractRunRequest,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ExtractRunResult]:
    """按选定方案抽取工单：口头要求（反面教材）/ JSON Mode / Pydantic Schema。

    返回每次尝试的轨迹（原文、报错、降级说明）——直观对比三级方案
    "谁保证合法 JSON、谁保证字段类型"，并展示失败现场回填重试的效果。
    """
    try:
        result = await structured_extract.run(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/session/chat",
    response_model=APIResponse[SessionChatResult],
    summary="Day 06 多轮会话：短期记忆 + Agent 循环",
)
async def chat_session(
    payload: SessionChatRequest,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[SessionChatResult]:
    """同一 session_id 的多次请求共享 SessionState（约束锚点/事实/摘要/近轮）。

    循环本身还是 Day 05 那套，区别是 messages 由记忆层投影而来，不再从零开始。
    """
    session_id, agent = session_agent.session_manager.get_or_create(
        payload.session_id, agent_loop.SYSTEM_PROMPT
    )
    request = payload.model_copy(
        update={"session_id": session_id}
    )  # 把自动生成的 id 回填给 service（供返回值使用）
    try:
        result = await agent.chat(db, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.get(
    "/session/{session_id}",
    response_model=APIResponse[SessionStateOut],
    summary="Day 06 会话状态快照",
)
async def get_session(session_id: str) -> APIResponse[SessionStateOut]:
    """查看会话真相源：约束、事实槽位、摘要、水位与淘汰记录。"""
    agent = session_agent.session_manager.get(session_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="会话不存在或已随进程重启清空")
    return APIResponse(data=agent.snapshot())


@router.delete(
    "/session/{session_id}",
    response_model=APIResponse[dict[str, bool]],
    summary="Day 06 重置会话（清空记忆）",
)
async def delete_session(session_id: str) -> APIResponse[dict[str, bool]]:
    """新建一份同 id 的空会话（记忆全清，id 可继续复用）。"""
    if not session_agent.session_manager.reset(session_id):
        raise HTTPException(status_code=404, detail="会话不存在或已随进程重启清空")
    return APIResponse(data={"reset": True})


# ---------- Day 07：长期记忆（画像类管道） ----------


@router.get(
    "/memory/{user_id}",
    response_model=APIResponse[MemoryProfileOut],
    summary="Day 07 用户长期画像",
)
async def get_memory(
    user_id: str, db: AsyncSession = Depends(get_db)
) -> APIResponse[MemoryProfileOut]:
    """返回该用户的全部事实、会话开头要注入的文本，以及同义键冲突提示。"""
    facts = await long_memory.list_facts(db, user_id)
    data = MemoryProfileOut(
        user_id=user_id,
        facts=facts,
        prompt=long_memory.recall_prompt(facts),
        conflicts=long_memory.detect_conflicts(facts),
    )
    return APIResponse(data=data)


@router.post(
    "/memory/{user_id}",
    response_model=APIResponse[MemoryWriteResult],
    summary="Day 07 写入 / 覆盖长期记忆",
)
async def write_memory(
    user_id: str,
    payload: MemoryWriteRequest,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[MemoryWriteResult]:
    """同一 key 覆盖而非追加；created 标明本次是新建还是覆盖。"""
    try:
        result = await long_memory.remember(
            db, user_id, payload.key, payload.value, payload.source_turn
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.delete(
    "/memory/{user_id}/{key}",
    response_model=APIResponse[dict[str, bool]],
    summary="Day 07 遗忘一条长期记忆",
)
async def forget_memory(
    user_id: str, key: str, db: AsyncSession = Depends(get_db)
) -> APIResponse[dict[str, bool]]:
    """按 key 删除一条事实（课程练习 1 的 forget）。"""
    deleted = await long_memory.forget(db, user_id, key)
    return APIResponse(data={"deleted": deleted})


@router.delete(
    "/memory/{user_id}",
    response_model=APIResponse[dict[str, int]],
    summary="Day 07 清空某用户的长期记忆",
)
async def clear_memory(
    user_id: str, db: AsyncSession = Depends(get_db)
) -> APIResponse[dict[str, int]]:
    """清空该用户全部事实，返回删除条数。"""
    deleted = await long_memory.clear_user(db, user_id)
    return APIResponse(data={"deleted": deleted})


# ---------- Day 07：RAG 知识库（文档类管道） ----------


@router.post(
    "/knowledge/docs",
    response_model=APIResponse[KnowledgeIndexResult],
    summary="Day 07 录入 / 重建知识文档（切块 + 向量化）",
)
async def upsert_document(
    payload: KnowledgeDocIn, db: AsyncSession = Depends(get_db)
) -> APIResponse[KnowledgeIndexResult]:
    """先删同 doc_id 的旧块再重建，保证旧版本不残留。"""
    try:
        result = await rag.index_document(
            db,
            doc_id=payload.doc_id,
            title=payload.title,
            content=payload.content,
            version=payload.version,
            size=payload.chunk_size,
            overlap=payload.overlap,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.get(
    "/knowledge/docs",
    response_model=APIResponse[list[KnowledgeDocOut]],
    summary="Day 07 知识文档列表",
)
async def list_documents(db: AsyncSession = Depends(get_db)) -> APIResponse[list[KnowledgeDocOut]]:
    """按 doc_id 聚合出文档列表（标题 / 版本 / 块数 / 更新时间）。"""
    rows = await knowledge_crud.list_doc_summaries(db)
    return APIResponse(data=[KnowledgeDocOut(**row) for row in rows])


@router.delete(
    "/knowledge/docs/{doc_id}",
    response_model=APIResponse[dict[str, int]],
    summary="Day 07 删除整篇文档的全部块",
)
async def delete_document(
    doc_id: str, db: AsyncSession = Depends(get_db)
) -> APIResponse[dict[str, int]]:
    """按 doc_id 删除全部切块，返回删除条数。"""
    deleted = await knowledge_crud.delete_doc(db, doc_id)
    return APIResponse(data={"deleted": deleted})


@router.post(
    "/knowledge/search",
    response_model=APIResponse[list[KnowledgeHit]],
    summary="Day 07 混合检索（向量 + BM25 + RRF）",
)
async def search_knowledge(
    payload: KnowledgeSearchRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[list[KnowledgeHit]]:
    """只检索不生成，方便前端单独观察两路排名如何被 RRF 融合。"""
    where = {"doc_id": payload.doc_id} if payload.doc_id else None
    hits = await rag.search(
        db, payload.query, top_k=payload.top_k, alpha=payload.alpha, where=where
    )
    return APIResponse(data=hits)


# ---------- Day 07：三条管道组装 ----------


@router.post(
    "/day07/chat",
    response_model=APIResponse[Day07ChatResult],
    summary="Day 07 问答：长期画像 + RAG + 工具",
)
async def chat_day07(
    payload: Day07ChatRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day07ChatResult]:
    """一次问答走完三条管道，返回最终回答与每条管道的执行摘要。"""
    try:
        result = await day07_agent.chat(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


# ---------- Day 08：上下文工程（四层编排器） ----------


@router.post(
    "/day08/context",
    response_model=APIResponse[ContextBuildResult],
    summary="Day 08 上下文编排：四层结构 + 预算账单 + 七问自检 + dump",
)
async def build_context(
    payload: ContextBuildRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[ContextBuildResult]:
    """只编排、不调模型：返回最终上下文、每层的预算账单、检索块的裁剪与摆放轨迹、七问自检结果。

    这样迭代上下文逻辑不必等模型回答（也不花 token）——"模型变笨"时先看 dump，
    确认约束还在、检索块没丢进中部、历史没被拦腰斩断，再去怀疑模型本身。
    """
    try:
        result = await day08_context.build(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


# ---------- Day 09 LangChain 初探（框架篇开篇） ----------


@router.get(
    "/day09/mapping",
    response_model=APIResponse[Day09MappingResult],
    summary="Day 09 零件对照表（手写 ↔ 框架）+ 依赖版本",
)
async def get_day09_mapping() -> APIResponse[Day09MappingResult]:
    """返回静态概念数据：手写零件与 LangChain 零件的对应关系、消息模式对照、预设问题。"""
    return APIResponse(data=await day09_langchain.mapping())


@router.post(
    "/day09/messages",
    response_model=APIResponse[Day09MessagesResult],
    summary="Day 09 消息抽象：裸 dict → 类型对象（不调模型）",
)
async def run_day09_messages() -> APIResponse[Day09MessagesResult]:
    """构造四类消息对象并转回 OpenAI 格式，演示"role 拼错"在两种世界里的不同结局。"""
    try:
        result = day09_langchain.messages_demo()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.post(
    "/day09/prompt",
    response_model=APIResponse[Day09PromptResult],
    summary="Day 09 ChatPromptTemplate：变量 / partial / fail-fast（不调模型）",
)
async def run_day09_prompt(payload: Day09PromptRequest) -> APIResponse[Day09PromptResult]:
    """填充模板变量；勾选"故意漏填"可看到模板在 invoke 阶段就 KeyError。"""
    try:
        result = day09_langchain.prompt_demo(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.post(
    "/day09/structured",
    response_model=APIResponse[Day09StructuredResult],
    summary="Day 09 with_structured_output：三字段回执抽取",
)
async def run_day09_structured(
    payload: Day09StructuredRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day09StructuredResult]:
    """把 Pydantic 模型交给 with_structured_output，返回解析结果、下发 Schema 与抖动字段。"""
    try:
        result = await day09_langchain.run_structured(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day09/lcel",
    response_model=APIResponse[Day09LcelResult],
    summary="Day 09 LCEL 管道：invoke / stream / batch / retry / lambda",
)
async def run_day09_lcel(
    payload: Day09LcelRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day09LcelResult]:
    """跑一次 LCEL 管道，返回组件链、回答、分片（stream）、批量耗时对比（batch）与重试轨迹。"""
    try:
        result = await day09_langchain.run_lcel(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day09/agent",
    response_model=APIResponse[Day09AgentResult],
    summary="Day 09 框架版 Agent 循环：@tool + bind_tools + MAX_STEPS",
)
async def run_day09_agent(
    payload: Day09AgentRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day09AgentResult]:
    """用 LangChain 消息对象重写 Day 01 的循环；可切换工具 description 变体观察选错工具。"""
    try:
        result = await day09_langchain.run_agent(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


# ---------- Day 10 LangGraph 状态图 ----------


@router.get(
    "/day10/mapping",
    response_model=APIResponse[Day10MappingResult],
    summary="Day 10 概念数据：三件套 / reducer / while 对照 / 六个坑 + 版本",
)
async def get_day10_mapping() -> APIResponse[Day10MappingResult]:
    """返回静态概念数据：State/Node/Edge 三件套、reducer 语义、手写 while 对照表、六个坑与预设问题。"""
    return APIResponse(data=await day10_langgraph.mapping())


@router.post(
    "/day10/state",
    response_model=APIResponse[Day10StateResult],
    summary="Day 10 状态与 reducer：覆盖 vs 追加（不调模型）",
)
async def run_day10_state() -> APIResponse[Day10StateResult]:
    """跑四个对照实验：无 reducer / operator.add / add_messages / 节点返回全量的坑。

    这是今天最容易翻车的地方——"读语义对、写语义错"，print 一切正常但最终结果少一半。
    """
    try:
        result = await day10_langgraph.state_reducer_demo()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.post(
    "/day10/min-graph",
    response_model=APIResponse[Day10MinGraphResult],
    summary="Day 10 最小线性图：检索 → 生成 + 逐节点观察",
)
async def run_day10_min_graph(
    payload: Day10MinGraphRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day10MinGraphResult]:
    """跑 START → retrieve → generate → END，同时返回 updates（补丁）与 values（完整状态）事件。"""
    try:
        result = await day10_langgraph.run_min_graph(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day10/conditional",
    response_model=APIResponse[Day10ConditionalResult],
    summary="Day 10 条件边：classify + route + 四条分支（含低置信度转人工）",
)
async def run_day10_conditional(
    payload: Day10ConditionalRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day10ConditionalResult]:
    """决策与执行分离：classify 只打标签、route 只选路、四条分支各自收口。"""
    try:
        result = await day10_langgraph.run_conditional(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day10/react",
    response_model=APIResponse[Day10ReactResult],
    summary="Day 10 完整 ReAct 环：agent ⇄ tools + 双刹车 + 降级交付",
)
async def run_day10_react(
    payload: Day10ReactRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day10ReactResult]:
    """把 Day 01 的 while 循环变成一条回头边；三条出口：语义收敛 / 语义刹车 / 物理保险丝。

    ``tool_mode=broken`` 验证 6 步后优雅收尾（练习 2）；``as_subgraph`` 把环打包成子图（练习 3）。
    """
    try:
        result = await day10_langgraph.run_react(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day10/parallel",
    response_model=APIResponse[Day10ParallelResult],
    summary="Day 10 并行扇出/汇聚：静态 + Send 动态 + 子图（本地假检索）",
)
async def run_day10_parallel(payload: Day10ParallelRequest) -> APIResponse[Day10ParallelResult]:
    """三件事一起看：扇出耗时取最大而非求和、Send 的分支数运行时决定、子图可当节点用。"""
    try:
        result = await day10_langgraph.run_parallel(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


# ---------- Day 11 Checkpointer 与人机协同 ----------


@router.get(
    "/day11/mapping",
    response_model=APIResponse[Day11MappingResult],
    summary="Day 11 概念数据：快照四要素 / 两套暂停机制 / 三档存档器 / 六个坑 + 版本",
)
async def get_day11_mapping() -> APIResponse[Day11MappingResult]:
    """返回静态概念数据：StateSnapshot 四要素、动态中断与静态断点对照、存档器选型、六个坑。"""
    return APIResponse(data=await day11_checkpointer.mapping())


@router.post(
    "/day11/basic",
    response_model=APIResponse[Day11BasicResult],
    summary="Day 11 存档三实验：同 thread 续跑 / 跨 thread 隔离 / get_state 快照",
)
async def run_day11_basic(
    payload: Day11BasicRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day11BasicResult]:
    """课程 01：两次 invoke 只带新消息，验证 turns 与历史来自存档帧而不是 Python 变量。"""
    try:
        result = await day11_checkpointer.run_basic(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day11/chat",
    response_model=APIResponse[Day11ChatResult],
    summary="Day 11 交互式续跑：同 thread_id 跨请求读档（MemorySaver）",
)
async def chat_day11(
    payload: Day11ChatRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day11ChatResult]:
    """每轮只传一条 HumanMessage 和同一个 thread_id；历史与计数由 checkpointer 带回来。"""
    try:
        result = await day11_checkpointer.chat(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day11/approval/start",
    response_model=APIResponse[Day11ApprovalStartResult],
    summary="Day 11 确认门：命中敏感工具后 interrupt() 冻结并存档",
)
async def start_day11_approval(
    payload: Day11ApprovalStartRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day11ApprovalStartResult]:
    """图停在 approval 节点，返回待审批载荷；进程可以死，thread_id 不变就能继续。"""
    try:
        result = await day11_checkpointer.start_approval(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day11/approval/resume",
    response_model=APIResponse[Day11ApprovalResumeResult],
    summary="Day 11 恢复审批：Command(resume=...) 续跑（支持改参数后放行）",
)
async def resume_day11_approval(
    payload: Day11ApprovalResumeRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day11ApprovalResumeResult]:
    """decision=true 放行 / false 拦截 / 对象表示人工改过参数。

    注意：恢复时 approval 节点会从头重放，但它里面只有 interrupt，所以不会重复扣款。
    """
    try:
        result = await day11_checkpointer.resume_approval(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.get(
    "/day11/approval/pending",
    response_model=APIResponse[list[Day11PendingOut]],
    summary="Day 11 练习 3：待审批队列（含已等待秒数）",
)
async def list_day11_pending() -> APIResponse[list[Day11PendingOut]]:
    """列出所有停在 approval 的会话；wait_seconds 超过 86400 即视为超时。"""
    return APIResponse(data=[Day11PendingOut(**row) for row in day11_checkpointer.list_pending()])


@router.post(
    "/day11/approval/expire",
    response_model=APIResponse[Day11ExpireResult],
    summary="Day 11 练习 3：审批超时自动驳回（Command(resume=False)）",
)
async def expire_day11_approvals(
    payload: Day11ExpireRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day11ExpireResult]:
    """模拟定时任务：扫描等待超时的会话，自动拒绝并让模型生成"审批超时已驳回"的回复。"""
    try:
        result = await day11_checkpointer.expire_approvals(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day11/replay",
    response_model=APIResponse[Day11ReplayResult],
    summary="Day 11 坑 1：interrupt 前有副作用会被重放两次（纯本地）",
)
async def run_day11_replay() -> APIResponse[Day11ReplayResult]:
    """对照实验：副作用写在 interrupt 之前（执行 2 次）vs 拆成决策/审批/执行（执行 1 次）。"""
    try:
        result = await day11_checkpointer.replay_demo()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.post(
    "/day11/time-travel",
    response_model=APIResponse[Day11TimeTravelResult],
    summary="Day 11 时间旅行：get_state_history → update_state → invoke(None)",
)
async def run_day11_time_travel(
    payload: Day11TimeTravelRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day11TimeTravelResult]:
    """从 answer 之前的那一帧改写 topic 后重跑，原历史帧保留（分叉不是覆盖）。"""
    try:
        result = await day11_checkpointer.run_time_travel(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day11/sqlite",
    response_model=APIResponse[Day11SqliteResult],
    summary="Day 11 落盘存档：AsyncSqliteSaver，换连接（≈ 换进程）仍能续跑",
)
async def run_day11_sqlite(
    payload: Day11SqliteRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day11SqliteResult]:
    """每次请求重开 SQLite 连接并重新编译图，历史从 var/day11_checkpoints.db 读。"""
    try:
        result = await day11_checkpointer.run_sqlite(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


# ---------- Day 12 多智能体 Supervisor ----------


@router.get(
    "/day12/mapping",
    response_model=APIResponse[Day12MappingResult],
    summary="Day 12 概念数据：三堵墙 / 三角色 / 交接方式 / 状态三层 / 三层终止 / 五个坑 / 选型与成本",
)
async def get_day12_mapping() -> APIResponse[Day12MappingResult]:
    """返回静态概念数据：单 Agent 三堵墙、Supervisor 三角色分工、三种交接方式、双层状态、三层终止、五个坑。"""
    return APIResponse(data=await day12_supervisor.mapping())


@router.post(
    "/day12/single",
    response_model=APIResponse[Day12SingleResult],
    summary="Day 12 课程 01：单 Agent 复现（假收敛 / 越权 / 工具返回值被丢弃）",
)
async def run_day12_single(
    payload: Day12SingleRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day12SingleResult]:
    """一个"什么都会"的 Agent：wide 给 5 个工具，narrow 只给 3 个只读工具，对比 issues 的差异。"""
    try:
        result = await day12_supervisor.run_single(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day12/team",
    response_model=APIResponse[Day12SupervisorResult],
    summary="Day 12 课程 02：最小 Supervisor（主管只派活 + 专家只干专活 + FINISH）",
)
async def run_day12_team(
    payload: Day12SupervisorRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day12SupervisorResult]:
    """主管用 Route 结构化输出派单（没有 content 字段），专家私有执行后只回传结论 + name。

    ``dedupe_guard`` 是练习 1：连续两次派给同一人且任务高度相似时跳过本次派单转 writer。
    """
    try:
        result = await day12_supervisor.run_team(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day12/full/start",
    response_model=APIResponse[Day12FullStartResult],
    summary="Day 12 课程 03：完整版跑到退款审批处冻结（审批 + 预算 + 存档）",
)
async def start_day12_full(
    payload: Day12FullRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day12FullStartResult]:
    """三个专家（researcher / coder / writer）+ 交接预算 + MemorySaver，写操作前 interrupt() 等人。"""
    try:
        result = await day12_supervisor.start_full(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day12/full/resume",
    response_model=APIResponse[Day12FullResumeResult],
    summary="Day 12 恢复审批：Command(resume=...) 续跑到 writer 成稿",
)
async def resume_day12_full(
    payload: Day12FullResumeRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day12FullResumeResult]:
    """decision=true 放行 / false 驳回 / 对象改参数后放行；退款流水只在 resume 之后 append 一次。"""
    try:
        result = await day12_supervisor.resume_full(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day12/parallel",
    response_model=APIResponse[Day12ParallelResult],
    summary="Day 12 课程 04 / 练习 3：Send 扇出两条调研分支 → join → writer 只调一次",
)
async def run_day12_parallel(
    payload: Day12ParallelRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day12ParallelResult]:
    """政策与订单两条分支并行调研，findings 由 operator.add 在 join 处合并。"""
    try:
        result = await day12_supervisor.run_parallel(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


# ---------- Day 13 流式输出 ----------


@router.get(
    "/day13/mapping",
    response_model=APIResponse[Day13MappingResult],
    summary="Day 13 概念数据：四模式 / API 选型 / interrupt 规则 / SSE 三件套 / 事件协议 / 五个坑",
)
async def get_day13_mapping() -> APIResponse[Day13MappingResult]:
    """返回静态概念数据：stream_mode 五种取值、三个 API 怎么选、暂停信号的三种探测方式、SSE 三件套。"""
    return APIResponse(data=await day13_stream.mapping())


@router.post(
    "/day13/modes",
    response_model=APIResponse[Day13ModesResult],
    summary="Day 13 课程 01：values / updates / messages 四模式对照（同一张图分别订阅）",
)
async def run_day13_modes(
    payload: Day13ModesRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day13ModesResult]:
    """同一张最小图跑三趟：values、updates、messages+updates（多模式共用一趟，省一半调用）。"""
    try:
        result = await day13_stream.run_modes(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day13/token-stream",
    response_model=APIResponse[Day13TokenResult],
    summary="Day 13 课程 02：多 Agent token 按 langgraph_node 分流 + 过滤主管 JSON",
)
async def run_day13_token_stream(
    payload: Day13TokenRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day13TokenResult]:
    """三个 LLM 一起吐 token：每个专家一个气泡，主管的 json 碎片全部过滤掉（附样例）。"""
    try:
        result = await day13_stream.run_token_stream(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day13/custom-stream",
    response_model=APIResponse[Day13CustomResult],
    summary="Day 13 课程 03：custom 模式——get_stream_writer 工具内进度上报",
)
async def run_day13_custom_stream(
    payload: Day13CustomRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day13CustomResult]:
    """订阅 custom + messages + updates：检索进度与 token 排在同一条时间轴上。"""
    try:
        result = await day13_stream.run_custom_stream(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day13/interrupt-stream",
    response_model=APIResponse[Day13InterruptResult],
    summary="Day 13 课程 04：interrupt + 流式同开（只订 messages 会漏掉 __interrupt__）",
)
async def run_day13_interrupt_stream(
    payload: Day13InterruptRequest,
) -> APIResponse[Day13InterruptResult]:
    """三次订阅对照：只订 messages（漏暂停）/ 双订阅（收到 __interrupt__）/ Command(resume=) 续跑。

    该实验不调用模型，结果完全确定：图停在哪、退款记几次都可以断言。
    """
    try:
        result = await day13_stream.run_interrupt_stream(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.get(
    "/day13/sse/chat",
    response_class=StreamingResponse,
    summary="Day 13 课程 05：LangGraph 事件 → SSE → 浏览器（token / node_start / done）",
)
async def sse_day13_chat(
    q: str = "用三句话介绍一下 SSE 流式输出",
    model_config_id: int | None = None,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """真流式接口：浏览器用 ``EventSource`` 消费。

    模型配置在返回响应之前解析（没有可用模型时仍能返回正常的 400）；
    之后的错误只能通过 SSE 的 ``error`` 事件告知前端。
    """
    try:
        cfg = await day13_stream.sse_model(db, model_config_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        day13_stream.sse_chat(cfg, q),
        media_type="text/event-stream",
        headers=day13_stream.sse_headers(),
    )


# ---------- Day 14 MCP 协议 ----------


@router.get(
    "/day14/mapping",
    response_model=APIResponse[Day14MappingResult],
    summary="Day 14 概念数据：M×N / FC vs MCP / 三角色 / 两传输 / 三原语 / 迁移对照 / 五个坑",
)
async def get_day14_mapping() -> APIResponse[Day14MappingResult]:
    """返回静态概念数据：为什么需要 MCP、FC 与 MCP 分工、三角色两传输、三种原语、五个坑。"""
    return APIResponse(data=await day14_mcp.mapping())


@router.post(
    "/day14/discover",
    response_model=APIResponse[Day14DiscoverResult],
    summary="Day 14 课程 01/02：spawn 两个 stdio Server，协议视角 + LangChain 视角一起看",
)
async def run_day14_discover() -> APIResponse[Day14DiscoverResult]:
    """真实拉起两个子进程，走 initialize → tools/list → resources/list → get_tools()。"""
    try:
        result = await day14_mcp.discover()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"MCP 子进程通信失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day14/call",
    response_model=APIResponse[Day14CallResult],
    summary="Day 14：直接发 tools/call（不经模型），验证 Server 端校验与『错误即结果』",
)
async def call_day14_tool(payload: Day14CallRequest) -> APIResponse[Day14CallResult]:
    """手工指定 server / tool / arguments，看 Server 怎么把校验失败变成 isError 文本。"""
    try:
        result = await day14_mcp.call_tool(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"MCP 子进程通信失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day14/agent",
    response_model=APIResponse[Day14AgentResult],
    summary="Day 14 课程 03：一个 Agent 横跨两个 MCP Server（查 CRM + 查文档）",
)
async def run_day14_agent(
    payload: Day14AgentRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day14AgentResult]:
    """get_tools() 把两个 Server 的工具转成扁平列表交给 create_agent，模型层毫无感知差异。"""
    try:
        result = await day14_mcp.run_agent(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型或 MCP 调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day14/mixed",
    response_model=APIResponse[Day14MixedResult],
    summary="Day 14 课程 05：MCP 工具 + 本地工具混装（余额不足 → 转人工），流式照常",
)
async def run_day14_mixed(
    payload: Day14MixedRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day14MixedResult]:
    """扣款被 CRM Server 拦住后，Agent 改调本地 escalate_to_human——两种来源同一条轨迹。"""
    try:
        result = await day14_mcp.run_mixed(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"模型或 MCP 调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day14/http",
    response_model=APIResponse[Day14HttpResult],
    summary="Day 14 课程 04：streamable-http 共享服务（临时 uvicorn，随机端口，用完即关）",
)
async def run_day14_http(payload: Day14HttpRequest) -> APIResponse[Day14HttpResult]:
    """后台线程起一个临时 MCP HTTP Server，客户端用 transport="http" 连上再调用。"""
    try:
        result = await day14_mcp.run_http(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"MCP HTTP 服务失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day14/noisy",
    response_model=APIResponse[Day14NoisyResult],
    summary="Day 14 坑 1 现场：stdout 污染协议通道的三种形态实测",
)
async def run_day14_noisy(payload: Day14NoisyRequest) -> APIResponse[Day14NoisyResult]:
    """print / 整行写 fd1 / 半行写 fd1 各跑一遍，用结果说明『stdout 是协议通道』。"""
    try:
        result = await day14_mcp.run_noisy(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


# ---------- Day 15 可观测性 ----------


@router.get(
    "/day15/mapping",
    response_model=APIResponse[Day15MappingResult],
    summary="Day 15 概念数据：失败形态 / 三大件 / 两条路线 / 八字段 / 五个坑 / 平台就绪状态",
)
async def get_day15_mapping() -> APIResponse[Day15MappingResult]:
    """返回静态概念数据 + LangSmith / Langfuse 的配置状态（只读环境变量）。"""
    return APIResponse(data=await day15_observability.mapping())


@router.post(
    "/day15/callback",
    response_model=APIResponse[Day15CallbackResult],
    summary="Day 15 课程 02：自建 CallbackHandler，从回调事件还原 trace 树",
)
async def run_day15_callback(
    payload: Day15CallbackRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day15CallbackResult]:
    """跑一次真实 Agent，用自定义 Handler 记录 LLM / 工具生命周期并还原调用树。"""
    try:
        result = await day15_observability.run_callback(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"回调追踪失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day15/platform",
    response_model=APIResponse[Day15PlatformResult],
    summary="Day 15 课程 01/03：LangSmith 零代码接入 / Langfuse Handler + 用户反馈打分",
)
async def run_day15_platform(
    payload: Day15PlatformRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day15PlatformResult]:
    """没配 key 时只返回配置说明（不跑模型、不改环境变量）；配好才真实上报。"""
    try:
        result = await day15_observability.run_platform(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"平台接入失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day15/own-trace",
    response_model=APIResponse[Day15OwnTraceResult],
    summary="Day 15 课程 04：自建 span + JSONL 落盘，回放成树并聚合指标",
)
async def run_day15_own_trace(
    payload: Day15OwnTraceRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day15OwnTraceResult]:
    """手写循环包 span（含 token 记账）→ 写 traces.jsonl → 建树 + p50/p95 + 成本。"""
    try:
        result = await day15_observability.run_own_trace(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"自建追踪失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day15/full-stack",
    response_model=APIResponse[Day15FullStackResult],
    summary="Day 15 课程 05：LangGraph + MCP 全链路（trace_id 跨进程传播）",
)
async def run_day15_full_stack(
    payload: Day15FullStackRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day15FullStackResult]:
    """客户端与 MCP Server 各写 JSONL，靠同一个 trace_id 合并成完整时间轴。"""
    try:
        result = await day15_observability.run_full_stack(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"全链路追踪失败：{exc}") from exc
    return APIResponse(data=result)


@router.get(
    "/day15/traces",
    response_model=APIResponse[Day15TracesResult],
    summary="Day 15：回放最近 N 条自建 trace + 全局指标 + 用户反馈聚合",
)
async def list_day15_traces(limit: int = 5) -> APIResponse[Day15TracesResult]:
    """从 traces.jsonl 读回记录：按 trace_id 分组、还原树、聚合 p50/p95 与成本。"""
    safe_limit = max(1, min(limit, 20))
    return APIResponse(data=day15_observability.list_traces(safe_limit))


@router.post(
    "/day15/feedback",
    response_model=APIResponse[Day15FeedbackResult],
    summary="Day 15：把用户点赞/点踩落到 trace 上（自建版 score）",
)
async def submit_day15_feedback(payload: Day15FeedbackRequest) -> APIResponse[Day15FeedbackResult]:
    """写 feedback.jsonl 并返回点赞率——这是 Day 16 评估集的第一批真值标签。"""
    return APIResponse(data=day15_observability.record_feedback(payload))


# ---------- Day 16 评估体系 ----------


@router.get(
    "/day16/mapping",
    response_model=APIResponse[Day16MappingResult],
    summary="Day 16 概念数据：三个困难 / 三层评估 / 指标体系 / 数据来源 / rubric / 五个坑 / 门禁阈值",
)
async def get_day16_mapping() -> APIResponse[Day16MappingResult]:
    """返回静态概念数据 + 两版提示词 + 当前数据集画像（不调模型）。"""
    return APIResponse(data=day16_eval.mapping())


@router.post(
    "/day16/dataset/build",
    response_model=APIResponse[Day16DatasetResult],
    summary="Day 16 课程 01：从 Day 15 点踩 trace + 人工边界 + 合成变体长成评估集",
)
async def build_day16_dataset(payload: Day16BuildRequest) -> APIResponse[Day16DatasetResult]:
    """按来源优先级合并去重后落盘 dataset.jsonl；每类 case 带稳定的 id。"""
    try:
        result = await day16_eval.build_dataset(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.get(
    "/day16/dataset",
    response_model=APIResponse[Day16DatasetResult],
    summary="Day 16：读回当前数据集与类别 / 难度 / 来源分布",
)
async def get_day16_dataset() -> APIResponse[Day16DatasetResult]:
    """文件不存在时返回 total=0 与「先构建」的提示，页面不报错。"""
    return APIResponse(data=day16_eval.list_dataset())


@router.post(
    "/day16/run",
    response_model=APIResponse[Day16RunResult],
    summary="Day 16 课程 02：跑评估——每 case 多次采样取多数，结果 + 过程指标一起采",
)
async def run_day16_eval(
    payload: Day16RunRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day16RunResult]:
    """逐 case 跑 Agent 并按 规则 / 工具序列 / 行为断言 / 收敛 综合判定，报告落盘。"""
    try:
        result = await day16_eval.run_eval(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001  上游模型服务失败
        raise HTTPException(status_code=502, detail=f"评估运行失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day16/judge",
    response_model=APIResponse[Day16JudgeResult],
    summary="Day 16 课程 03：LLM-as-judge——结构化 rubric 打分 + pairwise 换序去偏",
)
async def run_day16_judge(
    payload: Day16JudgeRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day16JudgeResult]:
    """judge 按 default→function_calling→json_mode 降级，全失败才走可观察特征兜底。"""
    try:
        result = await day16_eval.run_judge(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"judge 调用失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day16/regression",
    response_model=APIResponse[Day16RegressionResult],
    summary="Day 16 课程 04：baseline vs candidate 双跑 + 逐 case diff + 门禁判定",
)
async def run_day16_regression(
    payload: Day16RegressionRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day16RegressionResult]:
    """同一数据集、同一模型，只换 prompt；门禁看正确率降幅与 token 涨幅。"""
    try:
        result = await day16_eval.run_regression(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"回归运行失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day16/ci-gate",
    response_model=APIResponse[Day16CiResult],
    summary="Day 16 课程 05：CI 冒烟门禁——分层抽样冒烟集 + 与归档基线比较 + exit_code 语义",
)
async def run_day16_ci_gate(
    payload: Day16CiRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day16CiResult]:
    """真实 CI 里这一步 sys.exit(exit_code)；HTTP 版把退出码放进返回值。"""
    try:
        result = await day16_eval.run_ci_gate(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"冒烟门禁运行失败：{exc}") from exc
    return APIResponse(data=result)


# ---------- Day 17 成本与延迟优化 ----------


@router.get(
    "/day17/mapping",
    response_model=APIResponse[Day17MappingResult],
    summary="Day 17 概念数据：成本结构 / 延迟四段 / 四杠杆 / 缓存三层 / 六个坑 / 价格与阈值",
)
async def get_day17_mapping() -> APIResponse[Day17MappingResult]:
    """返回静态概念数据 + 价格表 + 预算三闸默认值（不调模型）。"""
    return APIResponse(data=day17_cost.mapping())


@router.post(
    "/day17/ledger",
    response_model=APIResponse[Day17LedgerResult],
    summary="Day 17 课程 01：成本账本——逐步 in/out token、放大率、重复发送占比",
)
async def run_day17_ledger(
    payload: Day17LedgerRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day17LedgerResult]:
    """默认纯估算（不花钱）；勾选 live 时真调一次模型，返回估算 vs 实测的偏差。"""
    try:
        result = await day17_cost.run_ledger(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"账本运行失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day17/router",
    response_model=APIResponse[Day17RouterResult],
    summary="Day 17 课程 02：模型分级路由——规则 / 小模型分类 / 三信号升级链",
)
async def run_day17_router(
    payload: Day17RouterRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day17RouterResult]:
    """规则路由零成本；勾选 use_llm 才真调模型做分类器（会花钱，但也是课堂要看的效果）。"""
    try:
        result = await day17_cost.router_demo(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"路由实验失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day17/compress",
    response_model=APIResponse[Day17CompressResult],
    summary="Day 17 课程 03：上下文压缩——静态瘦身 + 工具结果剪裁 + 预算触发摘要",
)
async def run_day17_compress(
    payload: Day17CompressRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day17CompressResult]:
    """默认走抽取式摘要（不花钱）；勾选 use_model 才用小模型做摘要。"""
    try:
        result = await day17_cost.compress_demo(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"压缩实验失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day17/cache",
    response_model=APIResponse[Day17CacheResult],
    summary="Day 17 课程 04：缓存三层——只读白名单 / 实体校验 / 用户隔离 / 前缀哈希",
)
async def run_day17_cache(payload: Day17CacheRequest) -> APIResponse[Day17CacheResult]:
    """纯逻辑实验（不调模型、不发网络请求）：三个安全阀的拦截现场都能看到原因。"""
    return APIResponse(data=day17_cost.cache_demo(payload))


@router.post(
    "/day17/parallel",
    response_model=APIResponse[Day17ParallelResult],
    summary="Day 17 课程 05：并行工具——串行 vs 并行耗时、三道保险与时间轴",
)
async def run_day17_parallel(payload: Day17ParallelRequest) -> APIResponse[Day17ParallelResult]:
    """真实 asyncio IO 模拟（1 秒级），返回实测加速比与逐工具时间轴。"""
    try:
        result = await day17_cost.parallel_demo(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.post(
    "/day17/budget-agent",
    response_model=APIResponse[Day17BudgetResult],
    summary="Day 17 课程 06：带三闸预算熔断的 Agent（熔断 → 摘工具 → 降级交付）",
)
async def run_day17_budget_agent(
    payload: Day17BudgetRequest, db: AsyncSession = Depends(get_db)
) -> APIResponse[Day17BudgetResult]:
    """simulate 是确定性脚本化模型（必然熔断，课堂看逻辑）；live 才真调模型。"""
    try:
        result = await day17_cost.budget_demo(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"预算熔断实验失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day17/ab",
    response_model=APIResponse[Day17AbResult],
    summary="Day 17 课程 06：四个杠杆逐档现算成本与延迟（正确率交给 Day 16 门禁）",
)
async def run_day17_ab(payload: Day17AbRequest, db: AsyncSession = Depends(get_db)) -> APIResponse[Day17AbResult]:
    """不调模型：按同一套参数现算，报告落盘 var/day17/ab_report.json。"""
    try:
        result = await day17_cost.ab_demo(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


# ============================================================
# Day 18 安全防护：注入 / 围栏 / 权限 / 外泄 / 确认门 / 沙箱 / 流水线
# ============================================================


@router.get(
    "/day18/mapping",
    response_model=APIResponse[Day18MappingResult],
    summary="Day 18 概念数据：五类威胁 / 三条原则 + fail-closed / 攻击面 / 坑 / 速查表 / 沙箱矩阵",
)
async def get_day18_mapping() -> APIResponse[Day18MappingResult]:
    """返回静态概念数据 + canary + 落盘路径（不调模型、不跑子进程）。"""
    return APIResponse(data=day18_security.mapping())


@router.post(
    "/day18/injection",
    response_model=APIResponse[Day18InjectionResult],
    summary="Day 18 课程 01：间接注入复现（无害输入 + 有毒工具结果 → 转走 9999）",
)
async def run_day18_injection(payload: Day18InjectionRequest) -> APIResponse[Day18InjectionResult]:
    """纯脚本化模型（不花钱、可复现）：同一条毒工单跑 naive / fence_only / guarded 三遍。"""
    try:
        result = day18_security.injection_demo(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.post(
    "/day18/fence",
    response_model=APIResponse[Day18FenceResult],
    summary="Day 18 课程 02：围栏 + 双向扫描（入站命中 / 出站越界 / 变形样本矩阵）",
)
async def run_day18_fence(payload: Day18FenceRequest) -> APIResponse[Day18FenceResult]:
    """纯逻辑：围栏、归一化、规则命中、行为校验，全部可单测。"""
    return APIResponse(data=day18_security.fence_demo(payload))


@router.post(
    "/day18/authz",
    response_model=APIResponse[Day18AuthzResult],
    summary="Day 18 课程 03：最小权限与 IDOR（身份网关注入 / 越权 NotFound / 好签名坏签名对照）",
)
async def run_day18_authz(payload: Day18AuthzRequest) -> APIResponse[Day18AuthzResult]:
    """纯逻辑：按 role/scopes 发工具、行级校验、收款账户服务端解析。"""
    return APIResponse(data=day18_security.authz_demo(payload))


@router.post(
    "/day18/exfil",
    response_model=APIResponse[Day18ExfilResult],
    summary="Day 18 课程 04：数据外泄（markdown 图片默认拦 / 裸链去活化 / PII 脱敏 / canary）",
)
async def run_day18_exfil(payload: Day18ExfilRequest) -> APIResponse[Day18ExfilResult]:
    """纯逻辑出站清洗，并附 8 条自检样本对照。"""
    return APIResponse(data=day18_security.exfil_demo(payload))


@router.post(
    "/day18/approval",
    response_model=APIResponse[Day18ApprovalResult],
    summary="Day 18 课程 05：危险操作确认门（interrupt 在副作用之前 + 幂等账本 + 闸位置对照）",
)
def run_day18_approval(payload: Day18ApprovalRequest) -> APIResponse[Day18ApprovalResult]:
    """同步执行（LangGraph 同步 invoke）：FastAPI 会放进线程池，不阻塞事件循环。"""
    try:
        result = day18_security.approval_demo(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"确认门实验失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day18/sandbox",
    response_model=APIResponse[Day18SandboxResult],
    summary="Day 18 课程 06：代码执行沙箱（独立进程 + 资源限制，并明确挡不住什么）",
)
def run_day18_sandbox(payload: Day18SandboxRequest) -> APIResponse[Day18SandboxResult]:
    """跑预设攻击样例或自定义代码；每个样例都有超时兜底。"""
    try:
        result = day18_security.sandbox_demo(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"沙箱实验失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day18/pipeline",
    response_model=APIResponse[Day18PipelineResult],
    summary="Day 18 课程 07：安全中间件流水线（六道中间件，任一 deny 立即短路）",
)
async def run_day18_pipeline(payload: Day18PipelineRequest) -> APIResponse[Day18PipelineResult]:
    """纯逻辑：结论 + 短路点 + 逐中间件 audit + 清洗后的上下文。"""
    return APIResponse(data=day18_security.pipeline_demo(payload))


@router.get(
    "/day18/attack-suite",
    response_model=APIResponse[Day18SuiteResult],
    summary="Day 18 课程 07：攻击集回归（10 条攻击的期望 vs 实际 + 通过率）",
)
async def run_day18_attack_suite() -> APIResponse[Day18SuiteResult]:
    """安全也要有回归集：改 prompt / 换模型 / 加工具都要重跑这 10 条。"""
    return APIResponse(data=day18_security.attack_suite())


# ============================================================
# Day 19 服务化部署：并发 / SSE / 会话 / run 管理 / 限流 / 灰度 / 总装
# ============================================================


@router.get(
    "/day19/mapping",
    response_model=APIResponse[Day19MappingResult],
    summary="Day 19 概念数据：服务差异 / 四层架构 / 并发模型 / SSE / 三道闸 / 存储 / 坑 / 速查表",
)
async def get_day19_mapping() -> APIResponse[Day19MappingResult]:
    """返回静态概念数据 + 依赖版本（不调模型、不起服务、不写文件）。"""
    return APIResponse(data=day19_service.mapping())


@router.post(
    "/day19/concurrency",
    response_model=APIResponse[Day19ConcurrencyResult],
    summary="Day 19 课程 01：同步阻塞 vs 协程并发（真实计时）+ 容量数学与雪崩推演",
)
async def run_day19_concurrency(
    payload: Day19ConcurrencyRequest,
) -> APIResponse[Day19ConcurrencyResult]:
    """worker 串行会线性翻倍，协程并发只等一次；容量 = worker 数 ÷ 单 run 时长。"""
    result = await day19_service.concurrency_demo(payload)
    return APIResponse(data=result)


@router.get(
    "/day19/sse/chat",
    response_class=StreamingResponse,
    summary="Day 19 课程 02：生产级 SSE（心跳 + 递增 id + meta/node/token/done/error + 断线续发）",
)
async def sse_day19_chat(
    q: str = "订单 A1001 到哪了？",
    slow_ms: int = 1200,
    heartbeat_ms: int = 400,
    fail: bool = False,
    last_event_id: int | None = None,
    last_event_id_header: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    """脚本化事件流（不调模型）：用 EventSource 直连即可观察心跳与断线续发。

    ``last_event_id`` 优先取查询参数，其次取浏览器重连自动带上的 ``Last-Event-ID`` 头；
    只补发 id 更大的事件，前端因此能「接着看」而不是从头跑。
    """
    resume_from = last_event_id if last_event_id is not None else -1
    if last_event_id_header:
        try:
            resume_from = max(resume_from, int(last_event_id_header))
        except ValueError:
            pass  # 头不合法就当没带，继续全量发送
    plan = day19_service.sse_plan(
        q[:200],
        slow_ms=max(0, min(slow_ms, 5000)),
        fail=fail,
    )
    return StreamingResponse(
        day19_service.sse_stream(
            plan,
            heartbeat_ms=max(100, min(heartbeat_ms, 30000)),
            last_event_id=resume_from,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/day19/sse/preview",
    response_model=APIResponse[Day19SsePreviewResult],
    summary="Day 19 课程 02：SSE 事件计划预览（原始帧 + 心跳 + Last-Event-ID 续发）",
)
async def preview_day19_sse(
    payload: Day19SsePreviewRequest,
) -> APIResponse[Day19SsePreviewResult]:
    """非流式预览：不连网也能看帧格式、心跳位置与断线后补发条数。"""
    return APIResponse(data=day19_service.sse_preview(payload))


@router.post(
    "/day19/session",
    response_model=APIResponse[Day19SessionResult],
    summary="Day 19 课程 03：thread_id 拆包 + 归属校验矩阵 + 审批 scope + 恢复载荷",
)
async def run_day19_session(payload: Day19SessionRequest) -> APIResponse[Day19SessionResult]:
    """纯逻辑权限矩阵：越权返 404、无 scope 返 403、载荷由服务端构造。"""
    return APIResponse(data=day19_service.session_demo(payload))


@router.post(
    "/day19/runs",
    response_model=APIResponse[Day19RunsResult],
    summary="Day 19 课程 04：run 生命周期（背压 503 / 幂等重放 / 软合并 / 越权轮询 / 取消）",
)
async def run_day19_runs(payload: Day19RunsRequest) -> APIResponse[Day19RunsResult]:
    """真实异步：多个 run 并发推进，超出容量立即 503，取消停在当前超步。"""
    try:
        result = await day19_service.run_manager_demo(payload)
    except Exception as exc:  # noqa: BLE001  管理器自身异常不应把接口打成 500
        raise HTTPException(status_code=502, detail=f"run 管理实验失败：{exc}") from exc
    return APIResponse(data=result)


@router.post(
    "/day19/rate-limit",
    response_model=APIResponse[Day19RateLimitResult],
    summary="Day 19 课程 05：令牌桶突发与回填 + 租户月配额校验/结算",
)
async def run_day19_rate_limit(
    payload: Day19RateLimitRequest,
) -> APIResponse[Day19RateLimitResult]:
    """纯逻辑：429 与 Retry-After、402 配额语义、只校验不扣减的反面对照。"""
    return APIResponse(data=day19_service.rate_limit_demo(payload))


@router.post(
    "/day19/rollout",
    response_model=APIResponse[Day19RolloutResult],
    summary="Day 19 课程 06：prompt 版本化 + hash 稳定灰度 + 门禁 + 30 秒回滚",
)
async def run_day19_rollout(payload: Day19RolloutRequest) -> APIResponse[Day19RolloutResult]:
    """纯逻辑：同一用户永远命中同一版本，分布接近百分比，门禁不过禁止放量。"""
    return APIResponse(data=day19_service.rollout_demo(payload))


@router.get(
    "/day19/skeleton",
    response_model=APIResponse[Day19SkeletonResult],
    summary="Day 19 课程 07：总装骨架（四层组件 / 请求生命周期 / 端点表 / 可抄代码）",
)
async def get_day19_skeleton() -> APIResponse[Day19SkeletonResult]:
    """麻雀虽小四层俱全的最小服务骨架，可直接当新项目起点。"""
    return APIResponse(data=day19_service.skeleton())


# ============================================================
# Day 20 可靠性：分类重试 / 降级阶梯 / 优雅退出 / 容量 / SLO / 混沌
# ============================================================


@router.get(
    "/day20/mapping",
    response_model=APIResponse[Day20MappingResult],
    summary="Day 20 概念数据：故障四分类 / 重试三要素 / 降级阶梯 / 退出时序 / 容量天花板 / SLO / 坑 / 速查表",
)
async def get_day20_mapping() -> APIResponse[Day20MappingResult]:
    """返回静态概念数据 + runbook（不调模型、不起服务、不写文件）。"""
    return APIResponse(data=day20_reliability.mapping())


@router.post(
    "/day20/retry",
    response_model=APIResponse[Day20RetryResult],
    summary="Day 20 课程 01：错误四分类 + 退避抖动重试 + run 级总预算（等待为模拟，不真睡）",
)
async def run_day20_retry(payload: Day20RetryRequest) -> APIResponse[Day20RetryResult]:
    """确定性失败立即上抛、429 尊重 Retry-After、解析失败回填 prompt、耗尽走降级出口。"""
    return APIResponse(data=day20_reliability.retry_demo(payload))


@router.post(
    "/day20/ladder",
    response_model=APIResponse[Day20LadderResult],
    summary="Day 20 课程 02：降级阶梯 L0→L1→L2→L3（能力对齐 / 缓存铁律 / 探活防乒乓）",
)
async def run_day20_ladder(payload: Day20LadderRequest) -> APIResponse[Day20LadderResult]:
    """纯逻辑：备模型不支持 FC 时跳过该级；降级答案不回写缓存；连续成功才回主路。"""
    return APIResponse(data=day20_reliability.ladder_demo(payload))


@router.post(
    "/day20/shutdown",
    response_model=APIResponse[Day20ShutdownResult],
    summary="Day 20 课程 03：优雅退出时序（摘流量 → 503 → drain → 落盘 → flush）",
)
async def run_day20_shutdown(payload: Day20ShutdownRequest) -> APIResponse[Day20ShutdownResult]:
    """确定性时间线模拟：谁跑完、谁落盘可续跑、谁被撕碎，以及 preStop 够不够长。"""
    return APIResponse(data=day20_reliability.shutdown_demo(payload))


@router.post(
    "/day20/capacity",
    response_model=APIResponse[Day20CapacityResult],
    summary="Day 20 课程 04：三道天花板 → Semaphore 定容 + 恒定速率压测读数与决策",
)
async def run_day20_capacity(payload: Day20CapacityRequest) -> APIResponse[Day20CapacityResult]:
    """钱 / 上游配额 / 机器资源取最小；排队模拟给出 p50/p95/503，按读数决策表给动作。"""
    return APIResponse(data=day20_reliability.capacity_demo(payload))


@router.post(
    "/day20/slo",
    response_model=APIResponse[Day20SloResult],
    summary="Day 20 课程 05：trace → 窗口指标 → 烧毁率 → 告警判定 + runbook",
)
async def run_day20_slo(payload: Day20SloRequest) -> APIResponse[Day20SloResult]:
    """合成同构 span（含一行坏 JSON）：最小样本量守门，短窗口 14.4x / 长窗口 6x 两档。"""
    return APIResponse(data=day20_reliability.slo_demo(payload))


@router.post(
    "/day20/chaos",
    response_model=APIResponse[Day20ChaosResult],
    summary="Day 20 课程 06：混沌演练（注入 + 预期声明表 + 盲区清单）",
)
async def run_day20_chaos(payload: Day20ChaosRequest) -> APIResponse[Day20ChaosResult]:
    """模拟注入下的 run，并逐条核对五条预期：判定落空处就是可靠性盲区。"""
    return APIResponse(data=day20_reliability.chaos_demo(payload))


# ============================================================
# Day 21 规划能力：失败复现 / 计划校验 / L1 / L2 / L3 / 图编排 / 评估门禁
# ============================================================


@router.get(
    "/day21/mapping",
    response_model=APIResponse[Day21MappingResult],
    summary="Day 21 概念数据：三症状 / ReAct 对照 / 计划字段 / 三不变量 / 三形态 / 坑 / 速查表",
)
async def get_day21_mapping() -> APIResponse[Day21MappingResult]:
    """返回静态概念数据；不调模型、不起服务、不写文件。"""
    return APIResponse(data=day21_planning.mapping())


@router.post(
    "/day21/failures",
    response_model=APIResponse[Day21FailuresResult],
    summary="Day 21 课程 01：复现三种「没有计划」的失败（监控全绿但答案错）",
)
async def run_day21_failures(payload: Day21FailuresRequest) -> APIResponse[Day21FailuresResult]:
    """剧本回放 ReAct 循环，逐条给出漏掉的事实与正确答案。"""
    return APIResponse(data=day21_planning.failures_demo(payload))


@router.post(
    "/day21/schema",
    response_model=APIResponse[Day21SchemaResult],
    summary="Day 21 课程 02：计划结构 + 三条不变量校验 + DAG 拓扑分层",
)
async def run_day21_schema(payload: Day21SchemaRequest) -> APIResponse[Day21SchemaResult]:
    """编译计划样本：非法计划在执行前就被拦下。"""
    return APIResponse(data=day21_planning.schema_demo(payload))


@router.post(
    "/day21/execute",
    response_model=APIResponse[Day21ExecuteResult],
    summary="Day 21 课程 03：L1 计划先行（Planner → Executor → Validator）",
)
async def run_day21_execute(payload: Day21ExecuteRequest) -> APIResponse[Day21ExecuteResult]:
    """逐层执行并机器校验；L1 治漏探查与硬扛，治不了走错不回头。"""
    return APIResponse(data=day21_planning.execute_demo(payload))


@router.post(
    "/day21/replan",
    response_model=APIResponse[Day21ReplanResult],
    summary="Day 21 课程 04：L2 偏离重规划（四动作 / 账本冻结 / 防乒乓）",
)
async def run_day21_replan(payload: Day21ReplanRequest) -> APIResponse[Day21ReplanResult]:
    """外采失败补自产分支；已完成步骤不可修改，重复补丁直接熔断。"""
    return APIResponse(data=day21_planning.replan_demo(payload))


@router.post(
    "/day21/reflexion",
    response_model=APIResponse[Day21ReflexionResult],
    summary="Day 21 课程 05：L3 反思回路（根因枚举 / 可执行指令 / 教训升级）",
)
async def run_day21_reflexion(payload: Day21ReflexionRequest) -> APIResponse[Day21ReflexionResult]:
    """纯逻辑演示算错→归因→重试，以及把业务结论写成判据造成的假失败。"""
    return APIResponse(data=day21_planning.reflexion_demo(payload))


@router.post(
    "/day21/graph",
    response_model=APIResponse[Day21GraphResult],
    summary="Day 21 课程 06：LangGraph 计划图（路由 / 反思 / 重规划 / 中断审批）",
)
async def run_day21_graph(payload: Day21GraphRequest) -> APIResponse[Day21GraphResult]:
    """真实状态图跑通路由循环；计划变更先人审，拒绝则走 abort。"""
    try:
        return APIResponse(data=day21_planning.graph_demo(payload))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/day21/eval",
    response_model=APIResponse[Day21EvalResult],
    summary="Day 21 课程 07：规划质量四指标 + A/B/C/D 四档对照 + CI 门禁",
)
async def run_day21_eval(payload: Day21EvalRequest) -> APIResponse[Day21EvalResult]:
    """同一份合成数据集跑四档方案，按三条阈值给门禁结论。"""
    return APIResponse(data=day21_planning.eval_demo(payload))


# ============================================================
# Day 22 代码执行沙箱：分工 / AST / L1 / 网络文件 / L2 / 代码循环 / 生产 / 压测
# ============================================================


@router.get(
    "/day22/mapping",
    response_model=APIResponse[Day22MappingResult],
    summary="Day 22 概念数据：七类威胁 / 五级隔离 / 五道闸 / 容器参数 / 六个坑 / 速查表",
)
async def get_day22_mapping() -> APIResponse[Day22MappingResult]:
    """返回静态概念表与当前平台的能力（resource / docker 是否可用）。"""
    return APIResponse(data=day22_sandbox.mapping())


@router.post(
    "/day22/why",
    response_model=APIResponse[Day22WhyResult],
    summary="Day 22 课程 01：模型心算错 + exec 清空 builtins 仍可逃逸（安全探针）",
)
async def run_day22_why(payload: Day22WhyRequest) -> APIResponse[Day22WhyResult]:
    """回放两种失败，并执行一条只取 pid 的安全逃逸链验证课程修正。"""
    return APIResponse(data=day22_sandbox.why_demo(payload))


@router.post(
    "/day22/ast",
    response_model=APIResponse[Day22AstResult],
    summary="Day 22 课程 02：AST 筛查的五个样本 + 四种绕过（lint 不是 jail）",
)
async def run_day22_ast(payload: Day22AstRequest) -> APIResponse[Day22AstResult]:
    """逐条给出放行/拦截与命中原因，暴露「手滑写文件没拦住」「两种绕过漏了」。"""
    return APIResponse(data=day22_sandbox.ast_demo(payload))


@router.post(
    "/day22/l1",
    response_model=APIResponse[Day22L1Result],
    summary="Day 22 课程 03：L1 子进程 + rlimit + 独立 cwd + 进程组清理",
)
async def run_day22_l1(payload: Day22L1Request) -> APIResponse[Day22L1Result]:
    """默认回放课程实测；mode=live 真起子进程，fork bomb 与 Windows 内存炸弹跳过。"""
    try:
        return APIResponse(data=day22_sandbox.l1_demo(payload))
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/day22/netfs",
    response_model=APIResponse[Day22NetfsResult],
    summary="Day 22 课程 04：软断网被 _socket 绕过，L1 可读可写宿主机文件",
)
async def run_day22_netfs(payload: Day22NetfsRequest) -> APIResponse[Day22NetfsResult]:
    """四条探针对照：哪条被拦、哪条墙根本不存在。"""
    try:
        return APIResponse(data=day22_sandbox.netfs_demo(payload))
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/day22/container",
    response_model=APIResponse[Day22ContainerResult],
    summary="Day 22 课程 05：L2 容器九个安全参数 + 六条隔离探针",
)
async def run_day22_container(payload: Day22ContainerRequest) -> APIResponse[Day22ContainerResult]:
    """生成完整 docker 命令并回放探针；live=true 且 docker 可用时跑一条纯计算探针。"""
    return APIResponse(data=day22_sandbox.container_demo(payload))


@router.post(
    "/day22/loop",
    response_model=APIResponse[Day22LoopResult],
    summary="Day 22 课程 06：写代码→执行→看报错→改代码（错误五要素回填）",
)
async def run_day22_loop(payload: Day22LoopRequest) -> APIResponse[Day22LoopResult]:
    """脚本化 coder + 假 runner，零 token 复现三次收敛与同类错误换策略。"""
    return APIResponse(data=day22_sandbox.loop_demo(payload))


@router.post(
    "/day22/bridge",
    response_model=APIResponse[Day22BridgeResult],
    summary="Day 22 练习 1：Day 21 的 s3 换真执行器，stdout 字符串解析后过机器判据",
)
async def run_day22_bridge(payload: Day22BridgeRequest) -> APIResponse[Day22BridgeResult]:
    """生成三行计算代码，执行后用 float(stdout) 交给 Day 21 的 verify()。"""
    try:
        return APIResponse(data=day22_sandbox.bridge_demo(payload))
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/day22/production",
    response_model=APIResponse[Day22ProductionResult],
    summary="Day 22 课程 07：画像 / AST / 缓存 / 池化 / 审计 / 指标的完整请求路径",
)
async def run_day22_production(payload: Day22ProductionRequest) -> APIResponse[Day22ProductionResult]:
    """按代码高危模式与任务类型选档，逐道闸记录放行或短路。"""
    return APIResponse(data=day22_sandbox.production_demo(payload))


@router.post(
    "/day22/pressure",
    response_model=APIResponse[Day22PressureResult],
    summary="Day 22 练习 2/3：十条攻击 × L1/L2 拦截矩阵 + 同类错误升级收益",
)
async def run_day22_pressure(payload: Day22PressureRequest) -> APIResponse[Day22PressureResult]:
    """把 L1 漏掉的 T1/T2 与「杀错进程」摆到台面上，附首次通过率对照。"""
    return APIResponse(data=day22_sandbox.pressure_demo(payload))
