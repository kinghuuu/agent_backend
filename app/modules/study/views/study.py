"""智能体开发学习接口：工具清单、工具调用日志、Agent 循环、ReAct、结构化输出、多轮会话。

除 Day 07 的长期记忆与知识库外，其余接口都不落库——学习页是"试验台"，
会话状态只活在进程内存里，重启即清。
"""

from fastapi import APIRouter, Depends, HTTPException
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
