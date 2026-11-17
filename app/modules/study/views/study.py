"""智能体开发学习接口：工具清单、工具调用日志、Agent 循环、ReAct、结构化输出、多轮会话。

所有接口都没有落库行为——学习页是"试验台"，会话状态只活在进程内存里，重启即清。
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.deps import get_db
from app.modules.study.schemas.study import (
    AgentRunRequest,
    AgentRunResult,
    ExtractRunRequest,
    ExtractRunResult,
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
