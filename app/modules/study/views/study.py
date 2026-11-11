"""智能体开发学习接口：工具清单、Function Calling Agent 循环、手写 ReAct 循环。

三个接口都没有落库行为——学习页是"试验台"，跑完即走，不产生会话数据。
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.deps import get_db
from app.modules.study.schemas.study import (
    AgentRunRequest,
    AgentRunResult,
    ReactRunRequest,
    ReactRunResult,
    ToolOut,
)
from app.modules.study.services import agent_loop, react_loop
from app.modules.study.services.tools import TOOL_SCHEMAS

router = APIRouter(tags=["智能体开发学习"])


@router.get(
    "/tools",
    response_model=APIResponse[list[ToolOut]],
    summary="Day 01 工具清单",
)
async def list_tools() -> APIResponse[list[ToolOut]]:
    """返回当前注册的全部工具与 JSON Schema 说明书（展示"模型眼里看到的工具长什么样"）。"""
    data = [
        ToolOut(
            name=schema["function"]["name"],
            description=schema["function"]["description"],
            parameters=schema["function"]["parameters"],
        )
        for schema in TOOL_SCHEMAS
    ]
    return APIResponse(data=data)


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
