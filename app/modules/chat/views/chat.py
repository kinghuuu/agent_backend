"""会话接口：可选模型、发送消息（流式 / 非流式）。"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.deps import get_db
from app.modules.chat.models import ChatConversation
from app.modules.chat.schemas import AvailableModelOut, ChatRequest, ChatResult
from app.modules.chat.services import chat as service
from app.modules.chat.views.common import get_conversation_or_404

router = APIRouter(tags=["会话"])

# 流式响应的公共响应头：
# - text/event-stream：SSE 标准 MIME
# - X-Accel-Buffering: no：关掉 Nginx 等反向代理的缓冲，否则前端会"一次性"收到全部内容
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.get(
    "/models",
    response_model=APIResponse[list[AvailableModelOut]],
    summary="会话可用模型",
)
async def list_chat_models(
    db: AsyncSession = Depends(get_db),
) -> APIResponse[list[AvailableModelOut]]:
    """返回启用中的 LLM 模型（默认模型在前），供前端模型下拉框使用。"""
    items = await service.list_available_models(db)
    data = [
        AvailableModelOut(
            id=item.id, name=item.name, model_name=item.model_name, is_default=item.is_default
        )
        for item in items
    ]
    return APIResponse(data=data)


@router.post(
    "/conversations/{conversation_id}/chat",
    response_model=APIResponse[ChatResult],
    summary="发送消息（非流式）",
)
async def chat(
    payload: ChatRequest,
    conversation: ChatConversation = Depends(get_conversation_or_404),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ChatResult]:
    """发送一条用户消息并等待完整回复（调试 / 不支持流式的客户端用）。"""
    try:
        cfg = await service.resolve_model(db, conversation, payload.model_config_id)
        result = await service.chat(db, conversation, payload, cfg)
    except ValueError as exc:  # 业务校验失败（如无可用模型）
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except service.ChatError as exc:  # 上游模型服务失败
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return APIResponse(data=result)


@router.post(
    "/conversations/{conversation_id}/chat/stream",
    summary="发送消息（流式 SSE）",
    response_class=StreamingResponse,
)
async def chat_stream(
    payload: ChatRequest,
    conversation: ChatConversation = Depends(get_conversation_or_404),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """流式对话：返回 ``text/event-stream``，每个事件是一个 ``data: {json}`` 块。

    模型解析放在返回响应之前，这样"没有可用模型"仍能用正常的 400 状态码告知前端；
    之后的错误（网络 / 上游报错）只能通过 SSE 的 ``error`` 事件传达。
    """
    # 先解析一次模型：没有可用模型时立刻返回 400，而不是在 SSE 里报错
    try:
        await service.resolve_model(db, conversation, payload.model_config_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return StreamingResponse(
        service.stream_chat(conversation.id, payload),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
