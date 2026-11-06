"""智能体会话管理接口：/api/v1/agent/conversations。"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.deps import get_db
from app.modules.agent.models import AgentConversation
from app.modules.agent.schemas import (
    ConversationCreate,
    ConversationDetailOut,
    ConversationOut,
    ConversationUpdate,
)
from app.modules.agent.services import conversation as service
from app.modules.agent.views.common import get_conversation_or_404

router = APIRouter(tags=["智能体"])


@router.get(
    "/conversations",
    response_model=APIResponse[list[ConversationOut]],
    summary="会话列表",
)
async def list_conversations(
    db: AsyncSession = Depends(get_db),
) -> APIResponse[list[ConversationOut]]:
    """按最近更新时间倒序返回全部会话（前端侧边栏用）。"""
    items = await service.list_conversations(db)
    return APIResponse(data=items)


@router.post(
    "/conversations",
    response_model=APIResponse[ConversationOut],
    status_code=status.HTTP_201_CREATED,
    summary="新建会话",
)
async def create_conversation(
    payload: ConversationCreate,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ConversationOut]:
    """新建会话；标题留空时后端置为「新对话」，首条消息后自动命名。"""
    obj = await service.create_conversation(db, payload)
    return APIResponse(message="创建成功", data=obj)


@router.get(
    "/conversations/{conversation_id}",
    response_model=APIResponse[ConversationDetailOut],
    summary="会话详情（含消息）",
)
async def get_conversation(
    obj: AgentConversation = Depends(get_conversation_or_404),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ConversationDetailOut]:
    """返回会话及其全部消息，前端切换会话时一次拿全。"""
    detail = await service.get_conversation_detail(db, obj.id)
    return APIResponse(data=detail)


@router.patch(
    "/conversations/{conversation_id}",
    response_model=APIResponse[ConversationOut],
    summary="重命名会话 / 切换模型",
)
async def update_conversation(
    payload: ConversationUpdate,
    obj: AgentConversation = Depends(get_conversation_or_404),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ConversationOut]:
    """只更新传入的字段（标题、绑定的模型）。"""
    obj = await service.update_conversation(db, obj, payload)
    return APIResponse(message="更新成功", data=obj)


@router.delete(
    "/conversations/{conversation_id}",
    response_model=APIResponse[None],
    summary="删除会话",
)
async def delete_conversation(
    obj: AgentConversation = Depends(get_conversation_or_404),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[None]:
    """删除会话及其消息（前端有二次确认）。"""
    await service.delete_conversation(db, obj)
    return APIResponse(message="删除成功")


@router.delete(
    "/conversations/{conversation_id}/messages",
    response_model=APIResponse[dict],
    summary="清空会话消息",
)
async def clear_messages(
    obj: AgentConversation = Depends(get_conversation_or_404),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[dict]:
    """清空消息但保留会话（相当于"重新开始一段对话"）。"""
    deleted = await service.clear_messages(db, obj)
    return APIResponse(message="已清空", data={"deleted": deleted})
