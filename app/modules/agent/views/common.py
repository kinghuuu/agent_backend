"""智能体视图层的公共依赖。"""

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_db
from app.modules.agent.models import AgentConversation
from app.modules.agent.services import conversation as service


async def get_conversation_or_404(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),
) -> AgentConversation:
    """路径参数 conversation_id -> 会话；不存在自动返回 404。"""
    obj = await service.get_conversation(db, conversation_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    return obj
