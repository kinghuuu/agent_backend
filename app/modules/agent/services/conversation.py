"""会话管理的业务层：新建、改名、删除、清空消息。"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.agent.crud import conversation as conversation_crud
from app.modules.agent.crud import message as message_crud
from app.modules.agent.models import AgentConversation
from app.modules.agent.schemas import ConversationCreate, ConversationUpdate

DEFAULT_TITLE = "新对话"


async def list_conversations(db: AsyncSession) -> list[AgentConversation]:
    """会话列表，按最近更新倒序（前端侧边栏直接展示）。"""
    return await conversation_crud.list_all(db)


async def get_conversation(
    db: AsyncSession, conversation_id: int
) -> AgentConversation | None:
    """按 id 查询会话，不存在返回 None（由视图层转 404）。"""
    return await conversation_crud.get(db, conversation_id)


async def get_conversation_detail(
    db: AsyncSession, conversation_id: int
) -> AgentConversation | None:
    """查询会话详情（含消息列表）。"""
    return await conversation_crud.get_with_messages(db, conversation_id)


async def create_conversation(
    db: AsyncSession, payload: ConversationCreate
) -> AgentConversation:
    """新建会话；标题留空时用「新对话」占位，等首条用户消息自动命名。"""
    title = (payload.title or "").strip() or DEFAULT_TITLE
    return await conversation_crud.create(
        db, {"title": title[:100], "model_config_id": payload.model_config_id}
    )


async def update_conversation(
    db: AsyncSession, obj: AgentConversation, payload: ConversationUpdate
) -> AgentConversation:
    """编辑会话，只更新请求中显式传入的字段。"""
    data = payload.model_dump(exclude_unset=True)
    if "title" in data:
        data["title"] = (data["title"] or "").strip() or obj.title
    if not data:
        return obj
    return await conversation_crud.update(db, obj, data)


async def delete_conversation(db: AsyncSession, obj: AgentConversation) -> None:
    """删除会话及其全部消息（消息由数据库级联删除）。"""
    await conversation_crud.delete(db, obj)


async def clear_messages(db: AsyncSession, obj: AgentConversation) -> int:
    """清空会话内的消息但保留会话本身，返回删除条数。

    清空后标题也复位，下一次提问会重新自动命名。
    """
    deleted = await message_crud.delete_by_conversation(db, obj.id)
    await conversation_crud.update(db, obj, {"title": DEFAULT_TITLE})
    return deleted
