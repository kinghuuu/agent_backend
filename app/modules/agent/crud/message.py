"""会话消息表的 crud（纯数据库操作，不含业务规则）。"""

from sqlalchemy import delete as sql_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.agent.models import AgentMessage


async def list_by_conversation(
    db: AsyncSession, conversation_id: int
) -> list[AgentMessage]:
    """按 id 升序取出一个会话的全部消息（即对话历史，顺序不能乱）。"""
    stmt = (
        select(AgentMessage)
        .where(AgentMessage.conversation_id == conversation_id)
        .order_by(AgentMessage.id.asc())
    )
    return list((await db.scalars(stmt)).all())


async def create(db: AsyncSession, data: dict) -> AgentMessage:
    """插入一条消息并返回带主键的对象。"""
    obj = AgentMessage(**data)
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def delete_by_conversation(db: AsyncSession, conversation_id: int) -> int:
    """清空一个会话的全部消息，返回删除条数。"""
    stmt = sql_delete(AgentMessage).where(
        AgentMessage.conversation_id == conversation_id
    )
    result = await db.execute(stmt)
    await db.commit()
    return result.rowcount or 0
