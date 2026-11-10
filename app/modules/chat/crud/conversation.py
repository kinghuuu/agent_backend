"""会话表的 crud（纯数据库操作，不含业务规则）。"""

import datetime as dt

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.modules.chat.models import ChatConversation


async def list_all(db: AsyncSession) -> list[ChatConversation]:
    """查询全部会话，按最近更新排在前面（会话列表的展示顺序）。"""
    stmt = select(ChatConversation).order_by(
        ChatConversation.updated_at.desc(), ChatConversation.id.desc()
    )
    return list((await db.scalars(stmt)).all())


async def get(db: AsyncSession, conversation_id: int) -> ChatConversation | None:
    """按主键查询，不存在返回 None。"""
    return await db.get(ChatConversation, conversation_id)


async def get_with_messages(
    db: AsyncSession, conversation_id: int
) -> ChatConversation | None:
    """查询会话并一次性预加载消息（``selectinload``）。

    异步 ORM 下不能"用到关系属性时再懒加载"（会抛 MissingGreenlet），
    所以详情接口必须在查询时显式预加载。
    """
    stmt = (
        select(ChatConversation)
        .where(ChatConversation.id == conversation_id)
        .options(selectinload(ChatConversation.messages))
    )
    return (await db.scalars(stmt)).first()


async def create(
    db: AsyncSession, data: dict
) -> ChatConversation:
    """插入一条会话并返回带主键的对象。"""
    obj = ChatConversation(**data)
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update(
    db: AsyncSession, obj: ChatConversation, data: dict
) -> ChatConversation:
    """按字典更新会话字段（data 只包含需要变更的字段）。"""
    for key, value in data.items():
        setattr(obj, key, value)
    await db.commit()
    await db.refresh(obj)
    return obj


async def touch(db: AsyncSession, obj: ChatConversation) -> None:
    """刷新会话的 updated_at，让"有新消息的会话"排到列表最前面。"""
    obj.updated_at = dt.datetime.now(dt.UTC)
    await db.commit()


async def delete(db: AsyncSession, obj: ChatConversation) -> None:
    """删除会话；消息由数据库 ON DELETE CASCADE 连带删除。"""
    await db.delete(obj)
    await db.commit()
