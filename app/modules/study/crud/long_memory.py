"""长期记忆表的纯数据库读写。

CRUD 层不做「覆盖还是追加」这类业务判断——那是 services 的事，这里只提供原子操作。
"""

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.models import StudyUserMemory


async def list_by_user(db: AsyncSession, user_id: str) -> list[StudyUserMemory]:
    """按 user_id 取全部事实，按 key 排序保证输出稳定。"""
    stmt = (
        select(StudyUserMemory)
        .where(StudyUserMemory.user_id == user_id)
        .order_by(StudyUserMemory.mem_key)
    )
    return list((await db.scalars(stmt)).all())


async def get_one(db: AsyncSession, user_id: str, mem_key: str) -> StudyUserMemory | None:
    """取单条事实（同 key 是否已存在，交由 service 决定覆盖）。"""
    stmt = select(StudyUserMemory).where(
        StudyUserMemory.user_id == user_id, StudyUserMemory.mem_key == mem_key
    )
    return (await db.scalars(stmt)).first()


async def create(
    db: AsyncSession, user_id: str, mem_key: str, value: str, source_turn: int | None = None
) -> StudyUserMemory:
    """插入一条事实并提交。"""
    obj = StudyUserMemory(
        user_id=user_id, mem_key=mem_key, value=value, source_turn=source_turn
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update_value(
    db: AsyncSession, obj: StudyUserMemory, value: str, source_turn: int | None = None
) -> StudyUserMemory:
    """覆盖已有事实的值（同 key 更新而非追加）。"""
    obj.value = value
    if source_turn is not None:
        obj.source_turn = source_turn
    await db.commit()
    await db.refresh(obj)
    return obj


async def delete_one(db: AsyncSession, user_id: str, mem_key: str) -> bool:
    """删除单条事实，返回是否真的删到了。"""
    result = await db.execute(
        delete(StudyUserMemory).where(
            StudyUserMemory.user_id == user_id, StudyUserMemory.mem_key == mem_key
        )
    )
    await db.commit()
    return bool(result.rowcount)


async def delete_user(db: AsyncSession, user_id: str) -> int:
    """清空某个用户的全部记忆，返回删除条数。"""
    result = await db.execute(
        delete(StudyUserMemory).where(StudyUserMemory.user_id == user_id)
    )
    await db.commit()
    return int(result.rowcount or 0)
