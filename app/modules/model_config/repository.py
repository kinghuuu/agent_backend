"""模型管理模块的仓储层（原 crud，纯数据库操作，不含业务规则）。

约定：第一个参数是 ``AsyncSession``；只做增删改查，不抛 HTTP 异常、不做校验。
业务校验放在同模块的 ``service.py``。
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.model_config.models import ModelConfig


async def get(db: AsyncSession, model_id: int) -> ModelConfig | None:
    """按主键查询，不存在返回 None。"""
    return await db.get(ModelConfig, model_id)


async def get_by_name(db: AsyncSession, name: str) -> ModelConfig | None:
    """按展示名称查询（名称全局唯一，用于重名校验）。"""
    stmt = select(ModelConfig).where(ModelConfig.name == name)
    return (await db.scalars(stmt)).first()


async def list_all(db: AsyncSession, model_type: str | None = None) -> list[ModelConfig]:
    """查询模型列表；传入 model_type 时按类型过滤，按 id 倒序（新建的在前）。"""
    stmt = select(ModelConfig).order_by(ModelConfig.id.desc())
    if model_type:
        stmt = stmt.where(ModelConfig.model_type == model_type)
    return list((await db.scalars(stmt)).all())


async def create(db: AsyncSession, data: dict) -> ModelConfig:
    """插入一条记录并返回带主键的对象。"""
    obj = ModelConfig(**data)
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update(db: AsyncSession, obj: ModelConfig, data: dict) -> ModelConfig:
    """按字典更新已有对象（data 只包含需要变更的字段）。"""
    for key, value in data.items():
        setattr(obj, key, value)
    await db.commit()
    await db.refresh(obj)
    return obj


async def delete(db: AsyncSession, obj: ModelConfig) -> None:
    """物理删除一条记录。"""
    await db.delete(obj)
    await db.commit()


async def clear_default(db: AsyncSession, model_type: str, exclude_id: int) -> None:
    """把同类型下其它记录的 is_default 置为 False。

    用于保证"每种模型类型最多一个默认模型"这一业务约束。
    """
    stmt = select(ModelConfig).where(
        ModelConfig.model_type == model_type,
        ModelConfig.is_default.is_(True),
        ModelConfig.id != exclude_id,
    )
    for obj in (await db.scalars(stmt)).all():
        obj.is_default = False
    await db.commit()
