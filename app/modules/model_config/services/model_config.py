"""模型配置的业务层：增删改查与业务规则（重名校验、默认模型互斥）。"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.model_config.crud import model_config as crud
from app.modules.model_config.models import ModelConfig
from app.modules.model_config.schemas import ModelConfigCreate, ModelConfigUpdate


# ---------- CRUD + 业务规则 ----------

async def list_model_configs(
    db: AsyncSession, model_type: str | None = None
) -> list[ModelConfig]:
    """查询模型配置列表，可按类型（llm/embedding/rerank）过滤。"""
    return await crud.list_all(db, model_type)


async def get_model_config(db: AsyncSession, model_id: int) -> ModelConfig | None:
    """按 id 查询单条配置，不存在返回 None（由视图层转 404）。"""
    return await crud.get(db, model_id)


async def create_model_config(db: AsyncSession, payload: ModelConfigCreate) -> ModelConfig:
    """新增模型配置。

    业务规则：
    - 展示名称全局唯一，重名抛 ``ValueError``（视图层转 400）
    - 若设为默认，则取消同类型下其它记录的默认标记
    """
    if await crud.get_by_name(db, payload.name):
        raise ValueError(f"展示名称「{payload.name}」已存在")
    obj = await crud.create(db, payload.model_dump())
    if obj.is_default:
        await crud.clear_default(db, obj.model_type, exclude_id=obj.id)
    return obj


async def update_model_config(
    db: AsyncSession, obj: ModelConfig, payload: ModelConfigUpdate
) -> ModelConfig:
    """编辑模型配置，只更新请求中显式传入的字段（``exclude_unset``）。"""
    data = payload.model_dump(exclude_unset=True)
    if (name := data.get("name")) and name != obj.name:
        if await crud.get_by_name(db, name):
            raise ValueError(f"展示名称「{name}」已存在")
    obj = await crud.update(db, obj, data)
    if obj.is_default:
        await crud.clear_default(db, obj.model_type, exclude_id=obj.id)
    return obj


async def delete_model_config(db: AsyncSession, obj: ModelConfig) -> None:
    """删除模型配置（物理删除）。"""
    await crud.delete(db, obj)


