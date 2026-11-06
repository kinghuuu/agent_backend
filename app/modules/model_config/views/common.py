"""模型管理视图层的公共依赖。"""

from fastapi import Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps import get_db
from app.modules.model_config.models import ModelConfig
from app.modules.model_config.services import model_config as service


async def get_model_or_404(
    model_id: int,
    db: AsyncSession = Depends(get_db),
) -> ModelConfig:
    """路径参数 model_id -> 模型配置；不存在自动返回 404。

    作为 FastAPI 依赖使用：``obj: ModelConfig = Depends(get_model_or_404)``，
    这样各路由不用重复写"查询 + 判空 + 抛 404"。
    """
    obj = await service.get_model_config(db, model_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="模型配置不存在")
    return obj
