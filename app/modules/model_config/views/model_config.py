"""模型管理模块的 CRUD 接口视图：/api/v1/models。"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.deps import get_db
from app.modules.model_config.models import ModelConfig
from app.modules.model_config.schemas import (
    EnabledPayload,
    ModelConfigCreate,
    ModelConfigOut,
    ModelConfigUpdate,
)
from app.modules.model_config.services import model_config as service
from app.modules.model_config.views.common import get_model_or_404

router = APIRouter(tags=["模型管理"])


@router.get(
    "",
    response_model=APIResponse[list[ModelConfigOut]],
    summary="模型配置列表",
)
async def list_models(
    model_type: str | None = Query(default=None, description="按类型过滤: llm/embedding/rerank"),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[list[ModelConfigOut]]:
    """查询模型配置列表，前端按 Tab 传入 model_type 过滤。"""
    items = await service.list_model_configs(db, model_type)
    return APIResponse(data=items)


@router.post(
    "",
    response_model=APIResponse[ModelConfigOut],
    status_code=status.HTTP_201_CREATED,
    summary="新增模型配置",
)
async def create_model(
    payload: ModelConfigCreate,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ModelConfigOut]:
    """新增模型配置；展示名称重复时返回 400。"""
    try:
        obj = await service.create_model_config(db, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(message="创建成功", data=obj)


@router.put(
    "/{model_id}",
    response_model=APIResponse[ModelConfigOut],
    summary="编辑模型配置",
)
async def update_model(
    payload: ModelConfigUpdate,
    obj: ModelConfig = Depends(get_model_or_404),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ModelConfigOut]:
    """编辑模型配置，只更新传入的字段（api_key 留空表示不修改）。"""
    try:
        obj = await service.update_model_config(db, obj, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return APIResponse(message="更新成功", data=obj)


@router.patch(
    "/{model_id}/enabled",
    response_model=APIResponse[ModelConfigOut],
    summary="启用/禁用模型",
)
async def toggle_model(
    payload: EnabledPayload,
    obj: ModelConfig = Depends(get_model_or_404),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ModelConfigOut]:
    """启用或禁用模型；被禁用的模型不参与后续的模型选择。"""
    obj = await service.update_model_config(
        db, obj, ModelConfigUpdate(enabled=payload.enabled)
    )
    return APIResponse(message="已启用" if payload.enabled else "已禁用", data=obj)


@router.delete(
    "/{model_id}",
    response_model=APIResponse[None],
    summary="删除模型配置",
)
async def delete_model(
    obj: ModelConfig = Depends(get_model_or_404),
    db: AsyncSession = Depends(get_db),
) -> APIResponse[None]:
    """删除模型配置（物理删除，前端有二次确认）。"""
    await service.delete_model_config(db, obj)
    return APIResponse(message="删除成功")
