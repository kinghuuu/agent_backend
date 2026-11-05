"""模型管理视图：/api/v1/models。"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.deps import get_db
from app.modules.model_config import service
from app.modules.model_config.schemas import (
    ContextTestPayload,
    EnabledPayload,
    ModelConfigCreate,
    ModelConfigOut,
    ModelConfigUpdate,
    ModelTestResult,
)

router = APIRouter(tags=["模型管理"])


async def _get_or_404(db: AsyncSession, model_id: int):
    """取出模型配置，不存在直接抛 404（供各路由复用）。"""
    obj = await service.get_model_config(db, model_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="模型配置不存在")
    return obj


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
    model_id: int,
    payload: ModelConfigUpdate,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ModelConfigOut]:
    """编辑模型配置，只更新传入的字段（api_key 留空表示不修改）。"""
    obj = await _get_or_404(db, model_id)
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
    model_id: int,
    payload: EnabledPayload,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ModelConfigOut]:
    """启用或禁用模型；被禁用的模型不参与后续的模型选择。"""
    obj = await _get_or_404(db, model_id)
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
    model_id: int,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[None]:
    """删除模型配置（物理删除，前端有二次确认）。"""
    obj = await _get_or_404(db, model_id)
    await service.delete_model_config(db, obj)
    return APIResponse(message="删除成功")


@router.post(
    "/{model_id}/test",
    response_model=APIResponse[ModelTestResult],
    summary="连通测试",
)
async def test_model(
    model_id: int,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ModelTestResult]:
    """连通测试：真实调用一次模型接口，返回是否成功、耗时与错误原因。"""
    obj = await _get_or_404(db, model_id)
    return APIResponse(data=await service.test_connectivity(obj))


@router.post(
    "/{model_id}/test-context",
    response_model=APIResponse[ModelTestResult],
    summary="上下文长度测试",
)
async def test_model_context(
    model_id: int,
    payload: ContextTestPayload,
    db: AsyncSession = Depends(get_db),
) -> APIResponse[ModelTestResult]:
    """上下文测试：发送约 num_tokens 长度的文本，验证模型的处理能力。"""
    obj = await _get_or_404(db, model_id)
    return APIResponse(data=await service.test_context_length(obj, payload.num_tokens))
