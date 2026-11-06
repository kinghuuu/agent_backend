"""模型管理模块的测试类接口视图：连通测试 / 上下文测试。"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.deps import get_db
from app.modules.model_config.models import ModelConfig
from app.modules.model_config.schemas import ContextTestPayload, ModelTestResult
from app.modules.model_config.services import model_test as service
from app.modules.model_config.views.common import get_model_or_404

router = APIRouter(tags=["模型管理"])


@router.post(
    "/{model_id}/test",
    response_model=APIResponse[ModelTestResult],
    summary="连通测试",
)
async def test_model(
    obj: ModelConfig = Depends(get_model_or_404),
) -> APIResponse[ModelTestResult]:
    """连通测试：真实调用一次模型接口，返回是否成功、耗时与错误原因。"""
    return APIResponse(data=await service.test_connectivity(obj))


@router.post(
    "/{model_id}/test-context",
    response_model=APIResponse[ModelTestResult],
    summary="上下文长度测试",
)
async def test_model_context(
    payload: ContextTestPayload,
    obj: ModelConfig = Depends(get_model_or_404),
) -> APIResponse[ModelTestResult]:
    """上下文测试：发送约 num_tokens 长度的文本，验证模型的处理能力。"""
    return APIResponse(data=await service.test_context_length(obj, payload.num_tokens))
