"""健康检查视图：/api/v1/health。"""

from fastapi import APIRouter, Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.schemas import APIResponse
from app.deps import get_db, get_redis
from app.modules.system.schemas import HealthCheck
from app.modules.system.service import check_health

router = APIRouter(tags=["系统"])


@router.get("", response_model=APIResponse[HealthCheck], summary="健康检查")
async def health_check(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> APIResponse[HealthCheck]:
    """探测应用依赖（PostgreSQL / Redis）的连通性。"""
    return APIResponse(data=await check_health(db, redis))
