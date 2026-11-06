"""系统模块的业务层：健康检查探测。"""

import logging

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.modules.system.schemas import HealthCheck

logger = logging.getLogger(__name__)


async def check_health(db: AsyncSession, redis: Redis) -> HealthCheck:
    """探测 PostgreSQL 与 Redis 的连通性。

    任一依赖不可用不影响接口本身返回 200，而是把整体状态标记为 ``degraded``，
    方便前端/监控区分"服务活着但依赖异常"和"服务挂了"。
    """
    database = "up"
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("PostgreSQL 不可用: %s", exc)
        database = "down"

    cache = "up"
    try:
        await redis.ping()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 不可用: %s", exc)
        cache = "down"

    return HealthCheck(
        status="ok" if database == "up" and cache == "up" else "degraded",
        app=settings.PROJECT_NAME,
        database=database,
        redis=cache,
    )
