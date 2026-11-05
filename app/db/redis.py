"""Redis 异步客户端管理。

全进程共用一个连接池：懒创建（首次使用时建连）、启动时 ping 校验、关闭时释放。
用途：缓存、会话短期记忆、限流等。
"""

import logging
from collections.abc import AsyncGenerator

from redis.asyncio import Redis

from app.core.config import settings

logger = logging.getLogger(__name__)

_redis: Redis | None = None


def get_redis_client() -> Redis:
    """获取全局 Redis 客户端（懒创建）。"""
    global _redis
    if _redis is None:
        _redis = Redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
    return _redis


async def init_redis() -> None:
    """启动时校验 Redis 连通性。"""
    client = get_redis_client()
    try:
        await client.ping()
        logger.info("Redis 连接成功: %s", settings.REDIS_URL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Redis 连接失败（不影响启动）: %s", exc)


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
        logger.info("Redis 连接已关闭")


async def get_redis() -> AsyncGenerator[Redis, None]:
    """FastAPI 依赖：在路由中注入 Redis 客户端。"""
    yield get_redis_client()
