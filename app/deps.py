"""接口视图层依赖的统一出口。

view 只从本模块 import 依赖（而不是直接 import ``app.db.*``），
将来替换实现（如加事务中间件、换缓存）只需改这一处。
"""

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.redis import get_redis as _get_redis
from app.db.session import get_db as _get_db

# 对外暴露统一依赖名
get_db = _get_db
get_redis = _get_redis

__all__ = ["get_db", "get_redis", "AsyncSession", "Redis"]
