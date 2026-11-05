"""数据库会话管理（SQLAlchemy 2.x 异步 + asyncpg 驱动）。

对外提供三样东西：``engine``（全局连接池）、``SessionLocal``（会话工厂）、
``get_db``（FastAPI 依赖，每个请求一个会话）。
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,  # 开发期打印 SQL，便于学习
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 依赖：每个请求一个会话，结束自动关闭。"""
    async with SessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
