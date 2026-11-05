"""应用入口。

职责：创建 FastAPI 实例、装配中间件与路由、管理应用生命周期（启动/关闭时建连与释放资源）。

启动方式：
    python main.py                                  # 读取 .env 中的 HOST / PORT
    uvicorn main:app --reload --port 15401
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import settings
from app.core.logging import setup_logging
from app.core.schemas import APIResponse
from app.db.redis import close_redis, init_redis
from app.db.session import engine
from app.modules.model_config.view import router as model_config_router
from app.modules.system.view import router as system_router

setup_logging(logging.DEBUG if settings.DEBUG else logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时建立依赖连接，关闭时释放。"""
    logger.info("%s 启动中...", settings.PROJECT_NAME)
    await init_redis()
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL 连接成功")
    except Exception as exc:  # noqa: BLE001
        logger.warning("PostgreSQL 连接失败（不影响启动）: %s", exc)
    yield
    await close_redis()
    await engine.dispose()
    logger.info("%s 已停止", settings.PROJECT_NAME)


def create_app() -> FastAPI:
    """应用工厂：装配中间件、全局异常处理与 v1 路由。

    拆成工厂函数便于测试时创建独立实例（不同配置可各自 create_app）。
    """
    app = FastAPI(
        title=settings.PROJECT_NAME,
        description="Agent 学习项目 - FastAPI 后端骨架",
        version="0.1.0",
        debug=settings.DEBUG,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 全局异常处理，保持统一响应格式
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("未处理异常: %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content=APIResponse(code=500, message=str(exc), data=None).model_dump(),
        )

    @app.get("/", summary="根路径", tags=["系统"])
    async def root():
        return APIResponse(data={"name": settings.PROJECT_NAME, "docs": "/docs"})

    # 注册各模块的接口视图（新增模块在这里加一行，前缀 = /api/v1/<资源>）
    app.include_router(system_router, prefix=f"{settings.API_V1_STR}/health")
    app.include_router(model_config_router, prefix=f"{settings.API_V1_STR}/models")
    return app


app = create_app()


if __name__ == "__main__":
    # 支持 python main.py 直接启动（开发期自动 reload）
    import uvicorn

    uvicorn.run(
        "main:app",  # reload 模式下必须用字符串形式的导入路径
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
