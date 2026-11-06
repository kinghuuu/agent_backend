"""系统模块的接口视图聚合（供 main.py 注册）。"""

from fastapi import APIRouter

from app.modules.system.views.health import router as health_router

router = APIRouter()
router.include_router(health_router, prefix="/health")
