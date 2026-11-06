"""模型管理模块的接口视图聚合（供 main.py 注册到 /api/v1/models）。"""

from fastapi import APIRouter

from app.modules.model_config.views.model_config import router as model_config_router
from app.modules.model_config.views.model_test import router as model_test_router

router = APIRouter()
router.include_router(model_config_router, prefix="/models")
router.include_router(model_test_router, prefix="/models")
