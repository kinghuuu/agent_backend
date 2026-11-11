"""智能体开发学习模块的接口视图聚合（供 main.py 注册到 /api/v1/study）。"""

from fastapi import APIRouter

from app.modules.study.views.study import router as study_router

router = APIRouter()
router.include_router(study_router, prefix="/study")
