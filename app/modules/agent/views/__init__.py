"""智能体模块的接口视图聚合（供 main.py 注册到 /api/v1/agent）。"""

from fastapi import APIRouter

from app.modules.agent.views.chat import router as chat_router
from app.modules.agent.views.conversation import router as conversation_router

router = APIRouter()
router.include_router(conversation_router, prefix="/agent")
router.include_router(chat_router, prefix="/agent")
