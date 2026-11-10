"""会话模块的接口视图聚合（供 main.py 注册到 /api/v1/chat）。"""

from fastapi import APIRouter

from app.modules.chat.views.chat import router as chat_router
from app.modules.chat.views.conversation import router as conversation_router

router = APIRouter()
router.include_router(conversation_router, prefix="/chat")
router.include_router(chat_router, prefix="/chat")
