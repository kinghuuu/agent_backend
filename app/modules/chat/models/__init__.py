"""会话模块的 ORM 模型（按功能拆分文件，这里统一导出）。

Alembic 通过 ``app/modules/__init__.py`` 间接导入本包，保证表都能被发现。
"""

from app.modules.chat.models.conversation import ChatConversation
from app.modules.chat.models.message import ChatMessage

__all__ = ["ChatConversation", "ChatMessage"]
