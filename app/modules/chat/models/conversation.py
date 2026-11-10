"""会话表的 ORM 模型。"""

from typing import TYPE_CHECKING

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:  # 仅类型检查时导入，避免模型间循环导入
    from app.modules.chat.models.message import ChatMessage


class ChatConversation(Base, TimestampMixin):
    """一次多轮对话（一个会话 = 一条记录，消息存在 chat_message 表）。"""

    __tablename__ = "chat_conversation"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键")
    title: Mapped[str] = mapped_column(
        String(100), default="新对话", comment="会话标题（默认取首条用户消息前 20 字）"
    )
    # 只存模型配置 id、不建外键：避免与 model_config 表产生表级耦合，
    # 模型被删除时这里变成"无效 id"，对话时自动回落到默认 LLM。
    model_config_id: Mapped[int | None] = mapped_column(
        Integer, default=None, index=True, comment="会话使用的 LLM 模型配置 id"
    )

    # passive_deletes=True：删除会话时交给数据库的 ON DELETE CASCADE 处理，
    # 避免 ORM 先把成百上千条消息读进内存再逐条删除。
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ChatMessage.id",
    )
