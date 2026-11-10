"""会话消息表的 ORM 模型。"""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin

if TYPE_CHECKING:  # 仅类型检查时导入，避免模型间循环导入
    from app.modules.chat.models.conversation import ChatConversation


class ChatMessage(Base, TimestampMixin):
    """会话中的一条消息（用户 / 助手）。

    这张表同时充当"对话记忆"：每次请求模型时，按 id 顺序取出本会话的历史消息，
    拼成 OpenAI 协议里的 messages 数组。
    """

    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键")
    conversation_id: Mapped[int] = mapped_column(
        ForeignKey("chat_conversation.id", ondelete="CASCADE"),
        index=True,
        comment="所属会话 id",
    )
    role: Mapped[str] = mapped_column(
        String(20), default="user", comment="角色: user / assistant / system"
    )
    content: Mapped[str] = mapped_column(Text, default="", comment="消息内容")
    status: Mapped[str] = mapped_column(
        String(20),
        default="success",
        comment="生成状态: success / error / stopped（中断）",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, default=None, comment="失败原因（status=error 时有值）"
    )
    prompt_tokens: Mapped[int | None] = mapped_column(
        Integer, default=None, comment="本次请求的输入 token 数（模型返回时才有）"
    )
    completion_tokens: Mapped[int | None] = mapped_column(
        Integer, default=None, comment="本次生成的输出 token 数（模型返回时才有）"
    )
    latency_ms: Mapped[int | None] = mapped_column(
        Integer, default=None, comment="生成耗时（毫秒）"
    )

    conversation: Mapped["ChatConversation"] = relationship(back_populates="messages")
