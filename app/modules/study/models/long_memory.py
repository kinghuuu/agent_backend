"""长期记忆（Day 07 画像类管道）的 ORM 模型。

与 Day 06 的 SessionState 区别：SessionState 活在进程内存、只管一次会话；
本表落库、按 user_id 组织，跨会话、跨重启都能召回。
"""

from sqlalchemy import Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class StudyUserMemory(Base, TimestampMixin):
    """per-user 的结构化事实（画像类长期记忆）。

    写入原则：写事实不写对话；同一 ``(user_id, mem_key)`` 覆盖而非追加——
    记忆必须跟着用户最新说法走，否则新旧两条并存会让模型自相矛盾。
    """

    __tablename__ = "study_user_memory"
    __table_args__ = (
        UniqueConstraint("user_id", "mem_key", name="uq_study_user_memory_user_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键")
    user_id: Mapped[str] = mapped_column(
        String(100), index=True, comment="用户 ID（业务侧稳定标识）"
    )
    mem_key: Mapped[str] = mapped_column(String(100), comment="事实键，如 dietary")
    value: Mapped[str] = mapped_column(Text, comment="事实值，如 不吃香菜")
    source_turn: Mapped[int | None] = mapped_column(
        Integer, default=None, comment="来源对话轮次（可追溯，不是给模型看的）"
    )
