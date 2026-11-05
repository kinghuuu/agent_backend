"""ORM 基础设施：声明式基类与通用 Mixin。

所有业务模型继承 ``Base``；需要审计时间的表再叠加 ``TimestampMixin``。
"""

import datetime as dt

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""

    pass


class TimestampMixin:
    """通用时间字段，业务表按需继承。"""

    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), comment="创建时间"
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        comment="更新时间",
    )
