"""模型配置表的 ORM 模型。"""

from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class ModelConfig(Base, TimestampMixin):
    """大模型配置（LLM / Embedding / Rerank）。

    每个功能模块一张表，字段就近写在模型类上。
    """

    __tablename__ = "model_config"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键")
    name: Mapped[str] = mapped_column(String(100), unique=True, comment="展示名称")
    model_name: Mapped[str] = mapped_column(String(100), comment="实际模型名，如 deepseek-chat")
    model_type: Mapped[str] = mapped_column(
        String(20), default="llm", index=True, comment="模型类型: llm / embedding / rerank"
    )
    base_url: Mapped[str] = mapped_column(String(255), comment="服务地址（OpenAI 兼容 base_url）")
    # 注意：学习项目明文存储；生产环境应加密（如 Fernet）后再落库
    api_key: Mapped[str | None] = mapped_column(String(255), default=None, comment="API Key")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, comment="是否启用")
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, comment="是否为该类型的默认模型")
    remark: Mapped[str | None] = mapped_column(Text, default=None, comment="备注")
