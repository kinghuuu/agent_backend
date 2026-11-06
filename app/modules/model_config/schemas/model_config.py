"""模型配置（增删改查）相关的请求/响应结构。"""

import datetime as dt
from enum import Enum

from pydantic import BaseModel, Field

from app.core.schemas import ORMModel


class ModelTypeEnum(str, Enum):
    """模型类型。"""

    LLM = "llm"
    EMBEDDING = "embedding"
    RERANK = "rerank"


class ModelConfigCreate(BaseModel):
    """新增模型配置。"""

    name: str = Field(min_length=1, max_length=100, description="展示名称")
    model_name: str = Field(min_length=1, max_length=100, description="实际模型名")
    model_type: ModelTypeEnum = ModelTypeEnum.LLM
    base_url: str = Field(min_length=1, max_length=255, description="服务地址（OpenAI 兼容）")
    api_key: str | None = Field(default=None, max_length=255, description="API Key")
    is_default: bool = Field(default=False, description="是否为该类型默认模型")
    remark: str | None = None


class ModelConfigUpdate(BaseModel):
    """编辑模型配置（只更新传入的字段）。"""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    model_name: str | None = Field(default=None, min_length=1, max_length=100)
    model_type: ModelTypeEnum | None = None
    base_url: str | None = Field(default=None, min_length=1, max_length=255)
    api_key: str | None = None
    is_default: bool | None = None
    remark: str | None = None


class EnabledPayload(BaseModel):
    """启用/禁用。"""

    enabled: bool


class ModelConfigOut(ORMModel):
    """模型配置输出（不回传 api_key）。"""

    id: int
    name: str
    model_name: str
    model_type: str
    base_url: str
    enabled: bool
    is_default: bool
    remark: str | None = None
    created_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None
