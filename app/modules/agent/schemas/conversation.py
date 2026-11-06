"""会话与消息相关的请求/响应结构。"""

import datetime as dt

from pydantic import BaseModel, Field

from app.core.schemas import ORMModel


class ConversationCreate(BaseModel):
    """新建会话；标题与模型都可以先不指定。"""

    title: str | None = Field(default=None, max_length=100, description="会话标题")
    model_config_id: int | None = Field(
        default=None, description="绑定的 LLM 模型配置 id；不传则用默认模型"
    )


class ConversationUpdate(BaseModel):
    """编辑会话（目前只支持改名与切换模型）。"""

    title: str | None = Field(default=None, min_length=1, max_length=100)
    model_config_id: int | None = None


class MessageOut(ORMModel):
    """单条消息输出。"""

    id: int
    conversation_id: int
    role: str
    content: str
    status: str
    error_message: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int | None = None
    created_at: dt.datetime | None = None


class ConversationOut(ORMModel):
    """会话列表项（不含消息，减小响应体）。"""

    id: int
    title: str
    model_config_id: int | None = None
    created_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None


class ConversationDetailOut(ConversationOut):
    """会话详情：带上完整消息列表，前端打开会话时一次取回。"""

    messages: list[MessageOut] = Field(default_factory=list)
