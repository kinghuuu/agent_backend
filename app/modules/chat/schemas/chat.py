"""对话（发送消息）相关的请求/响应结构。"""

from pydantic import BaseModel, Field

from app.modules.chat.schemas.conversation import MessageOut


class AvailableModelOut(BaseModel):
    """会话可选的模型（只暴露前端需要的字段，不含 base_url / api_key）。"""

    id: int
    name: str
    model_name: str
    is_default: bool


class ChatRequest(BaseModel):
    """发送一条用户消息。

    ``model_config_id`` 优先级：本次请求 > 会话已绑定 > 默认 LLM 模型，
    这样前端切换模型既可以是"本次生效"，也可以是"整个会话生效"。
    """

    content: str = Field(min_length=1, max_length=20000, description="用户消息内容")
    model_config_id: int | None = Field(default=None, description="本次使用的模型配置 id")
    system_prompt: str | None = Field(
        default=None, max_length=4000, description="系统提示词；不传则用内置默认值"
    )


class ChatResult(BaseModel):
    """非流式对话结果：一次返回用户消息与助手消息。"""

    model_name: str
    model_config_id: int
    user_message: MessageOut
    assistant_message: MessageOut
