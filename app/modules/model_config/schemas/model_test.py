"""模型连通性/上下文测试相关的请求/响应结构。"""

from typing import Any

from pydantic import BaseModel, Field


class ContextTestPayload(BaseModel):
    """上下文测试参数。"""

    num_tokens: int = Field(default=4000, ge=100, le=200000, description="测试用的上下文长度（约）")


class ModelTestResult(BaseModel):
    """连通/上下文测试结果。"""

    success: bool
    latency_ms: int
    message: str
    detail: dict[str, Any] | None = None
