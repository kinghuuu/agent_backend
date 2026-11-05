"""公共响应结构（跨模块共享的基础设施）。

``APIResponse``：全站统一的响应外壳；``ORMModel``：可从 ORM 对象直接构造的基类。
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """统一响应结构：{ code, message, data }"""

    code: int = 0
    message: str = "ok"
    data: T | None = None


class ORMModel(BaseModel):
    """需要从 ORM 对象直接转出的 Schema 继承此类。"""

    model_config = ConfigDict(from_attributes=True)
