"""模型管理模块的请求/响应结构（按功能拆分文件，这里统一导出）。

视图层从这里导入；命名约定：``XxxCreate`` 新增入参、``XxxUpdate`` 编辑入参
（字段全可选）、``XxxOut`` 出参（继承 ORMModel，不含敏感字段）。
"""

from app.modules.model_config.schemas.model_config import (
    EnabledPayload,
    ModelConfigCreate,
    ModelConfigOut,
    ModelConfigUpdate,
    ModelTypeEnum,
)
from app.modules.model_config.schemas.model_test import (
    ContextTestPayload,
    ModelTestResult,
)

__all__ = [
    "ContextTestPayload",
    "EnabledPayload",
    "ModelConfigCreate",
    "ModelConfigOut",
    "ModelConfigUpdate",
    "ModelTestResult",
    "ModelTypeEnum",
]
