"""系统模块的请求/响应结构（按功能拆分文件，这里统一导出）。"""

from app.modules.system.schemas.health import HealthCheck

__all__ = ["HealthCheck"]
