"""系统模块的响应结构。"""

from app.core.schemas import ORMModel


class HealthCheck(ORMModel):
    """健康检查结果。"""

    status: str
    app: str
    database: str
    redis: str
