"""业务模块包：一个子目录 = 一个功能模块（高内聚，新增模块互不影响）。

每个模块内部的分层约定（详见 AGENTS.md，各分层是文件夹，内部按功能拆多个 .py）：
    models/       ORM 模型（表结构）
    schemas/      Pydantic 请求/响应结构
    crud/         数据库读写（纯增删改查，不含业务规则）
    services/     业务层：业务规则、外部调用
    views/        接口视图：路由定义，在 main.py 的 create_app() 中注册

没有表结构的模块（如 system）可以只有 schemas/ services/ views/。

注意：这里要导入各模块的 models，保证 Alembic autogenerate 能发现全部表。
"""

from app.modules.chat import models  # noqa: F401
from app.modules.model_config import models as model_config_models  # noqa: F401
