"""业务模块包：一个子目录 = 一个功能模块（高内聚，新增模块互不影响）。

每个模块的目录约定（详见 AGENTS.md）：
    models.py      ORM 模型（表结构）
    schemas.py     Pydantic 请求/响应结构
    repository.py  仓储层：纯数据库读写（原 crud）
    service.py     业务层：业务规则、外部调用
    view.py        接口视图：路由定义，在 main.py 的 create_app() 中注册

没有表结构的模块（如 system）可以只有 schemas.py / service.py / view.py。

注意：这里要导入各模块的 models，保证 Alembic autogenerate 能发现全部表。
"""

from app.modules.model_config import models  # noqa: F401
