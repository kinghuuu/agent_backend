# agent_backend

FastAPI + PostgreSQL + Redis + Docker 的基础骨架，用于从零学习 Agent 开发。

## 目录结构

```
agent_backend/
├─ main.py                 # FastAPI 入口：生命周期、CORS、全局异常处理
├─ app/
│  ├─ core/                # 基础设施：配置(config.py)、日志(logging.py)、统一响应(schemas.py)
│  ├─ db/                  # 数据库基础设施：ORM基类、engine/session、Redis 客户端
│  ├─ deps.py              # 接口依赖统一出口（get_db / get_redis）
│  ├─ modules/             # ★ 业务模块：一个子目录 = 一个功能模块
│  │  ├─ model_config/     #    模型管理：models/ schemas/ crud/ services/ views/（分层子包，内部按功能拆 .py）
│  │  └─ system/           #    系统模块（健康检查）：schemas/ services/ views/
│  └─ utils/               # 工具函数
├─ alembic/                # 数据库迁移
├─ docker-compose.yml      # postgres + redis + backend
├─ Dockerfile
└─ requirements.txt
```

## 启动

```bash
# 1. 启动依赖（PostgreSQL 15400 / Redis 6379）
docker compose up -d postgres redis

# 2. 本地运行（推荐虚拟环境）
python -m venv .venv
.\.venv\Scripts\activate      # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 15401

# 也可以直接跑入口文件（端口/主机取自 .env 的 PORT / HOST）
python main.py

# 或者整体跑在容器里
docker compose up -d
```

- 接口文档：<http://127.0.0.1:15401/docs>
- 健康检查：<http://127.0.0.1:15401/api/v1/health>，返回 `{code, message, data}`

## 数据库迁移（★ 高频：改了 model 就执行这两步）

```bash
alembic revision --autogenerate -m "描述"   # 1. 依据 models 自动生成迁移脚本
alembic upgrade head                        # 2. 执行迁移，新表/新列立即生效（无需重启）
```

> 注意：新模型必须先在 `app/modules/__init__.py` 中导入，否则生成的脚本是空的；
> `alembic_version` 表是 alembic 自动维护的版本进度记录，不要手动删改。
> 其他：`alembic check`（校验一致性）、`alembic current` / `history`（看版本）、`downgrade -1`（回滚一步）

## 约定

- 所有响应统一为 `APIResponse`：
  ```json
  { "code": 0, "message": "ok", "data": {} }
  ```
- 视图（`views/`）只做参数校验与转发；业务写在模块的 `services/`，数据库读写写在模块的 `crud/`。
- 新增模块五步走（参考 `app/modules/model_config`）：
  1. `app/modules/<模块>/models.py` 建表模型，并在 `app/modules/__init__.py` 中导入
  2. `alembic revision --autogenerate -m "..."` + `alembic upgrade head`
  3. `app/modules/<模块>/schemas.py` 定义请求/响应结构
  4. `app/modules/<模块>/crud/<功能>.py` 写数据库操作 → `services/<功能>.py` 写业务规则
  5. `app/modules/<模块>/views/<功能>.py` 写接口视图，在模块 `views/__init__.py` 聚合并声明前缀，再在 `main.py` 挂载一行
