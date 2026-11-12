# AGENTS.md — agent_backend 智能体协作指南

> 本文件是**人与 AI 智能体共同遵守**的项目说明书。任何智能体（Claude / CodeBuddy / Cursor…）在本仓库写代码前，必须先读完本文件并严格遵循。改动项目结构或规范时，同步更新本文件。

## 1. 项目简介

从零学习 Agent 开发的后端工程。当前定位：**打好分层骨架，按模块逐步扩展**（模型管理 → 大模型会话 → Tool Calling → 记忆 → RAG → Agent 编排）。

- 技术栈：Python 3.11 + FastAPI + SQLAlchemy 2.x (asyncpg) + Alembic + Redis (redis-py asyncio) + Docker Compose
- 端口：后端 **15401**，PostgreSQL **15400**（root/root，库名 agent_backend），Redis **6379**
- 接口文档：<http://127.0.0.1:15401/docs>

### 当前模块清单

| 模块            | 接口前缀         | 说明                                                     |
| --------------- | ---------------- | -------------------------------------------------------- |
| 系统 / 健康检查 | `/api/v1/health` | 应用、PostgreSQL、Redis 连通状态                         |
| 模型管理        | `/api/v1/models` | LLM / Embedding / Rerank 配置 CRUD、连通测试、上下文测试 |
| 会话            | `/api/v1/chat`   | 会话管理（新建 / 重命名 / 删除 / 清空）、多轮对话（流式 SSE + 非流式）、可用模型列表 |
| 智能体开发学习  | `/api/v1/study`  | Day 01 实验台：工具清单（含 JSON Schema）、Function Calling Agent 循环、手写 ReAct 文本协议循环（均返回结构化执行轨迹，无落库） |

## 2. 启动与常用命令

### ★ 高频操作：改了 models 后同步数据库（刷库）

在 `agent_backend/` 目录下执行两步，新表/新列立即生效，无需重启服务：

```bash
alembic revision --autogenerate -m "描述变更"   # 第 1 步：依据 models 自动生成迁移脚本（存入 alembic/versions/）
alembic upgrade head                            # 第 2 步：把迁移脚本执行到数据库
```

**三个注意点**：

1. 新模型必须先在 `app/modules/__init__.py` 中导入，否则 autogenerate 检测不到，生成的脚本是空的
2. 第 1 步之后先打开 `versions/` 里新生成的脚本扫一眼，确认只有预期改动，再执行第 2 步
3. `alembic_version` 表是 alembic 自动维护的版本进度记录，**不要手动删改**

辅助命令：`alembic check`（校验模型与库是否一致）、`alembic current` / `alembic history`（查看版本）、`alembic downgrade -1`（回滚上一次，慎用）

### 启动相关

```bash
docker compose up -d                # 全容器启动（postgres + redis + backend）
docker compose up -d postgres redis # 只起依赖，本地跑后端
python main.py                      # 本地启动（读 .env 的 HOST/PORT，DEBUG=true 时热重载）
uvicorn main:app --reload --port 15401
docker compose logs backend -f      # 查看后端日志
```

### 不要生成 __pycache__（用户明确要求）

仓库里不允许出现 `__pycache__` / `.pyc`：
- 用户环境已设 `PYTHONDONTWRITEBYTECODE=1`（`setx`，新进程生效）
- `.venv/Lib/site-packages/sitecustomize.py` 里设了 `sys.dont_write_bytecode = True`，用这个 venv 跑的任何进程都不会写缓存
- 用其它 Python 跑脚本时自己带上 `PYTHONDONTWRITEBYTECODE=1`；收尾时若发现缓存目录，直接删掉（`.venv` 内的不用管）


## 3. 目录结构与分层约定

```
agent_backend/
├─ main.py                  # 应用入口：app 工厂、生命周期、全局异常、CORS
├─ app/
│  ├─ core/                 # 基础设施：config.py 配置、logging.py 日志、schemas.py 统一响应
│  ├─ db/                   # base.py ORM基类/Mixin；session.py engine+会话；redis.py Redis客户端
│  ├─ deps.py               # 依赖统一出口（get_db / get_redis），view 只从这里 import
│  ├─ modules/              # ★ 业务模块：一个子目录 = 一个功能模块
│  │  ├─ model_config/      #    models/ schemas/ crud/ services/ views/（每个分层一个文件夹，内部按功能拆 .py）
│  │  └─ system/            #    无表结构的模块可以只有 schemas/ services/ views/
│  └─ utils/                # 与业务无关的纯工具函数
├─ alembic/                 # 迁移（env.py 为异步模板，自动读取 settings.DATABASE_URL）
└─ docker-compose.yml / Dockerfile
```

**模块内部分层（以 `modules/model_config` 为例）**：

| 文件                       | 职责                                                |
| -------------------------- | --------------------------------------------------- |
| `models.py`                | ORM 表结构                                          |
| `schemas.py`               | Pydantic 请求/响应结构                              |
| 分层目录 | 职责 |
| --- | --- |
| `models/` | ORM 表结构，一个功能一个文件（如 `model_config.py`），`__init__.py` 统一导出 |
| `schemas/` | Pydantic 请求/响应结构，一个功能一个文件，`__init__.py` 统一导出 |
| `crud/` | 纯数据库读写，无业务规则，首参必须是 `AsyncSession`；service 调用时带子模块名（`crud.model_config.get(...)`） |
| `services/` | 业务规则、外部调用，可组合本模块 crud |
| `views/` | 接口视图：路由定义，只做参数校验 + 调 service + 包装 `APIResponse`；`__init__.py` 聚合并声明本模块资源前缀 |

路由统一在 `main.py` 的 `create_app()` 中挂载（`prefix = settings.API_V1_STR`），每个模块一行；资源前缀由模块自己的 `views/__init__.py` 声明。

**依赖方向（严禁反向）**：`views → services → crud → models`；跨模块只允许依赖对方 `services/` 公开的能力，禁止直接摸别人的 `crud/` / `models/` / `views/`；`db`/`core`/`deps` 为公共设施。

## 4. 编码规范（Python）

1. **注释与文档字符串用中文**，标识符（变量/函数/类名）用英文。这是学习项目，注释写给"以后的人和智能体"看。
2. **每个 `.py` 文件必须有模块级 docstring**，一句话说明文件职责。
3. **所有公开函数/方法必须有 docstring**：一句话说明用途；参数/返回值不直观时补充说明；非显而易见的实现要写"为什么"，而不是复述"做了什么"。
4. 类型注解：函数签名必须完整注解（`db: AsyncSession`、`-> ModelConfig` 等）。
5. 视图层（`views/`）规范：
   - 统一返回 `APIResponse`（`{code, message, data}`），`code=0` 表示成功；错误用 HTTP 状态码表达
   - 业务校验失败由 service 抛 `ValueError`，view 捕获后转 `HTTPException(400)`
   - 资源不存在统一 `HTTPException(404)`
   - 敏感字段（如 `api_key`）不得出现在任何 `*Out` 响应模型中
6. 数据库：
   - 业务表继承 `Base, TimestampMixin`（自动获得 created_at / updated_at）
   - 改表必须走 Alembic 迁移，**禁止手写 DDL**；新模型必须导入 `app/models/__init__.py`，否则 autogenerate 检测不到
7. 外部 HTTP 调用统一用 `httpx.AsyncClient`，必须设置 timeout 并分类处理异常。
8. 配置只从 `app.core.config.settings` 读取，禁止硬编码地址/密钥；新增配置项要同步改 `.env`、`.env.example`、`Settings` 三处。

## 5. 新增功能模块的标准流程

以 `app/modules/model_config` 模块为范本，六步：

1. `app/modules/<模块>/models.py` 建模型 → 导入到 `app/modules/__init__.py`
2. `alembic revision --autogenerate -m "add xxx"` → 检查生成的迁移脚本 → `alembic upgrade head`
3. `app/modules/<模块>/schemas.py` 定义 Create / Update / Out
4. `app/modules/<模块>/crud/<功能>.py` 写数据库操作
5. `app/modules/<模块>/services/<功能>.py` 写业务规则（含对外部服务的调用）
6. `app/modules/<模块>/views/<功能>.py` 写接口视图 → 在模块的 `views/__init__.py` 聚合并声明资源前缀 → 在 `main.py` 挂载一行

完成前自查清单：

- [ ] 迁移已在本地库执行成功，`/docs` 中的接口可跑通
- [ ] 所有响应都用 `APIResponse` 包装
- [ ] 每个文件有模块 docstring，公开函数有中文 docstring + 类型注解
- [ ] 已更新本文件「当前模块清单」与「变更记录」
- [ ] 若涉及前端联动，已同步前端仓库的「变更记录」

## 6. 变更追踪规范（强制）

**任何新增 / 修改 / 删除功能，都必须在下方「变更记录」表追加一行**，这是唯一的跨会话追踪机制：

```
| 日期 | 类型 | 模块/改动 | 关键文件 | 说明 |
```

- 类型：`新增` / `修改` / `修复` / `重构` / `规范`
- 关键文件写相对路径，一行说不完就列最重要的 2~4 个
- 规范本身的变更（分层、命名、流程）也要记录，类型标 `规范`

## 7. 变更记录

| 日期       | 类型 | 模块/改动 | 关键文件                                                                                                                          | 说明                                                                                                                                                                                                                                     |
| ---------- | ---- | --------- | --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-11-02 | 新增 | 项目骨架  | `main.py`, `app/**`, `docker-compose.yml`                                                                                         | FastAPI + PostgreSQL(15400) + Redis 骨架；健康检查、统一响应、Alembic 异步迁移                                                                                                                                                           |
| 2026-11-02 | 修改 | 入口位置  | `main.py`, `Dockerfile`, `docker-compose.yml`                                                                                     | 入口由 `app/main.py` 移至根目录，命令改为 `uvicorn main:app`；支持 `python main.py` 启动                                                                                                                                                 |
| 2026-11-02 | 修改 | 端口调整  | `.env`, `app/core/config.py`, `docker-compose.yml`                                                                                | 后端 8000 → 15401；CORS 允许前端 15173                                                                                                                                                                                                   |
| 2026-11-02 | 修复 | 镜像构建  | `Dockerfile`                                                                                                                      | pip 改用清华源 + 超时重试，解决构建时下载超时                                                                                                                                                                                            |
| 2026-11-05 | 新增 | 模型管理  | `app/models/model_config.py`, `app/crud/model_config.py`, `app/services/model_config.py`, `app/api/v1/endpoints/model_configs.py` | 三类模型配置 CRUD、默认模型互斥、连通测试、上下文测试；新增 `crud` 层                                                                                                                                                                    |
| 2026-11-05 | 规范 | 工程规范  | `AGENTS.md`, `CLAUDE.md`                                                                                                          | 建立目录分层、编码规范、六步新增流程与变更追踪机制；补齐全仓注释                                                                                                                                                                         |
| 2026-11-05 | 重构 | 目录结构  | `app/modules/**`, `app/views/**`, `app/core/schemas.py`, `app/deps.py`, `main.py`, `alembic/env.py`                               | 由"按层分目录"改为"按功能模块分目录"：models/schemas/repository(原 crud)/service 收敛进 `app/modules/<模块>`；接口视图独立为 `app/views/`；公共 Schema 移入 `core/schemas.py`，deps 移到 `app/deps.py`。路由路径与表结构不变，无需新迁移 |
| 2026-11-05 | 清理 | 缓存文件  | 全仓 `__pycache__/`, `*.pyc`                                                                                                      | 清除 Python 字节码缓存（运行时自动重建）；确认保留 alembic/ 与 alembic_version 表（迁移能力必需）                                                                                                                                        |
| 2026-11-05 | 文档 | 刷库流程  | `AGENTS.md` 第 2 节, `README.md`                                                                                                  | 将"改 models 后同步数据库"两步命令与注意点提升为高频操作章节                                                                                                                                                                             |
| 2026-11-05 | 重构 | 视图归属  | `app/modules/*/view.py`, `main.py`                                                                                                | 接口视图从独立的 `app/views/` 下沉到各模块内 `view.py`，路由改在 `main.py` 的 `create_app()` 中注册；接口路径不变                                                                                                                        |
| 2026-11-06 | 重构 | 模块内子包 | `app/modules/*/models/`, `schemas/`, `crud/`, `services/`, `views/` | 模块内单文件升级为分层子包：`repository` 改回 `crud`，各分层是文件夹、内部按功能拆多个 .py（model_config 已拆出 model_config / model_test 两个功能）；`views/__init__.py` 聚合路由并声明资源前缀；新增 `views/common.py` 提供 `get_model_or_404` 依赖。接口路径与表结构不变 |
| 2026-11-06 | 新增 | 智能体 | `app/modules/agent/**`, `app/utils/llm.py`, `alembic/versions/49cf9bbd74e5_*.py`, `main.py` | 智能体模块：会话表 + 消息表（Alembic 迁移已执行）；会话管理、多轮对话（SSE 流式 + 非流式）、模型选择（请求 > 会话绑定 > 默认 LLM）、历史消息拼 messages 充当"记忆"、失败/中断也落库。流式生成器内部自建数据库会话（请求级会话在 SSE 开始前已被关闭）。复用 `app/utils/llm.py`（从 model_test 抽出的地址拼接与请求头） |
| 2026-11-06 | 重构 | 公共工具 | `app/utils/llm.py`, `app/modules/model_config/services/model_test.py` | 把 OpenAI 兼容接口的 `chat/completions` 地址拼接与 Bearer 请求头抽成公共工具，model_test 改为复用，行为不变 |
| 2026-11-10 | 重命名 | 会话模块 | `app/modules/chat/**`（原 agent）, `alembic/versions/7648f48cf840_*.py`, `main.py` | 模块 agent → chat（原「智能体」只是会话功能，名字留给后续真正的智能体配置）：目录 / 类名（ChatConversation / ChatMessage）/ 接口前缀 `/api/v1/chat` 全部更名；两张表经迁移重命名为 `chat_conversation` / `chat_message`（索引、约束、序列同步改名，数据原样保留）；历史迁移 49cf9bbd74e5 不改写 |
| 2026-11-11 | 新增 | 学习实验台 | `app/modules/study/**`, `main.py` | 智能体开发学习模块：把 `scratch/day01` 的工具定义、Function Calling 循环、手写 ReAct 协议工程化（无表结构，只有 schemas/services/views 三层）；循环不再 print，返回结构化执行轨迹；模型接入复用「模型管理」默认 LLM（openai SDK 异步客户端，base_url 自动补 /v1）；接口 `/tools` `/agent/run` `/react/run`。配套前端 `/study` 页 |
| 2026-11-11 | 新增 | 学习工具  | `app/modules/study/services/tools.py` | Day 01 练习 1：新增 `search_news` 工具（按关键词返回 3 条假新闻标题），同步注册进 `TOOLS` / `TOOL_SCHEMAS`；description 写明"新闻类问题必须调用、禁止凭记忆编造"。Function Calling 与 ReAct 两个循环共用工具表，无需改循环代码 |
| 2026-11-12 | 修改 | ReAct 多轮 | `app/modules/study/services/react_loop.py`, `app/modules/study/schemas/study.py` | Day 01 练习 3：ReAct 循环支持多轮对话——history 由 run 函数局部变量提成 `ReactAgent` 实例成员，新增 `ReactSessionManager` 内存注册表（session_id → 实例，上限 50 插入序淘汰，重启即清空）；`/react/run` 请求新增可选 `session_id`，返回值新增 `session_id` / `history` / `turns`；每轮把「用户问题 + 模型输出 + Observation」并入跨轮记忆随 prompt 重发，模型可理解"那上海呢"式追问。views 无需改动（run 签名不变） |
