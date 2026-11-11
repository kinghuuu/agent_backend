"""学习模块的模型接入：复用「模型管理」里配置的 LLM，构造 OpenAI SDK 客户端。

与 scratch/day01 用 .env 配模型不同，工程化版本直接读数据库里的模型配置——
模型管理页改了配置，学习页立刻生效，两套代码共用一份数据源。
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.model_config.models import ModelConfig
from app.modules.model_config.services import model_config as model_config_service


def openai_base_url(base_url: str) -> str:
    """把数据库里的服务地址补成 OpenAI SDK 需要的形式（以 /v1 结尾）。

    SDK 自己会在 base_url 后拼 ``/chat/completions``，所以这里只负责保证
    地址带 ``/v1``；库里存的是 https://api.deepseek.com 时补成 .../v1。
    """
    base = base_url.rstrip("/")
    if not base.endswith("/v1") and "/v1/" not in base:
        base = f"{base}/v1"
    return base


def build_client(cfg: ModelConfig):
    """按模型配置构造异步 OpenAI 客户端（兼容 DeepSeek 等任何 OpenAI 协议服务）。"""
    from openai import AsyncOpenAI  # 延迟导入：模块加载时不依赖 openai 是否已安装

    return AsyncOpenAI(api_key=cfg.api_key or "EMPTY", base_url=openai_base_url(cfg.base_url))


async def resolve_model(db: AsyncSession, model_config_id: int | None = None) -> ModelConfig:
    """确定学习页使用的模型：请求指定 > 默认 LLM。

    指定的模型不存在 / 已禁用 / 类型不对时静默回落到默认模型，
    避免前端缓存了过期 id 就让学习页完全不可用。
    """
    if model_config_id:
        cfg = await model_config_service.get_model_config(db, model_config_id)
        if cfg and cfg.enabled and cfg.model_type == "llm":
            return cfg
    items = await model_config_service.list_model_configs(db, "llm")
    enabled = [item for item in items if item.enabled]
    if not enabled:
        raise ValueError("还没有可用的 LLM 模型，请先到「模型管理」中添加并启用")
    enabled.sort(key=lambda item: (not item.is_default, item.id))
    return enabled[0]
