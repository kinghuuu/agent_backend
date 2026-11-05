"""模型管理模块的业务层：默认模型互斥、连通测试、上下文测试。"""

import time

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.modules.model_config import repository as repo
from app.modules.model_config.models import ModelConfig
from app.modules.model_config.schemas import (
    ModelConfigCreate,
    ModelConfigUpdate,
    ModelTestResult,
)

logger = get_logger(__name__)


# ---------- CRUD + 业务规则 ----------

async def list_model_configs(
    db: AsyncSession, model_type: str | None = None
) -> list[ModelConfig]:
    """查询模型配置列表，可按类型（llm/embedding/rerank）过滤。"""
    return await repo.list_all(db, model_type)


async def get_model_config(db: AsyncSession, model_id: int) -> ModelConfig | None:
    """按 id 查询单条配置，不存在返回 None（由视图层转 404）。"""
    return await repo.get(db, model_id)


async def create_model_config(db: AsyncSession, payload: ModelConfigCreate) -> ModelConfig:
    """新增模型配置。

    业务规则：
    - 展示名称全局唯一，重名抛 ``ValueError``（视图层转 400）
    - 若设为默认，则取消同类型下其它记录的默认标记
    """
    if await repo.get_by_name(db, payload.name):
        raise ValueError(f"展示名称「{payload.name}」已存在")
    obj = await repo.create(db, payload.model_dump())
    if obj.is_default:
        await repo.clear_default(db, obj.model_type, exclude_id=obj.id)
    return obj


async def update_model_config(
    db: AsyncSession, obj: ModelConfig, payload: ModelConfigUpdate
) -> ModelConfig:
    """编辑模型配置，只更新请求中显式传入的字段（``exclude_unset``）。"""
    data = payload.model_dump(exclude_unset=True)
    if (name := data.get("name")) and name != obj.name:
        if await repo.get_by_name(db, name):
            raise ValueError(f"展示名称「{name}」已存在")
    obj = await repo.update(db, obj, data)
    if obj.is_default:
        await repo.clear_default(db, obj.model_type, exclude_id=obj.id)
    return obj


async def delete_model_config(db: AsyncSession, obj: ModelConfig) -> None:
    """删除模型配置（物理删除）。"""
    await repo.delete(db, obj)


# ---------- 模型连通性测试（OpenAI 兼容接口） ----------

def _chat_url(base_url: str) -> str:
    """拼接 chat/completions 地址，base_url 没带 /v1 时自动补上。"""
    base = base_url.rstrip("/")
    if not base.endswith("/v1") and "/v1/" not in base:
        base = f"{base}/v1"
    return f"{base}/chat/completions"


async def _chat(
    cfg: ModelConfig,
    messages: list[dict],
    *,
    max_tokens: int = 8,
    timeout: float = 30.0,
) -> dict:
    """调用 OpenAI 兼容的 chat/completions 接口并返回 JSON 响应。"""
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    payload = {
        "model": cfg.model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(_chat_url(cfg.base_url), json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()


def _fail(start: float, message: str) -> ModelTestResult:
    """构造失败的测试结果（统一记录 warning 日志）。"""
    logger.warning("模型测试失败: %s", message)
    return ModelTestResult(
        success=False,
        latency_ms=int((time.perf_counter() - start) * 1000),
        message=message,
    )


async def test_connectivity(cfg: ModelConfig) -> ModelTestResult:
    """连通测试：发一条短消息，验证地址 / 模型名 / Key 是否可用。"""
    start = time.perf_counter()
    try:
        data = await _chat(
            cfg, [{"role": "user", "content": "你好，请只回复：pong"}], max_tokens=8, timeout=15
        )
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return ModelTestResult(
            success=True,
            latency_ms=int((time.perf_counter() - start) * 1000),
            message=f"连通成功，模型回复：{reply or '(空)'}",
            detail=data.get("usage"),
        )
    except httpx.TimeoutException:
        return _fail(start, "连接超时（15s），请检查服务地址或网络")
    except httpx.HTTPStatusError as exc:
        return _fail(start, f"服务返回 {exc.response.status_code}: {exc.response.text[:200]}")
    except Exception as exc:  # noqa: BLE001
        return _fail(start, f"连接失败: {exc}")


async def test_context_length(cfg: ModelConfig, num_tokens: int = 4000) -> ModelTestResult:
    """上下文测试：发送约 num_tokens 长度的文本，验证模型能否正常处理。

    中文场景下按 1 字 ≈ 1 token 粗略估算，仅作连通性/长度能力的参考。
    """
    start = time.perf_counter()
    filler = "这是一段用于上下文长度测试的填充文本。"
    repeat = max(1, num_tokens // len(filler))
    content = filler * repeat
    messages = [
        {"role": "system", "content": "你是上下文测试助手，收到长文本后只回复两个字：收到"},
        {"role": "user", "content": content},
    ]
    try:
        data = await _chat(cfg, messages, max_tokens=16, timeout=60)
        reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        prompt_tokens = (data.get("usage") or {}).get("prompt_tokens")
        detail = {"sent_chars": len(content), "prompt_tokens": prompt_tokens}
        return ModelTestResult(
            success=True,
            latency_ms=int((time.perf_counter() - start) * 1000),
            message=f"上下文测试通过，模型回复：{reply or '(空)'}",
            detail=detail,
        )
    except httpx.TimeoutException:
        return _fail(start, "上下文测试超时（60s）")
    except httpx.HTTPStatusError as exc:
        return _fail(start, f"服务返回 {exc.response.status_code}: {exc.response.text[:200]}")
    except Exception as exc:  # noqa: BLE001
        return _fail(start, f"上下文测试失败: {exc}")
