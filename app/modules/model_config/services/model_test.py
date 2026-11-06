"""模型连通测试的业务层：连通测试与上下文测试（OpenAI 兼容接口）。"""

import time

import httpx

from app.core.logging import get_logger
from app.modules.model_config.models import ModelConfig
from app.modules.model_config.schemas import ModelTestResult

logger = get_logger(__name__)

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
