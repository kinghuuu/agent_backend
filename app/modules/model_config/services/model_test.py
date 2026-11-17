"""模型连通测试的业务层：连通测试与上下文测试（OpenAI 兼容接口）。"""

import time

import httpx

from app.core.logging import get_logger
from app.modules.model_config.models import ModelConfig
from app.modules.model_config.schemas import ModelTestResult
from app.utils.llm import auth_headers, chat_completions_url, embeddings_url

logger = get_logger(__name__)

# ---------- 模型连通性测试（OpenAI 兼容接口） ----------


async def _chat(
    cfg: ModelConfig,
    messages: list[dict],
    *,
    max_tokens: int = 8,
    timeout: float = 30.0,
) -> dict:
    """调用 OpenAI 兼容的 chat/completions 接口并返回 JSON 响应。"""
    payload = {
        "model": cfg.model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            chat_completions_url(cfg.base_url),
            json=payload,
            headers=auth_headers(cfg.api_key),
        )
        resp.raise_for_status()
        return resp.json()


async def _embeddings(cfg: ModelConfig, text: str, *, timeout: float = 15.0) -> dict:
    """调用 OpenAI 兼容的 embeddings 接口并返回 JSON 响应。

    Embedding 模型不支持 chat 端点，必须单独走 ``/embeddings``，
    否则服务端会报 ``model_not_supported``（阿里云百炼就是这个表现）。
    """
    payload = {"model": cfg.model_name, "input": text, "encoding_format": "float"}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(
            embeddings_url(cfg.base_url),
            json=payload,
            headers=auth_headers(cfg.api_key),
        )
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
    """连通测试：按模型类型选对应端点，验证地址 / 模型名 / Key 是否可用。"""
    start = time.perf_counter()

    if cfg.model_type == "embedding":
        return await _test_embedding(cfg, start)

    if cfg.model_type == "rerank":
        return _fail(
            start,
            "Rerank 不遵循 OpenAI 标准协议，暂不支持自动探测；"
            "请用业务侧检索接口验证，或先在模型管理里保留配置待后续接入",
        )

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


async def _test_embedding(cfg: ModelConfig, start: float) -> ModelTestResult:
    """Embedding 连通测试：走 ``/embeddings``，成功时顺带报告向量维度。"""
    try:
        data = await _embeddings(cfg, "连通性测试", timeout=15)
        vectors = data.get("data") or []
        dimension = len(vectors[0].get("embedding") or []) if vectors else 0
        return ModelTestResult(
            success=True,
            latency_ms=int((time.perf_counter() - start) * 1000),
            message=f"连通成功，向量维度 {dimension}（返回 {len(vectors)} 条）",
            detail={"dimension": dimension, "usage": data.get("usage")},
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
    这是对话模型专属概念，Embedding / Rerank 没有对话上下文，直接给出说明。
    """
    if cfg.model_type != "llm":
        return ModelTestResult(
            success=False,
            latency_ms=0,
            message=f"上下文测试仅适用于 LLM；当前模型类型为 {cfg.model_type}，"
            "Embedding / Rerank 没有对话上下文概念，请用「连通测试」验证",
        )

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
