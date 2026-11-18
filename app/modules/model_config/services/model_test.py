"""模型连通测试的业务层：连通测试与上下文测试。

LLM 走 OpenAI 兼容的 ``/chat/completions``，Embedding 走 ``/embeddings``，
Rerank 走通用 ``/rerank``（Cohere 风格的 query + documents 请求体）。
"""

import time

import httpx

from app.core.logging import get_logger
from app.modules.model_config.models import ModelConfig
from app.modules.model_config.schemas import ModelTestResult
from app.utils.llm import auth_headers, chat_completions_url, embeddings_url, rerank_urls

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
        return await _test_rerank(cfg, start)

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


# rerank 探测用固定样例：1 条相关 + 2 条无关，用于确认端点真的在做「相关性排序」
_RERANK_QUERY = "向量检索有什么用处"
_RERANK_DOCUMENTS = [
    "向量检索把文本映射成向量，再按语义相似度召回相关内容。",
    "红烧肉的做法：五花肉切块，加冰糖炒糖色后小火慢炖。",
    "Python 的 GIL 会限制多线程的并行执行效率。",
]


def _rerank_results(data: object) -> list[dict]:
    """从 rerank 响应中取出排序结果，兼容两种形态。

    - Cohere / Jina / 硅基流动：``{"results": [{"index", "relevance_score"}, ...]}``
    - 本地 HuggingFace TEI：直接是一个裸数组 ``[{"index", "score"}, ...]``

    非预期结构返回空列表，由调用方判定为失败。
    """
    raw = data.get("results") if isinstance(data, dict) else data
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


async def _test_rerank(cfg: ModelConfig, start: float) -> ModelTestResult:
    """Rerank 连通测试：发一次 ``query + documents`` 请求，看是否返回排序结果。

    Rerank 没有官方统一协议，但主流实现（Cohere / Jina / 硅基流动 / 本地
    Xinference）都用 ``POST /v1/rerank``，请求体为 ``{model, query, documents}``；
    本地 TEI 只暴露根路径 ``/rerank``。因此候选地址按 404 依次回退，其余错误报错。

    判定成功的标准是「拿到了排序结果且带分数」——只有真正的 rerank 端点才会
    返回这个结构，普通 chat / embedding 端点不会。
    """
    payload = {
        "model": cfg.model_name,
        "query": _RERANK_QUERY,
        "documents": _RERANK_DOCUMENTS,
        "top_n": len(_RERANK_DOCUMENTS),
    }
    tried: list[str] = []

    async with httpx.AsyncClient(timeout=15) as client:
        for url in rerank_urls(cfg.base_url):
            tried.append(url)
            try:
                resp = await client.post(
                    url, json=payload, headers=auth_headers(cfg.api_key)
                )
            except httpx.TimeoutException:
                return _fail(start, "连接超时（15s），请检查服务地址或网络")
            except Exception as exc:  # noqa: BLE001
                return _fail(start, f"连接失败: {exc}")

            # 404 说明这个路径下没有 rerank 接口，换下一个候选地址继续试
            if resp.status_code == 404:
                continue
            if resp.status_code >= 400:
                return _fail(
                    start, f"服务返回 {resp.status_code}: {resp.text[:200]}"
                )

            try:
                data = resp.json()
            except ValueError:
                return _fail(start, f"{url} 返回的不是 JSON，可能不是 rerank 接口")

            results = _rerank_results(data)
            if not results:
                return _fail(
                    start,
                    "接口可访问但没有返回排序结果：请确认「实际模型名」是 rerank 模型"
                    f"（如 BAAI/bge-reranker-v2-m3），当前为 {cfg.model_name}",
                )

            top = results[0]
            score = top.get("relevance_score", top.get("score"))
            return ModelTestResult(
                success=True,
                latency_ms=int((time.perf_counter() - start) * 1000),
                message=(
                    f"连通成功，返回 {len(results)} 条排序结果；"
                    f"Top1 是文档 #{top.get('index', 0)}，相关度 {score}"
                ),
                detail={
                    "url": url,
                    "results": results,
                    "usage": data.get("usage") if isinstance(data, dict) else None,
                },
            )

    return _fail(
        start,
        "未找到 rerank 端点，已尝试：" + "、".join(tried) + "；请检查服务地址是否正确",
    )


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
