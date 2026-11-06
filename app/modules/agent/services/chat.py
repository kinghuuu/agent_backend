"""智能体对话的业务层：模型选择、历史消息组装、模型调用与消息落库。

学习要点：
1. **记忆**：每次请求都从 ``agent_message`` 表取出本会话的历史，拼成 OpenAI 协议的
   ``messages`` 数组发给模型——这就是最朴素的"多轮对话记忆"，靠数据库而非内存维持。
2. **流式**：用 SSE（``text/event-stream``）把 token 一片片推给前端，首字延迟低；
   协议上是若干个 ``data: {json}\\n\\n`` 事件块，前端按 ``type`` 字段区分处理。
3. **落库时机**：用户消息在调用模型前写库（刷新页面不丢），助手消息在生成结束
   （或中断 / 失败）后写库；中断时保存已生成的部分内容。
"""

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.session import SessionLocal
from app.modules.agent.crud import conversation as conversation_crud
from app.modules.agent.crud import message as message_crud
from app.modules.agent.models import AgentConversation, AgentMessage
from app.modules.agent.schemas import ChatRequest, ChatResult, MessageOut
from app.modules.model_config.models import ModelConfig
from app.modules.model_config.services import model_config as model_service
from app.utils.llm import auth_headers, chat_completions_url

logger = get_logger(__name__)

# 会话标题的占位值：标题等于它时，用首条用户消息自动命名
DEFAULT_TITLE = "新对话"
# 每次请求最多带上的历史消息条数（防止上下文无限增长把 token 打爆）
MAX_HISTORY_MESSAGES = 20
DEFAULT_SYSTEM_PROMPT = (
    "你是一个乐于助人的中文 AI 助手，回答要准确、简洁、条理清晰；"
    "不确定的内容要明确说明。"
)
# 生成长文本可能很久，读超时放宽；连接超时保持严格，地址写错能快速失败
LLM_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class ChatError(Exception):
    """模型调用失败（网络、鉴权、服务端报错等）。

    与业务校验的 ``ValueError`` 区分开，视图层据此返回 502 而不是 400。
    """


# ---------- 模型选择 ----------


async def list_available_models(db: AsyncSession) -> list[ModelConfig]:
    """智能体可选的模型：仅启用中的 LLM，默认模型排最前。"""
    items = await model_service.list_model_configs(db, "llm")
    enabled = [item for item in items if item.enabled]
    return sorted(enabled, key=lambda item: (not item.is_default, item.id))


async def resolve_model(
    db: AsyncSession,
    conversation: AgentConversation,
    model_config_id: int | None = None,
) -> ModelConfig:
    """确定本次对话使用的模型。

    优先级：请求指定 > 会话绑定 > 默认 LLM。指定模型不存在 / 已禁用 / 类型不对时
    静默回落，避免前端缓存了过期 id 就让对话完全不可用。
    """
    target_id = model_config_id or conversation.model_config_id
    if target_id:
        cfg = await model_service.get_model_config(db, target_id)
        if cfg and cfg.enabled and cfg.model_type == "llm":
            return cfg
    models = await list_available_models(db)
    if not models:
        raise ValueError("还没有可用的 LLM 模型，请先到「模型管理」中添加并启用")
    return models[0]


# ---------- 消息组装与落库 ----------


def _build_llm_messages(
    history: list[AgentMessage], system_prompt: str | None
) -> list[dict[str, str]]:
    """把数据库里的历史消息转成 OpenAI 协议的 messages 数组。

    跳过空内容的助手消息（失败 / 中断时可能为空），否则会把"空回复"喂回模型。
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT}
    ]
    for msg in history:
        if msg.role not in ("user", "assistant"):
            continue
        if not (msg.content or "").strip():
            continue
        messages.append({"role": msg.role, "content": msg.content})
    return messages


def _auto_title(content: str) -> str:
    """用首条用户消息生成会话标题（压成一行，最多 20 字）。"""
    flat = " ".join(content.split())
    return flat[:20] + ("…" if len(flat) > 20 else "")


async def _prepare(
    db: AsyncSession,
    conversation: AgentConversation,
    payload: ChatRequest,
    cfg: ModelConfig,
) -> tuple[AgentMessage, list[dict[str, str]]]:
    """落库用户消息 → 自动命名会话 → 组装本次请求的 messages。"""
    history = await message_crud.list_by_conversation(db, conversation.id)
    user_message = await message_crud.create(
        db,
        {
            "conversation_id": conversation.id,
            "role": "user",
            "content": payload.content,
            "status": "success",
        },
    )
    history.append(user_message)

    changes: dict[str, Any] = {}
    # 首条消息决定标题；用户手动改过名字就不覆盖
    if conversation.title in ("", DEFAULT_TITLE):
        changes["title"] = _auto_title(payload.content)
    # 记住本次使用的模型，下次打开会话自动沿用
    if conversation.model_config_id != cfg.id:
        changes["model_config_id"] = cfg.id
    if changes:
        await conversation_crud.update(db, conversation, changes)

    recent = history[-MAX_HISTORY_MESSAGES:]
    return user_message, _build_llm_messages(recent, payload.system_prompt)


async def _save_assistant(
    db: AsyncSession,
    conversation: AgentConversation,
    content: str,
    status: str,
    *,
    error_message: str | None = None,
    usage: dict | None = None,
    latency_ms: int | None = None,
) -> AgentMessage:
    """保存助手消息（成功 / 失败 / 中断三种状态共用），并刷新会话的更新时间。"""
    data: dict[str, Any] = {
        "conversation_id": conversation.id,
        "role": "assistant",
        "content": content,
        "status": status,
        "error_message": error_message,
        "latency_ms": latency_ms,
    }
    if usage:
        data["prompt_tokens"] = usage.get("prompt_tokens")
        data["completion_tokens"] = usage.get("completion_tokens")
    message = await message_crud.create(db, data)
    await conversation_crud.touch(db, conversation)
    return message


# ---------- 模型调用 ----------


def _elapsed_ms(start: float) -> int:
    """从计时起点到现在的毫秒数。"""
    return int((time.perf_counter() - start) * 1000)


async def _request_once(cfg: ModelConfig, messages: list[dict[str, str]]) -> dict:
    """非流式调用（一次性拿到完整回复）。"""
    body = {"model": cfg.model_name, "messages": messages, "stream": False}
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            resp = await client.post(
                chat_completions_url(cfg.base_url),
                json=body,
                headers=auth_headers(cfg.api_key),
            )
    except httpx.TimeoutException as exc:
        raise ChatError("模型服务响应超时，请稍后重试") from exc
    except httpx.HTTPError as exc:
        raise ChatError(f"连接模型服务失败：{exc}") from exc

    if resp.status_code >= 400:
        raise ChatError(f"模型服务返回 {resp.status_code}：{resp.text[:300]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise ChatError("模型服务返回的不是合法 JSON") from exc


def _parse_completion(data: dict) -> tuple[str, dict | None]:
    """从非流式响应里取出回复文本与 token 用量。"""
    choices = data.get("choices") or []
    if not choices:
        raise ChatError(f"模型未返回 choices：{str(data)[:300]}")
    content = ((choices[0].get("message") or {}).get("content") or "").strip()
    return content, data.get("usage")


def _sse(payload: dict) -> str:
    """把字典编码成一个 SSE 事件块（``data: {...}\\n\\n``）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _parse_sse_line(line: str) -> dict | None:
    """解析上游返回的一行 SSE，返回 JSON 数据；空行 / 心跳 / [DONE] 返回 None。"""
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


# ---------- 对外能力：非流式 / 流式 ----------


async def chat(
    db: AsyncSession,
    conversation: AgentConversation,
    payload: ChatRequest,
    cfg: ModelConfig,
) -> ChatResult:
    """非流式对话：一次返回完整回复（前端用不到可不用，便于调试与学习对比）。"""
    user_message, messages = await _prepare(db, conversation, payload, cfg)
    start = time.perf_counter()
    try:
        data = await _request_once(cfg, messages)
        content, usage = _parse_completion(data)
    except ChatError as exc:
        # 失败也落库，前端刷新后能看到这条失败的记录
        await _save_assistant(
            db,
            conversation,
            "",
            "error",
            error_message=str(exc),
            latency_ms=_elapsed_ms(start),
        )
        raise

    assistant = await _save_assistant(
        db,
        conversation,
        content,
        "success",
        usage=usage,
        latency_ms=_elapsed_ms(start),
    )
    return ChatResult(
        model_name=cfg.model_name,
        model_config_id=cfg.id,
        user_message=MessageOut.model_validate(user_message),
        assistant_message=MessageOut.model_validate(assistant),
    )


async def stream_chat(
    conversation_id: int, payload: ChatRequest
) -> AsyncIterator[str]:
    """流式对话：逐块产出 SSE 文本，边生成边推给前端。

    事件类型：
    - ``start``：已保存用户消息，附带该消息（前端用它替换本地占位消息）
    - ``delta``：一个增量 token 片段
    - ``done``：生成结束，附带落库后的助手消息
    - ``error``：失败（HTTP 状态码已经是 200，只能靠事件体告知前端）

    注意：这里**自己开一个数据库会话**，不复用请求级 ``get_db``。原因是 FastAPI 会在
    返回 StreamingResponse 之后立刻结束请求依赖，请求级会话此时已被关闭，继续使用会
    报 "Instance is not persistent within this Session"。
    """
    start = time.perf_counter()
    parts: list[str] = []
    usage: dict | None = None

    async with SessionLocal() as db:
        conversation = await conversation_crud.get(db, conversation_id)
        if conversation is None:
            yield _sse({"type": "error", "message": "会话不存在"})
            return

        try:
            cfg = await resolve_model(db, conversation, payload.model_config_id)
            user_message, messages = await _prepare(db, conversation, payload, cfg)
            yield _sse(
                {
                    "type": "start",
                    "model_name": cfg.model_name,
                    "user_message": MessageOut.model_validate(user_message).model_dump(
                        mode="json"
                    ),
                }
            )

            body = {"model": cfg.model_name, "messages": messages, "stream": True}
            async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
                async with client.stream(
                    "POST",
                    chat_completions_url(cfg.base_url),
                    json=body,
                    headers=auth_headers(cfg.api_key),
                ) as resp:
                    if resp.status_code >= 400:
                        detail = (await resp.aread()).decode("utf-8", "ignore")
                        raise ChatError(
                            f"模型服务返回 {resp.status_code}：{detail[:300]}"
                        )
                    async for line in resp.aiter_lines():
                        chunk = _parse_sse_line(line)
                        if chunk is None:
                            continue
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            parts.append(piece)
                            yield _sse({"type": "delta", "content": piece})

            assistant = await _save_assistant(
                db,
                conversation,
                "".join(parts),
                "success",
                usage=usage,
                latency_ms=_elapsed_ms(start),
            )
            yield _sse(
                {
                    "type": "done",
                    "message": MessageOut.model_validate(assistant).model_dump(
                        mode="json"
                    ),
                }
            )
        except asyncio.CancelledError:
            # 前端点了「停止生成」/ 断开连接：保存已生成的部分，再让取消继续向上传播
            try:
                await _save_assistant(
                    db,
                    conversation,
                    "".join(parts),
                    "stopped",
                    error_message="已停止生成",
                    usage=usage,
                    latency_ms=_elapsed_ms(start),
                )
            except BaseException:  # noqa: BLE001 - 取消失败不能再抛，否则掩盖原始取消
                logger.warning("停止生成后保存部分内容失败", exc_info=True)
            raise
        except Exception as exc:  # noqa: BLE001 - 统一转成 SSE error 事件
            message = str(exc) if isinstance(exc, ChatError) else f"调用模型失败：{exc}"
            logger.warning("流式对话失败：%s", message, exc_info=True)
            saved: AgentMessage | None = None
            try:
                saved = await _save_assistant(
                    db,
                    conversation,
                    "".join(parts),
                    "error",
                    error_message=message,
                    usage=usage,
                    latency_ms=_elapsed_ms(start),
                )
            except Exception:  # noqa: BLE001
                logger.exception("保存失败消息时出错")
                await db.rollback()
            yield _sse(
                {
                    "type": "error",
                    "message": message,
                    "message_id": saved.id if saved else None,
                }
            )
