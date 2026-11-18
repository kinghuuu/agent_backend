"""Day 07 组装：把三条外部知识管道装进一个 Agent。

    画像类（长期记忆） → 会话开始全文注入 system
    文档类（RAG）      → 每轮按需混合检索，带来源标注注入
    事实类（工具）      → 交给 Day 03~05 的 Function Calling 循环实时查

再加两条工程保险：
1. history-aware 查询改写：多轮里的「那它呢」先还原成独立问句再检索；
2. 冲突裁决：用户陈述与资料金额不一致时，把冲突显式摆进 system，禁止缝合。
"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import AgentRunRequest, Day07ChatRequest
from app.modules.study.services import agent_loop, long_memory, rag
from app.modules.study.services.llm import build_client, resolve_model

# Day 07 追加的规则（课程第 4 节：摆冲突、禁缝合、标来源）
DAY07_RULES = """## 检索与记忆使用规则（Day 07）
1. 优先使用【用户长期画像】与【检索到的资料】，关键结论后用 `[来源: 文档名 §节]` 标注。
2. 资料与用户陈述冲突时：把双方说法都摆出来，建议以订单页实时数据为准，禁止各取一半缝合。
3. 订单状态、库存、金额、物流等实时数据一律调用工具查询，禁止用资料或记忆回答。
4. 检索资料为空、画像里也没有相关信息时，如实说"没有查到"，不要编造。"""

REWRITE_PROMPT = """把用户的最新问题改写成一句可以独立检索的完整问句。
要求：还原代词（它/这个/那个）为上文里的具体对象；保留型号、订单号等专有名词；
只输出改写后的问句，不要解释。

对话历史：
{history}

最新问题：{question}
"""


async def _rewrite_query(
    db: AsyncSession, history: list[dict[str, str]], question: str, model_config_id: int | None
) -> str:
    """多轮场景下把问题改写成独立问句；失败或无需改写时原样返回。"""
    if not history:
        return question
    try:
        cfg = await resolve_model(db, model_config_id)
        client = build_client(cfg)
        history_text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in history[-6:]
        )
        resp = await client.chat.completions.create(
            model=cfg.model_name,
            messages=[
                {
                    "role": "user",
                    "content": REWRITE_PROMPT.format(history=history_text, question=question),
                }
            ],
            temperature=0,
            max_tokens=120,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or question
    except Exception:  # noqa: BLE001  改写只是锦上添花，失败不能挡检索
        return question


def _build_system(profile: str, rag_context: str, conflicts: list[dict[str, Any]]) -> str:
    """拼装本轮 system：基础规则 + Day07 规则 + 画像 + 资料 + 冲突提示。"""
    parts = [agent_loop.SYSTEM_PROMPT, DAY07_RULES]
    if profile:
        parts.append(profile)
    if rag_context:
        parts.append(rag_context)
    if conflicts:
        lines = []
        for item in conflicts:
            lines.append(
                f"- {item['type']}：用户提到 {item['user_values']}，资料写的是 {item['doc_values']}。"
                f"{item['hint']}"
            )
        parts.append("【冲突提示（必须摆上台面，禁止缝合）】\n" + "\n".join(lines))
    return "\n\n".join(parts)


async def chat(db: AsyncSession, payload: Day07ChatRequest) -> dict[str, Any]:
    """执行一次 Day07 问答：画像 → 改写 → 检索 → 冲突 → Agent 循环 → 归档。"""
    # ① 画像类：会话开始全文注入
    facts = await long_memory.list_facts(db, payload.user_id)
    profile = long_memory.recall_prompt(facts)

    # ② 查询改写（多轮时把代词还原，否则"那它呢"检索必挂）
    query = await _rewrite_query(db, payload.history, payload.message, payload.model_config_id)

    # ③ 文档类：混合检索（可带元数据过滤）
    where = {"doc_id": payload.doc_id} if payload.doc_id else None
    hits = await rag.search(db, query, top_k=payload.top_k, alpha=payload.alpha, where=where)
    rag_context = rag.to_context(hits)

    # ④ 冲突裁决：把金额矛盾显式摆进上下文
    conflicts = rag.detect_numeric_conflicts(payload.message, hits)

    # ⑤ 事实类 + 生成：复用 Day 05 循环（工具仍可调用）
    system = _build_system(profile, rag_context, conflicts)
    run_result = await agent_loop.run(
        db,
        AgentRunRequest(
            question=payload.message,
            budget_profile=payload.budget_profile,
            max_steps=payload.max_steps,
            model_config_id=payload.model_config_id,
            user_confirmed=payload.user_confirmed,
        ),
        initial_messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": payload.message},
        ],
    )

    return {
        "answer": run_result["final_answer"],
        "user_id": payload.user_id,
        "profile": profile,
        "recalled_facts": facts,
        "rewritten_query": query,
        "hits": hits,
        "conflicts": conflicts,
        "run": run_result,
        "pipeline": {
            "profile_facts": len(facts),
            "retrieved_chunks": len(hits),
            "conflicts": len(conflicts),
            "loop_steps": len(run_result.get("steps", [])),
            "tool_calls": sum(
                len(step.get("tool_calls") or [])
                for step in run_result.get("steps", [])
            ),
            "stop_reason": run_result.get("stop_reason"),
        },
    }
