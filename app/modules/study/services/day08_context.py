"""Day 08 组装：把真实原料喂给 ContextBuilder，产出最终上下文。

编排逻辑本身在 ``context_builder.py``（纯函数、零 I/O、可单测）；本文件只做三件事：
1. 按请求拉真实数据（长期画像 / 混合检索 / 工具说明书）；
2. 把手填原料与真实数据合并成 ``ContextRequest``；
3. 给前端标注每样原料的来源（真实库 / 手填演示），避免"以为是真实数据"。

刻意不调模型：上下文工程的迭代对象是"拼装结果"，先能肉眼审查 dump，
再去关心模型答得怎么样——反了就会陷入"模型变笨 → 改 prompt"的死循环。
"""

import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import ContextBuildRequest
from app.modules.study.services import context_builder, long_memory, rag, tools

# 课程 4.2 的系统提示词：硬约束原句都在里面，尾部还会再复述一次（自检 #1 会比对）
DEFAULT_SYSTEM_PROMPT = """你是客服 Agent。回答规则：
1. 优先使用【用户长期画像】与【检索到的资料】中的信息。
2. 若资料与用户陈述冲突：摆出双方说法，禁止缝合输出。
3. 实时数据必须调工具查询：订单状态、库存、金额、物流一律查工具，禁止用资料或记忆回答。
4. 关键结论必须标注来源：格式 `[来源: 文档名 §节]`。"""

# 只复述最硬的两条：复述太多反而互相稀释
DEFAULT_CONSTRAINTS = ["实时数据必须调工具查询", "关键结论必须标注来源"]

# 没接 Day 06 会话时的示例摘要
DEMO_SHORT_MEMORY = "【会话摘要】用户咨询 XR-500 Pro 的售后问题，情绪平稳。"

# 课程 4.2 的四个检索块：块 6 与块 5 字面高度重合，正好演示去重
DEMO_BLOCKS: list[dict[str, Any]] = [
    {
        "text": "XR-500 Pro 保修期为两年，电池保修一年。",
        "title": "产品手册",
        "chunk_index": 5,
        "score": 0.91,
        "section": "规格与保修",
        "style": "narrative",
    },
    {
        "text": "年卡会员退款免运费，且享 95 折优惠。",
        "title": "退款政策",
        "chunk_index": 2,
        "score": 0.78,
        "section": "会员权益",
        "style": "clause",
    },
    {
        "text": "自签收之日起 30 天内可无理由退款，运费 10 元由买家承担。",
        "title": "退款政策",
        "chunk_index": 1,
        "score": 0.74,
        "section": "退款政策",
        "style": "clause",
    },
    {
        "text": "保修期为两年，电池保修一年。",
        "title": "产品手册",
        "chunk_index": 6,
        "score": 0.70,
        "section": "规格与保修",
        "style": "narrative",
    },
]

# 没有真实用户画像时的示例（让人一眼看出"画像被裁"长什么样）
DEMO_PROFILE_FACTS: list[dict[str, Any]] = [
    {"key": "membership", "value": "年卡会员", "updated_at": "2026-11-06 10:00"},
]


async def build(db: AsyncSession, payload: ContextBuildRequest) -> dict[str, Any]:
    """执行一次编排：画像 / 检索块按需拉真实数据，其余用请求里的手填原料。"""
    sources: dict[str, str] = {}

    # ① 画像类：传 user_id 就读真实长期记忆，否则用手填文本或演示事实
    facts: list[dict[str, Any]] = []
    profile_text = payload.profile_text.strip()
    if payload.user_id:
        facts = await long_memory.list_facts(db, payload.user_id)
        sources["profile"] = f"长期记忆（{payload.user_id}，{len(facts)} 条）"
    elif profile_text:
        sources["profile"] = "手填画像文本"
    else:
        facts = DEMO_PROFILE_FACTS
        sources["profile"] = "演示事实（未指定用户）"

    # ② 文档类：传 rag_query 就真实混合检索，否则用手填块（都没有则用课程演示块）
    blocks: list[dict[str, Any]] = [block.model_dump() for block in payload.rag_blocks]
    if payload.rag_query:
        where = {"doc_id": payload.doc_id} if payload.doc_id else None
        hits = await rag.search(
            db, payload.rag_query, top_k=payload.top_k, alpha=payload.alpha, where=where
        )
        blocks = [
            {
                "text": hit["content"],
                "title": hit["doc_title"],
                "chunk_index": hit["chunk_index"],
                "score": hit["rrf"],
                "section": hit.get("section"),
                "version": hit.get("doc_version"),
            }
            for hit in hits
        ]
        sources["rag"] = f"混合检索「{payload.rag_query}」命中 {len(hits)} 块"
    elif blocks:
        sources["rag"] = f"手填检索块 {len(blocks)} 个"
    else:
        blocks = [dict(block) for block in DEMO_BLOCKS]
        sources["rag"] = "课程演示块（4 个，含 1 个重复块）"

    # ③ 工具说明书：直接序列化真实注册表，成本是每周重发的隐性开销
    tools_schema = ""
    if payload.with_tools:
        tools_schema = json.dumps(tools.TOOL_SCHEMAS, ensure_ascii=False)
        sources["tools"] = f"真实工具注册表（{len(tools.TOOL_SCHEMAS)} 个工具）"
    else:
        sources["tools"] = "未注入"

    request = context_builder.ContextRequest(
        system_prompt=payload.system_prompt or DEFAULT_SYSTEM_PROMPT,
        user_query=payload.user_query,
        hard_constraints=payload.hard_constraints or DEFAULT_CONSTRAINTS,
        short_memory=payload.short_memory or DEMO_SHORT_MEMORY,
        profile_facts=facts,
        profile_text=profile_text,
        rag_blocks=blocks,
        tool_data=payload.tool_data,
        tools_schema=tools_schema,
        history=payload.history,
    )
    result = context_builder.ContextBuilder(
        context_builder.Budget(total=payload.total_budget)
    ).build(request, debug=payload.debug)

    data = result.as_dict()
    data["sources"] = sources
    data["profile_text"] = long_memory.recall_prompt(facts) if facts else ""
    return data
