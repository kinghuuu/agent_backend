"""Day 08 单测：上下文编排器（去重 / 位置学 / 裁剪方向 / 预算 / 七问自检）。

运行：``python -m pytest tests/test_context_builder.py -p no:cacheprovider -q``
全部是纯逻辑，不依赖 DB 与模型。
"""

from app.modules.study.services import context_builder as cb


def _block(text: str, chunk_index: int, score: float, **extra) -> dict:
    """构造一个检索块（字段与 RAG 命中的子集对齐）。"""
    block = {
        "text": text,
        "title": "产品手册",
        "chunk_index": chunk_index,
        "score": score,
        "section": "规格与保修",
    }
    block.update(extra)
    return block


DEMO_BLOCKS = [
    _block("XR-500 Pro 保修期为两年，电池保修一年。", 5, 0.91),
    _block("年卡会员退款免运费，且享 95 折优惠。", 2, 0.78),
    _block("自签收之日起 30 天内可无理由退款，运费 10 元由买家承担。", 1, 0.74),
    _block("保修期为两年，电池保修一年。", 6, 0.70),  # 块 5 的子集
]


# ---------- 去重 ----------


def test_dedup_drops_subset_block() -> None:
    """块 6 是块 5 的子集：Jaccard 只有约 0.6，必须靠包含度才抓得住。"""
    kept, dropped = cb.dedup_blocks(DEMO_BLOCKS)
    assert [block["chunk_index"] for block in kept] == [5, 2, 1]
    assert [block["chunk_index"] for block in dropped] == [6]
    assert "重合度" in dropped[0]["reason"]


def test_dedup_keeps_distinct_blocks() -> None:
    """内容不同的块不能被误杀（去重是"同一件事"而不是"同一主题"）。"""
    kept, dropped = cb.dedup_blocks(DEMO_BLOCKS[1:3])
    assert len(kept) == 2
    assert dropped == []


# ---------- 位置学 ----------


def test_order_blocks_by_position_u_shape() -> None:
    """U 型摆放：top1 头、top2 尾、垫底的填中间。"""
    kept, _ = cb.dedup_blocks(DEMO_BLOCKS)
    ordered = cb.order_blocks_by_position(kept)
    assert [block["chunk_index"] for block in ordered] == [5, 1, 2]


def test_order_blocks_short_list_is_identity() -> None:
    """只有一到两块时保持原顺序（没有中间可填）。"""
    two = DEMO_BLOCKS[:2]
    assert cb.order_blocks_by_position(two) == two


# ---------- token 级裁剪 ----------


def test_crop_block_narrative_keeps_head() -> None:
    """narrative 是总-分结构：结论在段首，截尾保头。"""
    long_text = "## 结论\n可退款。" + "补充说明。" * 60
    rendered = cb.crop_block(_block(long_text, 1, 0.9), 60)
    assert rendered.startswith("[来源: 产品手册§规格与保修 · 块1] ## 结论")
    assert rendered.endswith("……（已截断）")


def test_crop_block_clause_keeps_tail() -> None:
    """clause 类文档的例外条款写在最后：截头保尾（课程练习 2）。"""
    long_text = "一般条款。" * 60 + "例外：拆封后不支持无理由退款。"
    rendered = cb.crop_block(_block(long_text, 2, 0.9, style="clause"), 60)
    assert "例外：拆封后不支持无理由退款。" in rendered
    assert rendered.split("] ")[1].startswith("（前文已截断）")


def test_crop_block_keeps_source_when_no_room() -> None:
    """连正文都放不下时只留来源——信息打折，来源不打折。"""
    rendered = cb.crop_block(_block("保修两年", 3, 0.5), 5)
    assert rendered == "[来源: 产品手册§规格与保修 · 块3]"


# ---------- 画像裁剪（练习 1） ----------


def _facts(count: int) -> list[dict]:
    """构造 count 条事实，updated_at 每分钟递增（越靠后越新）。"""
    return [
        {
            "key": f"k{index:02d}",
            "value": f"值{index}",
            "updated_at": f"2026-11-{index // 24 + 1:02d} {index % 24:02d}:00",
        }
        for index in range(count)
    ]


def test_prune_profile_keeps_recent_first() -> None:
    """画像超预算时按 updated_at 保留最近的，被裁的 key 全数上报。"""
    text, dropped = cb.prune_profile(_facts(20), max_tokens=90)
    assert text.startswith("【用户长期画像】")
    assert dropped, "预算 90 tok 装不下 20 条，必须有被裁的"
    for key in dropped:
        assert f"- {key}:" not in text
    # 保留的条目按原顺序输出，不会变成倒序
    kept_indexes = [
        int(line.split(":")[0].replace("- k", "")) for line in text.splitlines()[1:]
    ]
    assert kept_indexes == sorted(kept_indexes)


def test_prune_profile_without_facts() -> None:
    """没有画像时返回空串，让上层直接跳过这一层。"""
    assert cb.prune_profile([], 200) == ("", [])


# ---------- 对话层裁剪与工具配对 ----------


def _history_with_tool() -> list[dict]:
    """一轮带工具调用的完整历史 + 一轮普通问答。"""
    return [
        {"role": "user", "content": "帮我查订单 O20260012"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t1", "type": "function", "function": {"name": "search_orders", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "已发货"},
        {"role": "user", "content": "那退款要运费吗"},
        {"role": "assistant", "content": "30 天内可退，运费 10 元"},
    ]


def test_fit_history_keeps_whole_rounds() -> None:
    """按轮裁剪：保留的首条必须是 user，不能从 assistant 中间切开。"""
    kept, evicted = cb.fit_history(_history_with_tool(), max_tokens=40)
    assert kept, "最后一轮至少要留下"
    assert kept[0]["role"] == "user"
    assert evicted, "预算 40 tok 装不下两轮，必须有被淘汰的"


def test_check_tool_pairing_drops_orphan() -> None:
    """孤儿 tool 消息必须被丢掉并报警（否则下一轮 API 直接 400）。"""
    messages = [*_history_with_tool(), {"role": "tool", "tool_call_id": "t9", "content": "野消息"}]
    cleaned, warnings = cb.check_tool_pairing(messages)
    assert all(message.get("tool_call_id") != "t9" for message in cleaned)
    assert any("t9" in warning for warning in warnings)


def test_check_tool_pairing_reports_missing_response() -> None:
    """有 tool_calls 却缺工具响应（半轮）也要报出来。"""
    messages = [
        {"role": "user", "content": "查订单"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "t2", "type": "function", "function": {"name": "search_orders", "arguments": "{}"}}
            ],
        },
    ]
    cleaned, warnings = cb.check_tool_pairing(messages)
    assert len(cleaned) == 2
    assert any("t2" in warning for warning in warnings)


# ---------- 整体编排 ----------


def _request(**extra) -> cb.ContextRequest:
    """一份覆盖七种原料的编排请求。"""
    payload = {
        "system_prompt": "你是客服 Agent。\n1. 实时数据必须调工具查询。\n2. 关键结论必须标注来源。",
        "user_query": "那退款要运费吗？",
        "hard_constraints": ["实时数据必须调工具查询", "关键结论必须标注来源"],
        "short_memory": "【会话摘要】用户咨询 XR-500 Pro 的售后问题，情绪平稳。",
        "profile_facts": [{"key": "membership", "value": "年卡会员", "updated_at": "2026-11-06 10:00"}],
        "rag_blocks": [dict(block) for block in DEMO_BLOCKS],
        "tool_data": "order_status: 已发货（今日 09:12 查询）",
        "tools_schema": '{"name": "search_orders"}',
        "history": _history_with_tool()[:3],
    }
    payload.update(extra)
    return cb.ContextRequest(**payload)


def test_build_four_layers_and_dump() -> None:
    """四层齐全、dump 可读、上下文末尾是当前问题。"""
    result = cb.ContextBuilder().build(_request(), debug=True)
    assert [layer.key for layer in result.layers] == ["instruction", "state", "knowledge", "history"]
    assert result.context.rstrip().endswith("user: 那退款要运费吗？")
    assert "上下文 dump" in result.dump
    assert "七问自检" in result.dump
    assert len(result.checks) == 7


def test_build_reminder_sits_right_before_question() -> None:
    """约束复述必须在当前问题的紧前面，中间不能再隔历史。"""
    result = cb.ContextBuilder().build(_request())
    tail = result.context.split("【再次提醒（硬约束，必须遵守）】")[1]
    assert tail.strip().startswith("实时数据必须调工具查询")
    assert tail.strip().endswith("user: 那退款要运费吗？")


def test_build_places_top1_head_and_top2_tail() -> None:
    """块 6 被去重，剩下三块按 5（头）/1（中）/2（尾）摆放。"""
    result = cb.ContextBuilder().build(_request())
    placed = {block.chunk_index: block.position for block in result.blocks if block.position}
    assert placed == {5: 1, 1: 2, 2: 3}
    assert any(block.action == "去重丢弃" for block in result.blocks)


def test_build_reports_all_checks() -> None:
    """正常输入下七问全部通过（自检不是摆设，正常数据不该误报）。"""
    result = cb.ContextBuilder().build(_request())
    assert [check.no for check in result.checks] == [1, 2, 3, 4, 5, 6, 7]
    assert [check.no for check in result.checks if not check.ok] == []


def test_build_warns_on_over_budget() -> None:
    """工具说明书撑爆预算时，必须报警而不是悄悄超限。"""
    result = cb.ContextBuilder(cb.Budget(total=1200)).build(
        _request(tools_schema="x" * 3000)
    )
    assert any("超预算" in warning for warning in result.warnings)
    assert any("90%" in warning for warning in result.warnings)


def test_build_reports_profile_pruning() -> None:
    """画像超预算时把被裁的 key 记进 warnings（课程练习 1 的落点）。"""
    budget = cb.Budget(total=2000)
    result = cb.ContextBuilder(budget).build(_request(profile_facts=_facts(20)))
    assert any("画像超预算" in warning for warning in result.warnings)


def test_build_drops_orphan_tool_and_flags_check_five() -> None:
    """编排层拦下孤儿 tool 消息，自检 #5 变成不通过并给出原因。"""
    history = [*_history_with_tool(), {"role": "tool", "tool_call_id": "t9", "content": "野消息"}]
    result = cb.ContextBuilder().build(_request(history=history))
    check = next(item for item in result.checks if item.no == 5)
    assert check.ok is False
    assert "t9" in check.detail


def test_build_ignores_duplicated_current_question() -> None:
    """history 末尾与 user_query 重复时不重复注入同一句话。"""
    history = [
        {"role": "user", "content": "退款要运费吗？"},
        {"role": "assistant", "content": "年卡会员免运费"},
        {"role": "user", "content": "那退款要运费吗？"},
    ]
    result = cb.ContextBuilder().build(_request(history=history))
    assert result.context.count("那退款要运费吗？") == 1


def test_budget_ratios_add_up() -> None:
    """预算表比例 + 输出预留必须正好是 1，否则账单对不上。"""
    budget = cb.Budget()
    total_ratio = (
        budget.instruction
        + budget.tools
        + budget.state
        + budget.knowledge
        + budget.history
        + budget.reserve
    )
    assert abs(total_ratio - 1.0) < 1e-9
