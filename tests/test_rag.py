"""Day 07 单测：切块 / 哈希向量 / BM25 / RRF / 冲突检测（纯逻辑，不依赖 DB 与模型）。

运行：``python -m pytest tests/test_rag.py -p no:cacheprovider -q``
"""

from app.modules.study.services import long_memory, rag


# ---------- 切块 ----------


def test_chunk_document_splits_by_section() -> None:
    """按 Markdown 标题分节：退款与会员权益不能混进同一个语义单元。"""
    text = (
        "## 退款政策\n30 天内可无理由退款。运费 10 元由买家承担。\n"
        "## 会员权益\n年卡会员退款免运费。"
    )
    chunks = rag.chunk_document(text, size=30, overlap=0)
    assert len(chunks) >= 2
    sections = {chunk["section"] for chunk in chunks}
    assert "退款政策" in sections
    assert "会员权益" in sections


def test_chunk_document_overlap_keeps_tail() -> None:
    """相邻块保留重叠：后一块开头必须包含前一块结尾，防止关键句被拦腰斩断。"""
    text = "\n".join(f"第{index}段内容" for index in range(20))
    chunks = rag.chunk_document(text, size=20, overlap=5)
    assert len(chunks) > 1
    assert chunks[1]["content"].startswith(chunks[0]["content"][-5:])


def test_chunk_document_empty() -> None:
    """空文档不产出切块。"""
    assert rag.chunk_document("   \n\n  ") == []


# ---------- 哈希向量 ----------


def test_hash_embed_normalized() -> None:
    """哈希回退向量做 L2 归一化，点积即余弦。"""
    vec = rag.hash_embed("退款政策")
    assert len(vec) == rag.HASH_DIM
    assert abs(sum(value * value for value in vec) - 1.0) < 1e-6


def test_cosine_identical_is_one() -> None:
    """同一向量余弦相似度为 1。"""
    vec = rag.hash_embed("XR-500 Pro 保修")
    assert abs(rag.cosine(vec, vec) - 1.0) < 1e-6


# ---------- BM25 ----------


def test_bm25_prefers_exact_model_number() -> None:
    """专有名词场景：BM25 让精确命中 XR-500 Pro 的块排第一，不被语义相似带偏。"""
    docs = [
        "我们的旗舰机型保修两年。",
        "XR-500 Pro 的保修期为两年。",
        "XR-300 的保修期为一年。",
    ]
    scores = rag.BM25(docs).scores("XR-500 Pro 保修多久")
    assert max(range(len(docs)), key=lambda index: scores[index]) == 1


# ---------- RRF ----------


def test_rrf_rewards_both_channels() -> None:
    """两路都靠前的文档，融合后应登顶（只比排名，不比原始分数量纲）。"""
    fused = rag.rrf_fuse([0, 2, 1], [1, 0, 2], alpha=0.5)
    assert max(fused, key=lambda index: fused[index]) == 0


def test_rrf_alpha_extremes() -> None:
    """alpha=1 时完全由向量路决定排名。"""
    fused = rag.rrf_fuse([0, 1], [1, 0], alpha=1.0)
    assert max(fused, key=lambda index: fused[index]) == 0


# ---------- 冲突检测 ----------


def test_detect_numeric_conflicts() -> None:
    """用户说的金额与资料金额不一致时，必须摆上台面。"""
    hits = [{"content": "运费 10 元由买家承担"}]
    conflicts = rag.detect_numeric_conflicts("客服说运费 8 元", hits)
    assert conflicts
    assert conflicts[0]["user_values"] == [8.0]
    assert conflicts[0]["doc_values"] == [10.0]
    # 用户没提数字时不制造冲突
    assert rag.detect_numeric_conflicts("运费多少", hits) == []


def test_detect_numeric_conflicts_no_false_positive() -> None:
    """金额一致时不算冲突。"""
    hits = [{"content": "运费 10 元"}]
    assert rag.detect_numeric_conflicts("运费 10 元吗", hits) == []


# ---------- 长期记忆纯逻辑 ----------


def test_memory_conflict_scan_finds_synonyms() -> None:
    """不同叫法指向同一事实且取值不同 → 提示合并（课程练习 1）。"""
    facts = [
        {"key": "dietary", "value": "吃香菜"},
        {"key": "food_preference", "value": "不吃香菜"},
    ]
    conflicts = long_memory.detect_conflicts(facts)
    assert len(conflicts) == 1
    assert set(conflicts[0]["keys"]) == {"dietary", "food_preference"}


def test_memory_updated_at_converted_to_local() -> None:
    """回归：数据库返回 UTC 时间，必须转本地时区，否则前端显示早 8 小时。

    naive 时间按 UTC 处理（asyncpg 对 timestamptz 返回带 utc 的 datetime，
    但防御性处理无 tzinfo 的情况）。
    """
    import datetime as dt

    utc = dt.datetime(2026, 11, 17, 10, 5, tzinfo=dt.timezone.utc)
    naive = dt.datetime(2026, 11, 17, 10, 5)
    assert long_memory._fmt_local(utc) == long_memory._fmt_local(naive)
    assert long_memory._fmt_local(None) is None


def test_memory_recall_prompt_and_normalize() -> None:
    """键归一化与画像注入文本。"""
    assert long_memory.normalize_key(" Food Preference ") == "food_preference"
    prompt = long_memory.recall_prompt(
        [{"key": "dietary", "value": "不吃香菜", "updated_at": "2026-11-05 12:20"}]
    )
    assert "不吃香菜" in prompt
    assert "长期画像" in prompt
    assert long_memory.recall_prompt([]) == ""
