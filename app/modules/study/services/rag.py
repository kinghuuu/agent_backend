"""RAG（Day 07 文档类管道）：切块 → 向量化 → 混合检索 → 注入。

流水线对应课程 1.3：
- 离线：按标题/段落切块（保留 15% 重叠）→ Embedding → 存库；
- 在线：向量检索 + BM25 关键词检索 → RRF 按排名融合 → Top-K 带来源注入。

两个工程取舍（学习项目）：
1. 向量存 JSON、Python 算余弦，零向量库依赖；真实项目换 pgvector / Qdrant 即可；
2. Embedding 优先用「模型管理」里启用的 embedding 模型，调用失败回退本地字符
   n-gram 哈希——保证没配模型时也能跑通整条链路，而不是直接 500。
"""

import hashlib
import math
import re
from collections import Counter
from typing import Any, Iterable

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.modules.model_config.services import model_config as model_config_service
from app.modules.study.crud import knowledge as crud
from app.utils.llm import auth_headers, embeddings_url

logger = get_logger(__name__)

# 切块默认参数（课程 2.4：中文 300~800 字，重叠约 15%）
DEFAULT_CHUNK_SIZE = 400
DEFAULT_OVERLAP = 60
# 哈希模拟向量的维度（与真实 embedding 模型维度无关，仅用于回退）
HASH_DIM = 512
# RRF 的平滑常数，课程与业界常用 60
RRF_K = 60


# ============================================================
# 1. 切块：按结构分节 + 段落聚合 + 重叠
# ============================================================


def chunk_document(
    text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[dict[str, str | None]]:
    """把 Markdown/纯文本切成块，返回 ``[{section, content}]``。

    为什么不用固定长度硬切：硬切会把「30 天可退」和它的例外条款分进两个块，
    检索到一半等于答案残缺。先按标题分节（结构边界≈语义边界），节内按段落
    聚合到目标大小，相邻块保留一段重叠，防止关键句跨边界被拦腰斩断。
    """
    if not text.strip():
        return []

    # 先拆成 (所属章节, 段落) 序列；标题行本身不进入正文
    blocks: list[tuple[str | None, str]] = []
    section: str | None = None
    for line in text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.+)$", line.strip())
        if heading:
            section = heading.group(1).strip()
            continue
        line = line.strip()
        if line:
            blocks.append((section, line))

    chunks: list[dict[str, str | None]] = []
    buf = ""
    buf_section: str | None = None
    for sec, para in blocks:
        # 章节变了：结构边界天然是语义边界，直接切开，不做跨节重叠
        # （跨节重叠会把上一节的条款带进下一节，反而污染语义）
        if buf and sec != buf_section:
            chunks.append({"section": buf_section, "content": buf})
            buf, buf_section = para, sec
            continue
        # 同一章节内超长才按大小切，并保留重叠防止关键句被拦腰斩断
        if buf and len(buf) + len(para) + 1 > size:
            chunks.append({"section": buf_section, "content": buf})
            tail = buf[-overlap:] if overlap > 0 else ""
            buf = f"{tail}\n{para}".strip() if tail else para
            buf_section = sec
        else:
            buf = f"{buf}\n{para}".strip() if buf else para
            buf_section = buf_section or sec
    if buf:
        chunks.append({"section": buf_section, "content": buf})

    return chunks


# ============================================================
# 2. Embedding：优先真实模型，失败回退哈希模拟
# ============================================================


def hash_embed(text: str, dim: int = HASH_DIM) -> list[float]:
    """字符 3-gram 哈希计数向量（L2 归一化后点积即余弦），仅作离线回退。"""
    vec = [0.0] * dim
    for i in range(max(0, len(text) - 2)):
        digest = hashlib.md5(text[i : i + 3].encode("utf-8")).hexdigest()
        vec[int(digest, 16) % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: Iterable[float], b: Iterable[float]) -> float:
    """余弦相似度（两边都已 L2 归一化时等价于点积）。"""
    return sum(x * y for x, y in zip(a, b))


async def _resolve_embedding_cfg(db: AsyncSession):
    """取「模型管理」里启用的 embedding 模型：默认优先，其次 id 最小。"""
    items = await model_config_service.list_model_configs(db, "embedding")
    enabled = [item for item in items if item.enabled]
    if not enabled:
        return None
    enabled.sort(key=lambda item: (not item.is_default, item.id))
    return enabled[0]


async def _api_embed(cfg, texts: list[str]) -> list[list[float]]:
    """调用 OpenAI 兼容的 ``/embeddings``，按 index 还原顺序。"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            embeddings_url(cfg.base_url),
            json={"model": cfg.model_name, "input": texts, "encoding_format": "float"},
            headers=auth_headers(cfg.api_key),
        )
        resp.raise_for_status()
        payload = resp.json()
    items = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in items]


async def embed_texts(db: AsyncSession, texts: list[str]) -> tuple[list[list[float]], str]:
    """批量向量化，返回 ``(向量列表, 来源)``；来源用于检索时对齐维度。"""
    if not texts:
        return [], "hash-fallback"
    cfg = await _resolve_embedding_cfg(db)
    if cfg is not None:
        try:
            return await _api_embed(cfg, texts), "api"
        except Exception as exc:  # noqa: BLE001  外部服务不可用不该拖垮学习页
            logger.warning("Embedding 模型调用失败，回退本地哈希模拟：%s", exc)
    return [hash_embed(text) for text in texts], "hash-fallback"


# ============================================================
# 3. BM25 关键词检索
# ============================================================


def tokenize(text: str) -> list[str]:
    """英文按词、中文按字切分——中文分词器是另一个话题，这里保持零依赖。"""
    return re.findall(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]", text.lower())


class BM25:
    """经典 BM25（k1=1.5, b=0.75），专治向量检索对专有名词不敏感。"""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs_tokens = [tokenize(doc) for doc in docs]
        self.avgdl = sum(len(t) for t in self.docs_tokens) / (len(docs) or 1)
        self.df: Counter[str] = Counter(
            word for tokens in self.docs_tokens for word in set(tokens)
        )

    def scores(self, query: str) -> list[float]:
        """返回每个文档对 query 的 BM25 得分。"""
        q_tokens = tokenize(query)
        n = len(self.docs_tokens)
        result: list[float] = []
        for tokens in self.docs_tokens:
            dl = len(tokens) or 1
            score = 0.0
            for word in q_tokens:
                df = self.df.get(word, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                tf = tokens.count(word)
                score += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            result.append(score)
        return result


# ============================================================
# 4. 混合检索 + RRF
# ============================================================


def _rank_order(scores: list[float]) -> list[int]:
    """按得分从高到低返回下标（稳定排序）。"""
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def rrf_fuse(
    vec_ranks: list[int], bm25_ranks: list[int], alpha: float = 0.5
) -> dict[int, float]:
    """RRF：只用排名不用原始分数，避免两路量纲不同无法相加。"""
    fused: dict[int, float] = {}
    for rank, index in enumerate(vec_ranks):
        fused[index] = fused.get(index, 0.0) + alpha / (RRF_K + rank + 1)
    for rank, index in enumerate(bm25_ranks):
        fused[index] = fused.get(index, 0.0) + (1 - alpha) / (RRF_K + rank + 1)
    return fused


async def index_document(
    db: AsyncSession,
    doc_id: str,
    title: str,
    content: str,
    version: str | None = None,
    size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> dict[str, Any]:
    """重建一篇文档的索引：先删旧块，再切块、向量化、落库。

    用「删了重建」而不是增量更新：文档级别整体替换最简单，也不会留下旧版本残块。
    """
    chunks = chunk_document(content, size, overlap)
    if not chunks:
        raise ValueError("文档内容为空，无法建立索引")

    vectors, source = await embed_texts(db, [chunk["content"] for chunk in chunks])
    await crud.delete_doc(db, doc_id)
    rows = [
        {
            "doc_id": doc_id,
            "doc_title": title,
            "doc_version": version,
            "chunk_index": index,
            "section": chunk["section"],
            "content": chunk["content"],
            "embedding": vector,
            "source": source,
        }
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    count = await crud.insert_chunks(db, rows)
    return {"doc_id": doc_id, "chunks": count, "embedding_source": source}


async def search(
    db: AsyncSession,
    query: str,
    top_k: int = 3,
    alpha: float = 0.5,
    where: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """混合检索：向量 + BM25 两路排名，RRF 融合后取 Top-K。

    ``where`` 是元数据过滤（如 ``{"doc_id": "policy_2026"}``），先过滤再算分，
    避免「问退款却召回规格手册」。
    """
    where = where or {}
    doc_id = where.get("doc_id")
    chunks = await crud.list_chunks(db, doc_id)
    if not chunks:
        return []
    if "doc_version" in where:
        chunks = [c for c in chunks if c.doc_version == where["doc_version"]]
    if not chunks:
        return []

    texts = [chunk.content for chunk in chunks]
    bm25_scores = BM25(texts).scores(query)

    # 向量路：分来源计算，维度对不上就跳过（防止 API 与哈希向量混算）
    qvec_hash = hash_embed(query)
    api_cfg = await _resolve_embedding_cfg(db)
    qvec_api: list[float] | None = None
    if api_cfg is not None and any(chunk.source == "api" for chunk in chunks):
        try:
            qvec_api = (await _api_embed(api_cfg, [query]))[0]
        except Exception as exc:  # noqa: BLE001
            logger.warning("查询向量化失败，仅用哈希块参与向量检索：%s", exc)
            qvec_api = None

    vec_scores: list[float] = []
    for chunk in chunks:
        if chunk.source == "api":
            score = (
                cosine(qvec_api, chunk.embedding)
                if qvec_api is not None and len(qvec_api) == len(chunk.embedding)
                else 0.0
            )
        else:
            score = (
                cosine(qvec_hash, chunk.embedding)
                if len(qvec_hash) == len(chunk.embedding)
                else 0.0
            )
        vec_scores.append(score)

    vec_ranks = _rank_order(vec_scores)
    bm25_ranks = _rank_order(bm25_scores)
    fused = rrf_fuse(vec_ranks, bm25_ranks, alpha)

    top_indices = sorted(fused, key=lambda i: -fused[i])[:top_k]
    hits: list[dict[str, Any]] = []
    for index in top_indices:
        chunk = chunks[index]
        hits.append(
            {
                "doc_id": chunk.doc_id,
                "doc_title": chunk.doc_title,
                "doc_version": chunk.doc_version,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "content": chunk.content,
                "rrf": round(fused[index], 6),
                "vec_score": round(vec_scores[index], 4),
                "bm25_score": round(bm25_scores[index], 4),
                "vec_rank": vec_ranks.index(index) + 1,
                "bm25_rank": bm25_ranks.index(index) + 1,
                "embedding_source": chunk.source,
            }
        )
    return hits


# ============================================================
# 5. 注入与冲突检测
# ============================================================


def to_context(hits: list[dict[str, Any]]) -> str:
    """检索结果 → 带来源标注的注入文本（逼模型有据可依）。"""
    if not hits:
        return ""
    blocks = []
    for hit in hits:
        version = f"（{hit['doc_version']}）" if hit.get("doc_version") else ""
        section = f"§{hit['section']}" if hit.get("section") else ""
        blocks.append(
            f"[来源: {hit['doc_title']}{version}{section} · 块{hit['chunk_index']}] {hit['content']}"
        )
    return "【检索到的资料】\n" + "\n---\n".join(blocks)


_MONEY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*元")


def detect_numeric_conflicts(question: str, hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """廉价冲突检测：用户问题里的金额与检索资料里的金额不一致时摆上台面。

    只覆盖「数字型事实」这一种高频场景；完整裁决仍靠 prompt 规则 + 模型。
    """
    asked = {float(value) for value in _MONEY_RE.findall(question)}
    if not asked:
        return []
    doc_values: set[float] = set()
    for hit in hits:
        doc_values |= {float(value) for value in _MONEY_RE.findall(hit["content"])}
    if not doc_values or asked == doc_values:
        return []
    return [
        {
            "type": "金额不一致",
            "user_values": sorted(asked),
            "doc_values": sorted(doc_values),
            "hint": "不要缝合两边各取一半：请摆出双方说法，并建议以订单页实时金额为准",
        }
    ]
