"""RAG 知识库（Day 07 文档类管道）的 ORM 模型。

一份文档被切成多个 chunk 落库；向量以 JSON 数组存放，检索时在 Python 侧算
余弦相似度。学习项目零额外依赖；真实项目换 pgvector / Qdrant，服务层接口不变。
"""

from sqlalchemy import JSON, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin


class StudyKnowledgeChunk(Base, TimestampMixin):
    """知识文档的一个切块（含正文、元数据与向量）。

    元数据（doc_title / doc_version / section）是切块时就要埋进去的：
    回答时标注来源、检索时按 doc_id 过滤、冲突时按版本裁决，都靠它。
    """

    __tablename__ = "study_knowledge_chunk"
    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index", name="uq_study_knowledge_chunk_doc_index"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键")
    doc_id: Mapped[str] = mapped_column(
        String(100), index=True, comment="文档 ID（同 doc_id 的块属同一文档）"
    )
    doc_title: Mapped[str] = mapped_column(String(200), comment="文档标题（来源标注用）")
    doc_version: Mapped[str | None] = mapped_column(
        String(50), default=None, comment="版本/时效，如 2026-10 版"
    )
    chunk_index: Mapped[int] = mapped_column(Integer, comment="块序号（从 0 开始）")
    section: Mapped[str | None] = mapped_column(
        String(200), default=None, comment="所属章节标题"
    )
    content: Mapped[str] = mapped_column(Text, comment="块正文")
    embedding: Mapped[list] = mapped_column(JSON, comment="向量（float 数组）")
    source: Mapped[str | None] = mapped_column(
        String(30), default=None, comment="向量来源: api / hash-fallback"
    )
