"""知识库切块表的纯数据库读写。"""

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.models import StudyKnowledgeChunk


async def list_chunks(
    db: AsyncSession, doc_id: str | None = None
) -> list[StudyKnowledgeChunk]:
    """取全部块；传 doc_id 时只取该文档（用于重建索引或调试）。"""
    stmt = select(StudyKnowledgeChunk).order_by(
        StudyKnowledgeChunk.doc_id, StudyKnowledgeChunk.chunk_index
    )
    if doc_id:
        stmt = stmt.where(StudyKnowledgeChunk.doc_id == doc_id)
    return list((await db.scalars(stmt)).all())


async def insert_chunks(db: AsyncSession, rows: list[dict[str, Any]]) -> int:
    """批量插入切块（调用方负责先删旧版），返回插入条数。"""
    objs = [StudyKnowledgeChunk(**row) for row in rows]
    db.add_all(objs)
    await db.commit()
    return len(objs)


async def delete_doc(db: AsyncSession, doc_id: str) -> int:
    """按 doc_id 删除整篇文档的全部块，返回删除条数。"""
    result = await db.execute(
        delete(StudyKnowledgeChunk).where(StudyKnowledgeChunk.doc_id == doc_id)
    )
    await db.commit()
    return int(result.rowcount or 0)


async def list_doc_summaries(db: AsyncSession) -> list[dict[str, Any]]:
    """按文档聚合成列表：标题、版本、块数、最近更新时间。"""
    stmt = (
        select(
            StudyKnowledgeChunk.doc_id,
            StudyKnowledgeChunk.doc_title,
            StudyKnowledgeChunk.doc_version,
            func.count(StudyKnowledgeChunk.id).label("chunks"),
            func.max(StudyKnowledgeChunk.updated_at).label("updated_at"),
        )
        .group_by(
            StudyKnowledgeChunk.doc_id,
            StudyKnowledgeChunk.doc_title,
            StudyKnowledgeChunk.doc_version,
        )
        .order_by(StudyKnowledgeChunk.doc_id)
    )
    rows = (await db.execute(stmt)).all()
    return [
        {
            "doc_id": row.doc_id,
            "doc_title": row.doc_title,
            "doc_version": row.doc_version,
            "chunks": int(row.chunks),
            "updated_at": row.updated_at,
        }
        for row in rows
    ]
