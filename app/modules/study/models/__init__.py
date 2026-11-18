"""学习模块的 ORM 模型（Day 07 起有表：长期记忆 + RAG 知识块）。

Alembic 通过 ``app/modules/__init__.py`` 间接导入本包，保证表都能被发现。
"""

from app.modules.study.models.knowledge import StudyKnowledgeChunk
from app.modules.study.models.long_memory import StudyUserMemory

__all__ = ["StudyKnowledgeChunk", "StudyUserMemory"]
