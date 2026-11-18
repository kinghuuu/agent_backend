"""add study long memory and rag chunk tables

Revision ID: a1b2c3d4e5f6
Revises: 7648f48cf840
Create Date: 2026-11-17 10:00:00.000000

Day 07：学习模块首次落库。
- study_user_memory：长期记忆（画像类），(user_id, mem_key) 唯一，同 key 覆盖；
- study_knowledge_chunk：RAG 切块（文档类），向量以 JSON 数组存放，
  (doc_id, chunk_index) 唯一，便于整篇重建。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '7648f48cf840'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'study_user_memory',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键'),
        sa.Column('user_id', sa.String(length=100), nullable=False, comment='用户 ID（业务侧稳定标识）'),
        sa.Column('mem_key', sa.String(length=100), nullable=False, comment='事实键，如 dietary'),
        sa.Column('value', sa.Text(), nullable=False, comment='事实值，如 不吃香菜'),
        sa.Column('source_turn', sa.Integer(), nullable=True, comment='来源对话轮次（可追溯，不是给模型看的）'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'mem_key', name='uq_study_user_memory_user_key'),
    )
    op.create_index(op.f('ix_study_user_memory_user_id'), 'study_user_memory', ['user_id'], unique=False)

    op.create_table(
        'study_knowledge_chunk',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键'),
        sa.Column('doc_id', sa.String(length=100), nullable=False, comment='文档 ID（同 doc_id 的块属同一文档）'),
        sa.Column('doc_title', sa.String(length=200), nullable=False, comment='文档标题（来源标注用）'),
        sa.Column('doc_version', sa.String(length=50), nullable=True, comment='版本/时效，如 2026-10 版'),
        sa.Column('chunk_index', sa.Integer(), nullable=False, comment='块序号（从 0 开始）'),
        sa.Column('section', sa.String(length=200), nullable=True, comment='所属章节标题'),
        sa.Column('content', sa.Text(), nullable=False, comment='块正文'),
        sa.Column('embedding', sa.JSON(), nullable=False, comment='向量（float 数组）'),
        sa.Column('source', sa.String(length=30), nullable=True, comment='向量来源: api / hash-fallback'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('doc_id', 'chunk_index', name='uq_study_knowledge_chunk_doc_index'),
    )
    op.create_index(op.f('ix_study_knowledge_chunk_doc_id'), 'study_knowledge_chunk', ['doc_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_study_knowledge_chunk_doc_id'), table_name='study_knowledge_chunk')
    op.drop_table('study_knowledge_chunk')
    op.drop_index(op.f('ix_study_user_memory_user_id'), table_name='study_user_memory')
    op.drop_table('study_user_memory')
