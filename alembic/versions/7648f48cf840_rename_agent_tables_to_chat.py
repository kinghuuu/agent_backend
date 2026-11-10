"""rename agent tables to chat

Revision ID: 7648f48cf840
Revises: 49cf9bbd74e5
Create Date: 2026-11-10 10:15:33.580288

模块更名：智能体（agent）→ 会话（chat）。两张表按新名字重命名，数据原样保留；
PostgreSQL 重命名表不会连带改索引 / 约束 / 序列的名字，这里一并改掉，避免名不副实。
"""
from typing import Sequence, Union

from alembic import op

revision: str = '7648f48cf840'
down_revision: Union[str, None] = '49cf9bbd74e5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.rename_table('agent_conversation', 'chat_conversation')
    op.rename_table('agent_message', 'chat_message')

    op.execute(
        'ALTER INDEX ix_agent_conversation_model_config_id '
        'RENAME TO ix_chat_conversation_model_config_id'
    )
    op.execute(
        'ALTER INDEX ix_agent_message_conversation_id '
        'RENAME TO ix_chat_message_conversation_id'
    )
    op.execute(
        'ALTER TABLE chat_conversation RENAME CONSTRAINT '
        'agent_conversation_pkey TO chat_conversation_pkey'
    )
    op.execute(
        'ALTER TABLE chat_message RENAME CONSTRAINT '
        'agent_message_pkey TO chat_message_pkey'
    )
    op.execute(
        'ALTER TABLE chat_message RENAME CONSTRAINT '
        'agent_message_conversation_id_fkey TO chat_message_conversation_id_fkey'
    )
    op.execute(
        'ALTER SEQUENCE agent_conversation_id_seq RENAME TO chat_conversation_id_seq'
    )
    op.execute('ALTER SEQUENCE agent_message_id_seq RENAME TO chat_message_id_seq')


def downgrade() -> None:
    op.execute('ALTER SEQUENCE chat_message_id_seq RENAME TO agent_message_id_seq')
    op.execute(
        'ALTER SEQUENCE chat_conversation_id_seq RENAME TO agent_conversation_id_seq'
    )
    op.execute(
        'ALTER TABLE chat_message RENAME CONSTRAINT '
        'chat_message_conversation_id_fkey TO agent_message_conversation_id_fkey'
    )
    op.execute(
        'ALTER TABLE chat_message RENAME CONSTRAINT '
        'chat_message_pkey TO agent_message_pkey'
    )
    op.execute(
        'ALTER TABLE chat_conversation RENAME CONSTRAINT '
        'chat_conversation_pkey TO agent_conversation_pkey'
    )
    op.execute(
        'ALTER INDEX ix_chat_message_conversation_id '
        'RENAME TO ix_agent_message_conversation_id'
    )
    op.execute(
        'ALTER INDEX ix_chat_conversation_model_config_id '
        'RENAME TO ix_agent_conversation_model_config_id'
    )

    op.rename_table('chat_message', 'agent_message')
    op.rename_table('chat_conversation', 'agent_conversation')
