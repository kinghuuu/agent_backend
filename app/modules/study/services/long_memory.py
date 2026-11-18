"""长期记忆（Day 07 画像类管道）。

三条原则（课程 1.2）：
1. 写事实不写对话——存 ``{"dietary": "不吃香菜"}``，不存整段聊天记录；
2. 同一 key 覆盖而非追加——记忆跟着用户最新说法走，否则新旧并立自相矛盾；
3. 会话开始时全文注入——数据量小（几十条），注入比检索可靠、不会漏。

冲突检测（练习 1）：不同叫法指向同一事实时（dietary / food_preference），
值不一致就提示合并，别等模型随机挑一条来用。
"""

import datetime as dt
import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.crud import long_memory as crud
from app.modules.study.models import StudyUserMemory

# 同义键组：组内出现多条不同值的键，视为"记忆可能自相矛盾"
SYNONYM_GROUPS: list[set[str]] = [
    {"dietary", "diet", "food", "food_preference", "饮食", "忌口"},
    {"name", "nickname", "昵称", "称呼"},
    {"theme", "editor_theme", "主题"},
    {"membership", "vip", "会员"},
]


def normalize_key(key: str) -> str:
    """键归一化：去首尾空白、转小写、空格/连字符转下划线。"""
    return re.sub(r"[\s\-]+", "_", key.strip().lower())


def _group_of(key: str) -> str:
    """把键映射到同义组标识；不在任何组里就自成一派。"""
    for group in SYNONYM_GROUPS:
        if key in group:
            return min(group)  # 组内取字典序最小者作组名，保证结果稳定
    return key


def _fmt_local(value: dt.datetime | None) -> str | None:
    """数据库里的 timestamptz 是 UTC，转成服务器本地时区再展示。

    直接 strftime 会显示成 UTC（比北京时间早 8 小时），用户会以为时间不对。
    这里统一 ``astimezone()`` 转本地；若部署在 Docker 且容器时区是 UTC，
    需要给容器加 ``TZ=Asia/Shanghai`` 环境变量。
    """
    if not isinstance(value, dt.datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone().strftime("%Y-%m-%d %H:%M")


def _fact_out(obj: StudyUserMemory) -> dict[str, Any]:
    """ORM → 前端友好的 dict（时间统一成本地时区字符串）。"""
    return {
        "key": obj.mem_key,
        "value": obj.value,
        "source_turn": obj.source_turn,
        "updated_at": _fmt_local(obj.updated_at),
    }


async def remember(
    db: AsyncSession,
    user_id: str,
    key: str,
    value: str,
    source_turn: int | None = None,
) -> dict[str, Any]:
    """写入或覆盖一条事实，返回 ``{created, fact}``。

    同 key 已存在时覆盖 value（而不是新增一条）——这是长期记忆最容易写错的地方。
    """
    mem_key = normalize_key(key)
    if not mem_key:
        raise ValueError("事实键不能为空")
    existing = await crud.get_one(db, user_id, mem_key)
    if existing is None:
        obj = await crud.create(db, user_id, mem_key, value, source_turn)
        return {"created": True, "fact": _fact_out(obj)}
    obj = await crud.update_value(db, existing, value, source_turn)
    return {"created": False, "fact": _fact_out(obj)}


async def forget(db: AsyncSession, user_id: str, key: str) -> bool:
    """删除一条事实（课程练习 1 的 forget）。"""
    return await crud.delete_one(db, user_id, normalize_key(key))


async def clear_user(db: AsyncSession, user_id: str) -> int:
    """清空某用户的全部长期记忆，返回删除条数。"""
    return await crud.delete_user(db, user_id)


async def list_facts(db: AsyncSession, user_id: str) -> list[dict[str, Any]]:
    """列出某用户的全部事实。"""
    return [_fact_out(obj) for obj in await crud.list_by_user(db, user_id)]


def fact_line(fact: dict[str, Any]) -> str:
    """单条事实的注入格式。

    抽成函数是为了让 Day 08 的画像预算裁剪（按条装填）产出与 ``recall_prompt``
    完全一致的文本——同一份画像，两处格式化不能长得不一样。
    """
    return f"- {fact['key']}: {fact['value']}（更新于 {fact.get('updated_at') or '未知'}）"


def recall_prompt(facts: list[dict[str, Any]]) -> str:
    """把画像拼成 system 注入文本；没有记忆时返回空串。

    数据量小时全文注入，比向量检索更可靠——不会有"召回失败"这种漏。
    """
    if not facts:
        return ""
    return "【用户长期画像】\n" + "\n".join(fact_line(item) for item in facts)


def detect_conflicts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """扫描同义键组，找出值不一致的键对（练习 1 的 conflicts）。

    只做"键归一化 + 同组比对"这一层廉价检测；真正的语义矛盾交给模型或人工确认。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in facts:
        grouped.setdefault(_group_of(item["key"]), []).append(item)

    conflicts: list[dict[str, Any]] = []
    for group, items in grouped.items():
        values = {item["value"] for item in items}
        if len(items) > 1 and len(values) > 1:
            conflicts.append(
                {
                    "group": group,
                    "keys": [item["key"] for item in items],
                    "values": [f"{item['key']}={item['value']}" for item in items],
                    "suggestion": f"同一事实有多种叫法且取值不同，建议只保留一条（如统一用 {group}）",
                }
            )
    return conflicts
