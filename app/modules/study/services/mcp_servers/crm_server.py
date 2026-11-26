"""Day 14 代码 02：更像样的 MCP Server——多工具 + Resource + Prompt。

- ``get_customer``  ：读操作；
- ``deduct_balance``：**写操作**，Server 端自己做参数校验与余额前置检查
  （Day 04「副作用分级 + 确认门」原则在 Server 端同样适用，跨进程后调用方更不可信）；
- ``crm://stats``   ：Resource，宿主程序直接读，不占模型注意力；
- ``refund_flow``   ：Prompt，用户选择触发的预置模板。

独立运行：``python crm_server.py``。日志只准写 stderr，stdout 留给 JSON-RPC。
"""

import logging
import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

# 日志走 stderr（logging 默认就是 stderr；这里显式指定，避免有人在 Server 里 print）
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
logger = logging.getLogger("crm-server")

mcp = FastMCP("crm-tools")

# Server 进程私有的状态：stdio 模式下每个客户端会话各自一份（演示可复现）
CUSTOMERS: dict[str, dict[str, Any]] = {
    "C001": {"name": "张三", "vip": True, "balance": 1200.0},
    "C002": {"name": "李四", "vip": False, "balance": 80.5},
}

# 审计流水：写操作留痕，证明"副作用发生在 Server 进程里"
AUDIT: list[dict[str, Any]] = []


@mcp.tool()
def get_customer(customer_id: str) -> str:
    """按客户 ID 查询客户信息（姓名 / 等级 / 余额）。

    Args:
        customer_id: 客户编号，例如 "C001"；不存在时返回可读提示而不是抛异常

    Returns:
        一行文本：姓名 / VIP / 余额。
    """
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return f"客户 {customer_id} 不存在"
    return f"姓名={customer['name']} VIP={customer['vip']} 余额={customer['balance']}元"


@mcp.tool()
def deduct_balance(customer_id: str, amount: float) -> str:
    """从客户余额中扣款。写操作，每次调用都会记录审计日志。

    Args:
        customer_id: 客户编号，例如 "C001"
        amount: 扣款金额（元），必须为正数

    Returns:
        成功返回剩余余额；金额非法或余额不足时返回失败原因（错误信息即 prompt）。
    """
    customer = CUSTOMERS.get(customer_id)
    if not customer:
        return f"客户 {customer_id} 不存在"
    if amount <= 0:
        return "扣款金额必须为正数"
    if customer["balance"] < amount:
        return f"余额不足：当前 {customer['balance']} 元，需 {amount} 元"
    customer["balance"] -= amount
    AUDIT.append({"customer_id": customer_id, "amount": amount, "balance": customer["balance"]})
    logger.info("扣款 %s 元，剩余 %s 元", amount, customer["balance"])
    return f"扣款成功，{customer['name']} 剩余 {customer['balance']} 元"


@mcp.resource("crm://stats")
def crm_stats() -> str:
    """CRM 当前统计快照（宿主程序可直接读，不经过模型决策）。"""
    total = sum(customer["balance"] for customer in CUSTOMERS.values())
    return f"客户数={len(CUSTOMERS)} 总余额={total}元 审计流水={len(AUDIT)}条"


@mcp.prompt()
def refund_flow(customer_id: str = "C001", amount: str = "100") -> str:
    """退款/扣款的标准处理话术模板（用户选择触发，不由模型自行决定）。

    Args:
        customer_id: 客户编号
        amount: 金额（元）
    """
    return (
        f"你正在处理客户 {customer_id} 的 {amount} 元扣款请求。"
        "流程：① 先调用 get_customer 核对余额；② 再调用 deduct_balance 扣款；"
        "③ 失败时把失败原因原样告知用户，不要编造成功。"
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
