"""Day 15 全链路追踪用的 MCP Server：工具主动接收 trace_id，并记下自己的 span。

为什么要在工具参数里加一个 ``trace_id``？
LangGraph 侧有平台的 trace，MCP Server 是另一个进程，两边默认对不上账。
客户端把 trace_id 当普通参数传过来，Server 就能按同一个 id 写自己的 JSONL——
排查时用一行 grep 把「模型说了什么、调了什么」和「工具实际执行了多久、有没有错」
拼成完整链路。这是跨进程 / 跨团队系统的可观测性基本功。

日志路径由宿主通过环境变量 ``DAY15_SERVER_SPANS`` 下发（单测可以指向临时文件），
没传时回落到 ``agent_backend/var/day15/server_spans.jsonl``。

独立运行：``python expense_server.py``。日志只准写 stderr，stdout 留给 JSON-RPC。
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# 日志走 stderr（Day 14 的教训：stdio 传输里 stdout 是协议通道）
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(levelname)s %(message)s")
logger = logging.getLogger("expense-server")

mcp = FastMCP("expense-tools")

# 记录文件：宿主优先，默认落在 agent_backend/var/day15/
DEFAULT_LOG = Path(__file__).resolve().parents[5] / "var" / "day15" / "server_spans.jsonl"
SPAN_LOG = Path(os.environ.get("DAY15_SERVER_SPANS") or DEFAULT_LOG)

# 制度库（与进程内版本同源，方便对照）
POLICIES: dict[str, str] = {
    "餐饮": "差旅餐饮每日报销上限 100 元，需发票。",
    "交通": "高铁二等座全额报销，打车需注明事由。",
    "住宿": "一线城市住宿每晚上限 600 元，需水单。",
    "退款": "商品退款 3 个工作日内到账。",
}

# 单号计数器：进程内递增（stdio 下每个会话一个进程，重启会回到 2045）
_SERIAL = {"value": 2045}


def log_span(trace_id: str, tool: str, arguments: dict[str, Any], duration_ms: float, status: str) -> None:
    """把一次工具执行写成一行 JSONL：trace_id 是跨进程串联的唯一钥匙。"""
    row = {
        "trace_id": trace_id,
        "tool": tool,
        "args": arguments,
        "dur_ms": duration_ms,
        "status": status,
        "ts": time.time(),
        "source": "server",
    }
    try:
        SPAN_LOG.parent.mkdir(parents=True, exist_ok=True)
        with SPAN_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:  # 写日志失败不该让业务调用失败
        logger.warning("span 写入失败：%s", exc)


def _elapsed_ms(started: float) -> float:
    """耗时（毫秒，保留一位小数）。"""
    return round((time.perf_counter() - started) * 1000, 1)


@mcp.tool()
def search_expense_policy(trace_id: str, keyword: str) -> str:
    """查询公司报销制度（只读），并把执行记录到服务端 span。

    Args:
        trace_id: 追踪标识，由调用方原样传入（不要改写）
        keyword: 制度关键词，例如 餐饮 / 交通 / 住宿 / 退款

    Returns:
        对应条款文本；未命中时返回可读提示。
    """
    started = time.perf_counter()
    status = "ok"
    try:
        return POLICIES.get(keyword, f"未找到「{keyword}」相关条款。")
    except Exception as exc:  # noqa: BLE001  错误也要留痕，方便定位到具体 span
        status = f"error:{type(exc).__name__}"
        raise
    finally:
        log_span(trace_id, "search_expense_policy", {"keyword": keyword}, _elapsed_ms(started), status)


@mcp.tool()
def submit_expense(trace_id: str, amount: float, category: str, reason: str) -> str:
    """提交报销申请（写操作），Server 端自己校验参数并留痕。

    Args:
        trace_id: 追踪标识，由调用方原样传入（不要改写）
        amount: 金额（元），必须为正数
        category: 类别，例如 餐饮 / 交通 / 住宿
        reason: 事由

    Returns:
        成功返回单号与摘要；金额非法时返回失败原因（错误信息即 prompt）。
    """
    started = time.perf_counter()
    status = "ok"
    try:
        if amount <= 0:
            status = "error:invalid_amount"
            return f"报销金额必须为正数，收到 {amount}"
        _SERIAL["value"] += 1
        return f"报销单已创建：{category} {amount} 元，事由：{reason}。单号 EXP-{_SERIAL['value']}。"
    except Exception as exc:  # noqa: BLE001
        status = f"error:{type(exc).__name__}"
        raise
    finally:
        log_span(
            trace_id,
            "submit_expense",
            {"amount": amount, "category": category},
            _elapsed_ms(started),
            status,
        )


if __name__ == "__main__":
    mcp.run(transport="stdio")
