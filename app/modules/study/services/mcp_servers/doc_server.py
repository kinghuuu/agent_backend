"""Day 14 代码 01：最小 MCP Server（stdio 传输）。

独立运行：``python doc_server.py``（它会作为子进程被客户端拉起，自己不需要常驻）。

对照 Day 03 的进程内写法，迁移只换了装饰器：
``@tool`` → ``@mcp.tool()``，docstring 依旧是模型看到的唯一说明书。

铁律：**stdout 是协议通道**，这个文件里不允许出现任何 ``print``。
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("doc-tools")

# Server 进程私有的状态（stdio 模式下随子进程生灭；要持久化就在 Server 里接数据库）
DOCS: dict[str, str] = {
    "doc1": "LangGraph 是基于状态图的多 Agent 编排框架，State / Node / Edge 三件套描述控制流。",
    "doc2": "LangGraph 的 Checkpointer 支持断点续跑与人机协同，一超步存一帧快照。",
    "doc3": "MCP 协议统一了工具接入标准：Host 通过 Client 连接 Server，工具清单用 tools/list 发现。",
    "doc4": "SSE（Server-Sent Events）是单向的服务器推送通道，帧以两个换行结尾。",
}


@mcp.tool()
def search_docs(query: str) -> str:
    """在文档库中检索相关内容。

    输入空格分隔的关键词，返回命中段落；一条都没命中时返回「未找到相关文档」，
    而不是抛异常——错误信息即 prompt，模型会自己改关键词重试。

    Args:
        query: 空格分隔的检索关键词，例如 "Checkpointer 断点续跑"

    Returns:
        命中的文档段落，多段之间用换行分隔。
    """
    keywords = [word for word in query.replace("，", " ").split() if word]
    hits = [doc_id for doc_id, text in DOCS.items() if any(word in text for word in keywords)]
    if not hits:
        return "未找到相关文档"
    return "\n".join(DOCS[doc_id] for doc_id in hits)


if __name__ == "__main__":
    # stdio：用标准输入输出当通信管道，进程不打印任何多余内容到 stdout
    mcp.run(transport="stdio")
