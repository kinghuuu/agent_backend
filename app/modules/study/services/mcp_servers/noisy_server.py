"""Day 14 坑 1 的现场：stdio Server 里往 stdout 打日志会毒死协议流。

这个文件除了多两行 ``print``，和 ``doc_server.py`` 完全一样。
标准输入输出在 stdio 传输里就是**协议通道**（一行一个 JSON-RPC 帧），
任何纯文本混进去都会让客户端解析失败或行为异常。

三种污染形态用环境变量 ``DAY14_NOISY_MODE`` 切换，效果不同（实测见接口返回）：

- ``print``  ：``print(...)`` 整行垃圾——客户端**可能**把这一行当垃圾跳过，
  也可能报 ``Invalid JSON``，取决于它到达的时刻（时好时坏，最容易被误判成"没事"）；
- ``fd_line``：``os.write(1, "...\\n")`` 整行垃圾，同上；
- ``fd``     ：``os.write(1, "...")`` 不带换行——半行文本和下一帧 JSON 粘在一起，
  帧边界破裂，客户端**必然**挂死（默认用这个，因为它最稳定地演示事故）。

修复方式（课程第 4 节）：日志一律写 stderr——``logging`` 默认就是 stderr，
或 ``print(..., file=sys.stderr)``；严禁第三方库/调试残留往 stdout 输出。
"""

import os
import sys

from mcp.server.fastmcp import FastMCP

MODE = os.getenv("DAY14_NOISY_MODE", "fd")


def pollute(text: str) -> None:
    """污染协议通道：往 stdout 写纯文本（三种形态由 DAY14_NOISY_MODE 决定）。

    - ``print``  ：写 ``sys.stdout``（带换行，走文本模式的缓冲）；
    - ``fd_line``：写 fd 1、带换行（C 扩展 / 子进程输出走的就是这条路）；
    - ``fd``     ：写 fd 1、不带换行——半行文本会和下一帧 JSON 粘在同一行。
    """
    if MODE == "print":
        print(text, flush=True)
    elif MODE == "fd_line":
        os.write(1, (text + "\n").encode())
    else:
        os.write(1, text.encode())  # 不带换行：和下一帧 JSON 粘在一起


pollute("[noisy] server booting")

mcp = FastMCP("noisy-tools")


@mcp.tool()
def search_docs(query: str) -> str:
    """在文档库中检索相关内容（这个 Server 故意污染 stdout）。

    Args:
        query: 检索关键词
    """
    pollute(f"[noisy] searching {query}")
    return f"（如果客户端能收到这行，说明 stdout 没被污染）query={query}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
