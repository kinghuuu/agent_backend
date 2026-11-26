"""Day 14 的三个 MCP Server 脚本（独立进程，不是 import 进宿主用的模块）。

为什么做成**真实脚本文件**而不是在 service 里拼字符串：

- stdio 传输的本质是「客户端 spawn 一个子进程，用 stdin/stdout 说 JSON-RPC」，
  脚本必须真的能被 ``python xxx_server.py`` 拉起，教学才成立；
- 用 ``sys.executable`` 拉起，保证子进程和宿主是同一个解释器（同一个 .venv），
  否则会出现"宿主管用、子进程缺依赖"这种 Day 13 遇到过的环境错位。

三个脚本：

- ``doc_server.py``    ：最小 Server，一个 ``search_docs`` 工具；
- ``crm_server.py``    ：多工具 + Resource + Prompt，含写操作校验；
- ``noisy_server.py``  ：故意 print 到 stdout 的反面教材（坑 1 现场）。

统一约定：日志只能写 stderr。stdout 是协议通道，任何 ``print`` 都会污染 JSON-RPC 帧。
"""
