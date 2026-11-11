"""智能体开发学习模块：把 Day 01 的练习代码工程化，供前端学习页可视化测试。

模块内容（无表结构，只有 schemas / services / views 三层）：
    services/tools.py       工具定义与 JSON Schema 说明书（Agent = LLM + 工具 + 循环 + 状态 中的「工具」）
    services/llm.py         从「模型管理」读取默认 LLM，构造 OpenAI SDK 异步客户端
    services/agent_loop.py  Function Calling 版最小 Agent 循环（带执行轨迹）
    services/react_loop.py  不用 Function Calling 的手写 ReAct 文本协议循环（带执行轨迹）
    views/study.py          三个接口：工具清单 / 运行 Agent / 运行 ReAct
"""
