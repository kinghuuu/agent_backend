"""Day 09 单测：LangChain 零件（消息 / 模板 / 工具 schema / LCEL / 重试），不联网。

运行：``python -m pytest tests/test_day09_langchain.py -p no:cacheprovider -q``
没装 langchain 时整文件跳过，不影响其它测试。
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_openai")

from app.modules.study.schemas.study import (  # noqa: E402
    Day09CustomerReply,
    Day09PromptRequest,
)
from app.modules.study.services import day09_langchain as day09  # noqa: E402


# ---------- 消息抽象 ----------


def test_messages_demo_types_and_roundtrip() -> None:
    """四类消息的 type 与 OpenAI role 一一对应，且能无损转回裸 dict。"""
    result = day09.messages_demo()
    assert [row["type"] for row in result["messages"]] == ["system", "human", "ai", "tool"]
    assert [row["openai_role"] for row in result["messages"]] == ["system", "user", "assistant", "tool"]
    payload = result["openai_payload"]
    assert payload[3] == {"role": "tool", "tool_call_id": "t_1", "content": "order_status: 已发货"}


def test_messages_demo_typeless_dict_passes_but_typed_fails() -> None:
    """反例：裸 dict 拼错 role 静默通过；正例：ToolMessage 缺 tool_call_id 构造即校验失败。"""
    result = day09.messages_demo()
    assert result["raw_typo"]["payload"]["role"] == "assitant"
    # ToolMessage 缺 tool_call_id：构造时就抛异常（不同版本是 KeyError 或 ValidationError）
    assert "tool_call_id" in result["typed_error"]
    assert ":" in result["typed_error"]
    # 补充事实：多传的未知 kwargs 被 pydantic 静默忽略，防拼错靠的是必填字段 + IDE 提示
    assert "extra=ignore" in result["extra_ignored"] or "接受了未知参数" in result["extra_ignored"]


# ---------- Prompt 模板 ----------


def _prompt_request(**extra) -> Day09PromptRequest:
    """构造模板演示请求。"""
    payload = {
        "company": "极星科技",
        "question": "XR-500 Pro 退款要运费吗？",
        "history": [
            {"role": "user", "content": "保修多久？"},
            {"role": "assistant", "content": "整机保修两年"},
        ],
    }
    payload.update(extra)
    return Day09PromptRequest(**payload)


def test_prompt_demo_fills_placeholder_history() -> None:
    """placeholder 把历史消息原样铺进 system 与当前问题之间（对应 Day 08 的第 4 层）。"""
    result = day09.prompt_demo(_prompt_request())
    assert result["input_variables"] == ["company", "question"]
    assert result["optional_variables"] == ["history"]
    assert result["partial_frozen"] == ["company"]
    assert [row["type"] for row in result["messages"]] == [
        "system",
        "human",
        "ai",
        "human",
    ]
    assert result["messages"][0]["content"].startswith("你是极星科技的客服 Agent")
    assert result["messages"][-1]["content"] == "XR-500 Pro 退款要运费吗？"
    assert result["error"] is None


def test_prompt_demo_missing_variable_fails_fast() -> None:
    """漏填必需变量（question）时模板在 invoke 阶段就 KeyError，而不是静默产出残缺 prompt。"""
    result = day09.prompt_demo(_prompt_request(omit_variable=True))
    assert result["error"] is not None
    assert "KeyError" in result["error"]
    assert "question" in result["error"]


def test_placeholder_variable_is_optional() -> None:
    """placeholder 的 history 缺失不报错：『没有历史』是合法状态（与课程原文的差异点）。"""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [("system", "你是{company}的客服"), ("placeholder", "{history}"), ("human", "{question}")]
    )
    filled = prompt.partial(company="极星科技").invoke({"question": "退款要运费吗"})
    assert [message.type for message in filled.to_messages()] == ["system", "human"]


# ---------- 工具 schema（@tool 自动生成） ----------


def test_build_agent_tools_generates_schema_from_docstring() -> None:
    """docstring 变成 description，Annotated+Field 变成参数说明——Day 04 的第一层被收编。"""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    tools = day09._build_agent_tools("full")
    assert [tool.name for tool in tools] == ["get_order_status", "get_return_policy"]

    spec = convert_to_openai_tool(tools[0])["function"]
    assert spec["name"] == "get_order_status"
    assert "何时用" in spec["description"]
    assert spec["parameters"]["properties"]["order_id"]["description"].startswith("订单号")


def test_build_agent_tools_without_docstring_raises() -> None:
    """variant=missing：框架在定义工具时就 NotImplemented 地拦住（fail-fast）。"""
    with pytest.raises(ValueError):
        day09._build_agent_tools("missing")


def test_tool_invoke_with_tool_call_dict() -> None:
    """框架替我们做的 dispatch：传 tool_call dict 就能执行，不用手写注册表。"""
    tools = {tool.name: tool for tool in day09._build_agent_tools("full")}
    call = {"name": "get_order_status", "args": {"order_id": "SO20261101001"}, "id": "t1", "type": "tool_call"}
    result = tools["get_order_status"].invoke(call)
    assert "已发货" in result.content
    assert result.tool_call_id == "t1"

    bad = {"name": "get_order_status", "args": {"order_id": "2026"}, "id": "t2", "type": "tool_call"}
    assert "必须以 SO 开头" in tools["get_order_status"].invoke(bad).content


# ---------- LCEL 与重试 ----------


def test_runnable_lambda_pipe_without_llm() -> None:
    """竖线只是 Runnable 的组合：不接模型也能串起来，普通函数要包 RunnableLambda。"""
    from langchain_core.runnables import RunnableLambda

    chain = RunnableLambda(lambda values: values["question"]) | RunnableLambda(
        lambda text: f"{text}｜已加工"
    )
    assert chain.invoke({"question": "什么是 RAG？"}) == "什么是 RAG？｜已加工"


def test_with_retry_retries_until_success() -> None:
    """with_retry 自动重试：业务函数只写正常路径（测试里关掉退避等待）。"""
    from langchain_core.runnables import RunnableLambda

    state = {"count": 0}

    def flaky(_: str) -> str:
        state["count"] += 1
        if state["count"] < 3:
            raise RuntimeError("模拟 502")
        return "第三次成功"

    safe = RunnableLambda(flaky).with_retry(stop_after_attempt=3, wait_exponential_jitter=False)
    assert safe.invoke("x") == "第三次成功"
    assert state["count"] == 3


def test_with_retry_gives_up_after_attempts() -> None:
    """重试耗尽后把最后一次异常抛出来——不吞错。"""
    from langchain_core.runnables import RunnableLambda

    def always_fail(_: str) -> str:
        raise RuntimeError("一直失败")

    safe = RunnableLambda(always_fail).with_retry(
        stop_after_attempt=2, wait_exponential_jitter=False
    )
    with pytest.raises(RuntimeError):
        safe.invoke("x")


# ---------- 结构化输出的合同 ----------


def test_customer_reply_schema_carries_descriptions() -> None:
    """Field(description=) 会进 JSON Schema：它就是下发给模型的说明书。"""
    schema = Day09CustomerReply.model_json_schema()
    assert set(schema["properties"]) == {"reply", "sentiment", "needs_escalation"}
    assert schema["properties"]["sentiment"]["description"] == "用户情绪"
    assert schema["properties"]["needs_escalation"]["type"] == "boolean"


def test_structured_runner_supports_three_methods() -> None:
    """with_structured_output 的三种下发方式都能构造 Runnable（真实降级在接口层验证）。"""
    cfg = SimpleNamespace(model_name="deepseek-chat", api_key="sk-test", base_url="https://api.deepseek.com")
    llm = day09.build_chat_model(cfg)
    for method in ("default", "function_calling", "json_mode"):
        runner = day09._structured_runner(llm, Day09CustomerReply, method)
        assert hasattr(runner, "ainvoke")


def test_chat_model_uses_db_config() -> None:
    """模型配置（地址 / 密钥 / 模型名）能正确落到 ChatOpenAI，换供应商只改工厂函数。"""
    cfg = SimpleNamespace(
        model_name="deepseek-chat",
        api_key="sk-test",
        base_url="https://api.deepseek.com",
    )
    llm = day09.build_chat_model(cfg, temperature=0)
    assert llm.model_name == "deepseek-chat"
    assert str(llm.openai_api_base).rstrip("/") == "https://api.deepseek.com/v1"
    assert llm.temperature == 0
    assert llm.max_retries == 0
