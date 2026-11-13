"""智能体开发学习接口的 Pydantic 结构。

设计要点：Agent 循环的返回值不是"一个答案"，而是一条**执行轨迹**（steps）——
前端把轨迹渲染成时间线，才能可视化地观察模型每一轮"决策 → 调工具 → 观察"的过程。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolOut(BaseModel):
    """单个工具的说明书（给前端展示用）。"""

    name: str
    description: str
    parameters: dict[str, Any]


class ToolCallTrace(BaseModel):
    """一次工具调用的记录：模型想调谁、传了什么参数、真实执行结果是什么。"""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str


class AgentStepTrace(BaseModel):
    """Agent 循环的一轮：要么调了工具（tool_calls），要么给出最终回答（content）。"""

    step: int
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    content: str | None = None


class AgentRunRequest(BaseModel):
    """运行 Function Calling Agent 循环的请求。"""

    question: str = Field(min_length=1, description="用户问题")
    max_steps: int = Field(default=8, ge=1, le=12, description="最大循环步数（防死循环）")
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")


class AgentRunResult(BaseModel):
    """Agent 循环的完整轨迹。"""

    steps: list[AgentStepTrace]
    final_answer: str
    stopped_by_limit: bool = Field(description="True 表示跑满 max_steps 仍没给出最终回答")
    model: str = Field(description="实际使用的模型名")


class ReactStepTrace(BaseModel):
    """ReAct 文本协议的一轮：模型原文 + 解析出的动作 + 工具观察结果。"""

    step: int
    raw_text: str = Field(description="模型本轮输出的原文（Thought/Action/...）")
    action_name: str | None = None
    action_args: dict[str, Any] | None = None
    observation: str | None = None
    is_error: bool = Field(default=False, description="True 表示解析失败（格式错误，已要求重试）")


class ReactRunRequest(BaseModel):
    """运行手写 ReAct 循环的请求。"""

    question: str = Field(min_length=1, description="用户问题")
    max_steps: int = Field(default=6, ge=1, le=12, description="最大循环步数")
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")
    session_id: str | None = Field(
        default=None,
        description=(
            "会话 ID：多次请求传同一个 ID 可保留 history 实现'接着上一句继续问'；"
            "不传则后端自动新建会话（返回值里会给出 ID，可拿它继续追问）"
        ),
    )


class ReactRunResult(BaseModel):
    """ReAct 循环的完整轨迹。"""

    steps: list[ReactStepTrace]
    final_answer: str
    stopped_by_limit: bool
    model: str
    session_id: str = Field(description="本次会话 ID，下次请求带上即可继续追问")
    history: str = Field(description="该会话累计的纯文本记忆（用户问题 + 模型输出 + Observation）")
    turns: int = Field(description="该会话已完成的提问轮数")


class ExtractAttemptTrace(BaseModel):
    """结构化抽取的一次尝试：模型原文、解析结果、失败原因、本轮实际使用的输出约束。"""

    attempt: int
    raw_content: str = Field(description="模型本轮输出的原文")
    parsed: dict[str, Any] | None = Field(default=None, description="解析/校验通过的结果")
    error: str | None = Field(default=None, description="解析或 Schema 校验失败的原因（会回填给模型重试）")
    response_format: str = Field(
        description="本轮实际使用的输出约束：none（口头要求）/ json_object / json_schema"
    )
    note: str | None = Field(default=None, description="降级/兜底说明（如服务不支持 json_schema 自动降级）")


class ExtractRunRequest(BaseModel):
    """Day 02 结构化输出实验请求：客户意图提取。"""

    text: str = Field(min_length=1, description="待抽取的用户原话")
    mode: Literal["naive", "json_mode", "json_schema"] = Field(
        default="json_schema",
        description="三级方案：naive 口头要求（反面教材）/ json_mode JSON Mode / json_schema Pydantic Schema",
    )
    max_retries: int = Field(
        default=2, ge=0, le=3,
        description="解析失败后带错误现场重试的次数（naive 模式是反面教材，固定不重试）",
    )
    model_config_id: int | None = Field(default=None, description="指定模型，不传用默认 LLM")


class ExtractRunResult(BaseModel):
    """结构化抽取结果：尝试轨迹 + 最终工单。"""

    mode: str = Field(description="本次使用的抽取方案")
    attempts: list[ExtractAttemptTrace] = Field(description="每次尝试的轨迹（含失败现场）")
    ticket: dict[str, Any] | None = Field(
        default=None, description="最终通过校验的工单；连续失败时为 null，绝不编造"
    )
    success: bool = Field(description="是否成功拿到通过校验的工单")
    model: str = Field(description="实际使用的模型名")
