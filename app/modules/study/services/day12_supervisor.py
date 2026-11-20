"""Day 12 多智能体 Supervisor：让一个 Agent 指挥一群专家。

课程 day12 的四个脚本 + 三道练习在这里收成一个模块：

    单 Agent 失败复现   → ``run_single``              （01：一个什么都会的 Agent 怎么假收敛 / 越权）
    最小 Supervisor     → ``run_team``                （02：主管只派活 + 专家只干专活 + FINISH）
    生产形态（完整版）  → ``start_full`` / ``resume_full``
                                                      （03：审批闸门 + 交接预算 + checkpointer，练习 1 去重守卫）
    并行专家            → ``run_parallel``            （04 + 练习 3：Send 扇出两个调研分支 → join → writer 只调一次）

贯穿全篇的三条结论：

1. 主管的「人格」不是 prompt 约束出来的，是**输出 schema 卡出来的**——`Route` 里没有
   content 字段，它想自己回答也没地方放；
2. 专家必须 **私有执行 + 摘要回传**：内部可以跑多轮工具，回传给团队的只有一句结论 +
   `AIMessage(name=角色)`，否则上下文互相污染；
3. 跳转 + 改状态一次完成用 ``Command(goto=..., update=...)``，**用了它就别再给该节点加静态出边**
   （两条来源冲突会抛 InvalidUpdateError 或跳出诡异路径）。

两个与课程代码不同的工程决定（本项目实测结论）：

- 课程 02/03 用 ``create_react_agent`` 搭专家子图；本模块改为**手写专家 ReAct 子图**。
  原因：① langgraph 1.x 已把 ``create_react_agent`` 标记弃用（迁移到
  ``langchain.agents.create_agent``），本项目不想在学习代码里留弃用 API；
  ② 手写才能把「私有工具集 + 步数刹车 + 摘要回传」收在一处，并且能用脚本化假模型做纯逻辑单测。
- 主管的结构化输出按 ``default → function_calling → json_mode`` 逐级降级，全失败才退关键词兜底
  （与本项目 deepseek-flash thinking 模式的实测行为一致，见 Day 09/10 两次踩坑记录）。

langgraph / langchain 全部延迟导入：缺依赖时接口返回可读的 400，而不是让应用起不来。
"""

import asyncio
import difflib
import json
import operator
import re
import time
from typing import Annotated, Any, Awaitable, Callable, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.study.schemas.study import (
    Day12FullRequest,
    Day12FullResumeRequest,
    Day12ParallelRequest,
    Day12SingleRequest,
    Day12SupervisorRequest,
)
from app.modules.study.services.day09_langchain import build_chat_model
from app.modules.study.services.day10_langgraph import _graph_structure, _jsonable, _text
from app.modules.study.services.day11_checkpointer import _last_answer, _snapshot_view
from app.modules.study.services.llm import resolve_model

INSTALL_HINT = "pip install -U langgraph langchain-openai"

# 交接预算（课程 03 的 MAX_HANDOFF）：超过就强制转 writer 降级交付
DEFAULT_MAX_HANDOFF = 6

# 两种强制收尾的派单话术：预算耗尽 vs 语义终止
DEGRADED_ASSIGNMENT = (
    "交接预算已耗尽。请在现有结论基础上给用户一个诚实的阶段性答复，"
    "明确说明哪些已经查清、哪些还没做完，不要编造没执行的动作。"
)
FINISH_ASSIGNMENT = "把已有结论写成最终对客答复，只写结论里出现过的事实。"

# 两个图的专家名册（roster）：主管能派的人。用常量便于单测与前端展示。
ROSTER_MIN = ["researcher", "writer"]
ROSTER_FULL = ["researcher", "coder", "writer"]

# ============================================================
# 静态概念数据（供前端渲染，全部来自课程原文）
# ============================================================

# 课程 1：单 Agent 的三堵墙
WALLS: list[dict[str, str]] = [
    {
        "no": "1",
        "title": "工具膨胀，模型开始选错",
        "symptom": "该查订单时调了 search_policy（都带『查』字）；把订单号传给 query 参数",
        "cause": "在一个决策点上要求它同时区分 18 个高度相似的选项（Day 04 的 7±2）",
        "fix": "按职责拆专家：每个专家只看到自己那 3~4 个工具，选项变少就不容易选错",
    },
    {
        "no": "2",
        "title": "prompt 臃肿，注意力被稀释",
        "symptom": "查报表时先『确认订单状态』；两套业务规则互相打架",
        "cause": "四套规则塞进同一个 system prompt，中段被 Lost in the Middle 吃掉（Day 06/08）",
        "fix": "一个专家一份 prompt，规则只写自己这摊事",
    },
    {
        "no": "3",
        "title": "权限与模型无法分级",
        "symptom": "查资料、跑 SQL、退款全在一个 Agent 里：想只给退款加审批都做不到",
        "cause": "权限是混在一起的，没法按能力段位单独配模型 / 闸门 / 数据可见性",
        "fix": "写操作单独成节点，前面挂 interrupt() 审批（Day 04 确认门 + Day 11 中断）",
    },
]

# 课程 2.1：三角色分工
ROLES: list[dict[str, str]] = [
    {
        "role": "supervisor",
        "only": "看进度、派单、判结束",
        "never": "不回答用户、不调业务工具（工具列表物理为空）",
        "why": "输出 schema 里没有 content 字段，它没有『说话』的位置",
    },
    {
        "role": "专家 expert",
        "only": "干自己那一件事（检索 / 计算 / 写文案）",
        "never": "不派活给别人、不决定整体结束",
        "why": "prompt 写负向边界 + 工具物理隔离（writer 的工具给空数组）",
    },
    {
        "role": "汇总 writer",
        "only": "把成果写成最终答复",
        "never": "不重新调研、不新增事实",
        "why": "让『收尾』也走固定角色，保证每条路径都以成稿结束",
    },
]

# 课程 2.3：专家之间的三种交接方式
HANDOFF_MODES: list[dict[str, str]] = [
    {
        "mode": "A. 静态边 + 条件路由",
        "how": "add_conditional_edges(\"supervisor\", route_fn)",
        "fit": "分支少、逻辑简单",
        "trap": "路由函数只能返回节点名，不能同时改 state——派单内容得再开一个字段中转",
    },
    {
        "mode": "B. Command(goto=, update=)",
        "how": "节点 return Command(goto=专家, update={assignment, steps:1})",
        "fit": "推荐：一次搞定『跳转 + 打补丁』",
        "trap": "别再给该节点加静态出边，否则两条来源冲突",
    },
    {
        "mode": "C. handoff 工具",
        "how": "create_handoff_tool(agent_name=...)",
        "fit": "想让模型自己决定转交（专家之间直接对话）",
        "trap": "交接链路不可控，容易互相推诿；本模块不用",
    },
]

# 课程 2.4：双层状态（共享 / 账本 / 私有）
STATE_LAYERS: list[dict[str, str]] = [
    {
        "layer": "共享层",
        "fields": "messages / next / assignment",
        "rule": "所有人都能看；回传消息必须带 name，否则主管分不清谁说了什么",
    },
    {
        "layer": "账本层",
        "fields": "findings（只追加） / steps（交接次数）",
        "rule": "给主管看进度；findings 拼进主管 prompt，才不会重复派同一个活",
    },
    {
        "layer": "私有层",
        "fields": "code_output / research_notes …",
        "rule": "只有对应专家读写，不塞 messages——避免所有专家被迫读一大段 stdout",
    },
]

# 课程 2.5：三层终止
TERMINATIONS: list[dict[str, str]] = [
    {
        "kind": "语义终止",
        "how": "主管输出 FINISH → 转 writer（或 END）",
        "note": "首选出口：模型认为活干完了",
    },
    {
        "kind": "资源终止",
        "how": "config={\"recursion_limit\": 30}",
        "note": "数的是超步数；主管+专家交替，一次派单 ≈ 2 个超步",
    },
    {
        "kind": "预算终止",
        "how": "steps 累计到 MAX_HANDOFF 就强制转 writer 降级交付",
        "note": "多智能体专属保险：语义终止失灵（乒乓）时还能拿到阶段性成果",
    },
]

# 课程 4：选型判据
DECISION_TABLE: list[dict[str, str]] = [
    {"criterion": "工具数量", "single": "≤ 8，语义不重叠", "supervisor": "> 12，或明显分成几组"},
    {"criterion": "职责差异", "single": "都是『查一查、答一答』", "supervisor": "检索 / 写代码 / 写文案 / 审批，差异大"},
    {"criterion": "权限", "single": "全只读", "supervisor": "有写操作需单独闸门"},
    {"criterion": "上下文", "single": "一份就够", "supervisor": "各专家只需看自己那部分"},
    {"criterion": "成本敏感", "single": "敏感（1x）", "supervisor": "可接受 2~5x"},
    {"criterion": "可并行", "single": "无", "supervisor": "有独立子任务"},
]

# 课程 4：成本实测感受（同一问题，gpt-4o-mini 量级）
COST_TABLE: list[dict[str, str]] = [
    {"arch": "单 Agent（4 工具）", "calls": "3~4", "tokens": "1x", "latency": "1x", "routing": "高"},
    {"arch": "单 Agent（18 工具）", "calls": "5~8", "tokens": "1.5x", "latency": "1.6x", "routing": "明显下降"},
    {"arch": "Supervisor（3 专家，串行 4 次交接）", "calls": "9~12", "tokens": "~4x", "latency": "~3x", "routing": "高且可解释"},
    {"arch": "Supervisor + 并行 2 路", "calls": "8~10", "tokens": "~4x", "latency": "~2x", "routing": "高"},
]

# 课程 5：五个必踩的坑
PITFALLS: list[dict[str, str]] = [
    {
        "no": "1",
        "title": "无限乒乓（A→B→A→B）",
        "symptom": "researcher 和 coder 互相甩锅，recursion_limit 报错或预算耗尽",
        "cause": "主管看不到『谁已经做过什么』，或 roster 描述让两者能力重叠",
        "fix": "findings 账本拼进主管 prompt + roster 写负向边界 + MAX_HANDOFF 预算（三件套一起上）",
    },
    {
        "no": "2",
        "title": "专家抢活",
        "symptom": "writer 自己编了个退款金额；coder 说『我顺便帮你查了政策』",
        "cause": "专家 prompt 只写『你负责什么』，没写『你不许干什么』",
        "fix": "prompt 加硬约束 + 工具物理隔离（writer 的工具列表给空数组）",
    },
    {
        "no": "3",
        "title": "上下文污染",
        "symptom": "跑到第 4 轮 token 已 12000；writer 开始复述『我刚才检索到……』",
        "cause": "所有专家的完整中间过程都进了共享 messages",
        "fix": "私有执行 + 摘要回传：内部跑 5 轮工具，回传给团队的只有最后一段结论 + name",
    },
    {
        "no": "4",
        "title": "主管自己动手",
        "symptom": "主管输出了完整答复，压根没派单",
        "cause": "用了自由文本 + 正则解析，模型自然想自己把活干了",
        "fix": "with_structured_output(Route)，schema 里没有 content 字段——让它没有『说话』的位置",
    },
    {
        "no": "5",
        "title": "既加静态边又返回 Command",
        "symptom": "InvalidUpdateError，或明明 goto=\"writer\" 却回了 supervisor",
        "cause": "节点出边有两条来源：add_edge 与 Command(goto=)，两者冲突",
        "fix": "二选一：用 Command 的节点不加静态出边；只有终点（writer → END）用静态边",
    },
]

# 课程 8：今日必掌握清单
CHECKLIST: list[str] = [
    "说清单 Agent 的三堵墙：工具膨胀选错、prompt 臃肿稀释、权限无法分级",
    "Supervisor 三角色分工：主管只派活、专家只干专活、writer 只收尾",
    "会用 with_structured_output(Route) 把主管的输出空间卡死（没有 content 字段）",
    "会用 Command(goto=..., update=...) 一次完成『跳转 + 打补丁』",
    "状态双层设计：共享层 + 账本层 + 私有层",
    "专家回传必须带 name，否则主管分不清谁说了什么",
    "三层终止：FINISH → writer、MAX_HANDOFF 预算、recursion_limit",
    "五个坑能对号入座：乒乓、抢活、上下文污染、主管自己动手、边与 Command 混用",
    "能做架构选型：工具 ≤ 8 就别拆；拆了要能算清 3~5 倍成本",
]

# 各实验的预设问题（前端一键填入）
PRESETS: dict[str, list[str]] = {
    "single": [
        "订单 A1001 签收几天了？按政策能不能退？帮我退掉，并给客户写一段说明。",
        "订单 A1001 的退款进度怎么样？顺便把 299 元退了，再给客户写一段解释。",
    ],
    "team": [
        "订单 A1001 签收几天了？按政策能不能退？给客户写一段说明。",
        "订单 A1002 能不能退？能退的话写一段说明给客户。",
    ],
    "full": [
        "订单 A1001 签收3天，能不能退？退 299，顺便算 299*0.85",
        "订单 A1002 签收 10 天，还能不能退？退 88，算一下 88*0.9",
    ],
    "parallel": [
        "订单 A1001 签收几天了？按政策能不能退？给客户写一段说明。",
        "订单 A1002 的退款政策是什么？顺便查一下它签收多久了。",
    ],
}

# 单 Agent 的系统提示词：故意写成"什么都会"，用来复现三堵墙
SINGLE_SYSTEM = """你是全能客服，可以查政策、查订单、退款、写话术。
要求：
1. 先查政策再回答；
2. 涉及退款要核对订单；
3. 最终用 write_reply 输出对客话术。"""

# 最小版主管：只派给 researcher / writer
MIN_SUPERVISOR_PROMPT = """你是团队主管，负责把任务分派给专家。你本人不回答用户、不使用任何工具。

专家名单：
- researcher：检索政策、核对订单事实。不写对客文案、不下结论性建议。
- writer：把已有结论写成对客说明。不检索、不新增事实。

分派规则：
1. 每轮只派一个人，且必须给出明确的一句话任务。
2. 事实还没查清 → researcher；事实已清楚但要给用户答复 → writer。
3. 已经拿到用户可用的答复，或不需要再做事 → FINISH。
4. 不要重复派已完成的活。
5. 只输出 json 对象，字段：next（researcher/writer/FINISH）、task、reason。"""

# 完整版主管：多一个 coder（会触发审批的执行者）
FULL_SUPERVISOR_PROMPT = """你是团队主管，负责把任务分派给专家。你本人不回答用户、不使用任何工具、不承诺任何结果。

专家名单：
- researcher：检索政策、核对订单事实。不写文案、不计算、不下结论。
- coder：执行计算与退款落地（写操作会弹人工审批）。不做检索、不写文案。
- writer：把已有结论写成对客说明。不检索、不计算、不新增事实。

分派规则：
1. 每轮只派一个人，且必须给出明确的一句话任务。
2. 事实没查清 → researcher；事实清楚且有数字要算 / 有退款要落地 → coder；
   结论齐了、要给用户答复 → writer。
3. 任务已经完成（用户可用答复已产出）→ FINISH，由 writer 收尾。
4. 不要重复派已完成的活；找不到该派谁就给 FINISH。
5. 只输出 json 对象，字段：next（researcher/coder/writer/FINISH）、task、reason。"""

# 专家 prompt 一律写「正向职责 + 负向边界」两段（坑 2 的修法）
RESEARCHER_PROMPT = """你是资料研究员。只做检索与事实核对：需要数据就调工具，工具没给的信息一律不说。
硬约束：不写对客话术、不承诺退款、不编造订单数据、不做金额计算。
输出：3 句以内的结论，必须标注依据（订单查询 / 政策库）。"""

WRITER_PROMPT = """你是对客文案。只根据对话里已有的调研结论写一段 100 字内的对客说明。
硬约束：不检索、不计算、不新增事实、不承诺没执行过的动作。
你的工具列表是空的——想查也查不到，这是故意的。"""


# ============================================================
# 0. 公共设施：依赖探测 / 存档器 / 假数据 / 配置
# ============================================================


def _require_langgraph() -> None:
    """依赖探测：没装 langgraph 时给安装命令，而不是 ImportError 堆栈。"""
    try:
        import langgraph.graph  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 只在缺依赖的环境触发
        raise ValueError(f"未安装 langgraph，请先执行：{INSTALL_HINT}") from exc


def _config(thread_id: str) -> dict[str, Any]:
    """构造 invoke 配置：thread_id 是会话身份（Day 11 的老朋友）。"""
    return {"configurable": {"thread_id": thread_id}}


def _thread_id(prefix: str) -> str:
    """生成一次性的演示会话 ID（生产应从业务键派生）。"""
    return f"{prefix}-{uuid4().hex[:6]}"


def _version(name: str) -> str:
    """读已安装包版本：框架篇的坑大半和版本有关。"""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001  没装就是没装，前端照常展示
        return "未安装"


# 进程级存档器：多个请求共享，才能演示"停在审批上、下一次请求续跑"
_DAY12_SAVER: Any | None = None

# 退款落地流水（进程内存，演示用）：thread_id → 一条条执行记录
REFUND_LOG: list[dict[str, Any]] = []


def _saver() -> Any:
    """进程级 MemorySaver 单例（与 Day 11 同一套做法）。"""
    global _DAY12_SAVER
    if _DAY12_SAVER is None:
        from langgraph.checkpoint.memory import MemorySaver

        _DAY12_SAVER = MemorySaver()
    return _DAY12_SAVER


# 假业务数据：订单库与政策库
ORDERS: dict[str, dict[str, Any]] = {
    "A1001": {"order_id": "A1001", "status": "已签收", "amount": 299, "signed_days_ago": 3},
    "A1002": {"order_id": "A1002", "status": "已签收", "amount": 88, "signed_days_ago": 10},
}

POLICIES: dict[str, str] = {
    "退款": "退款需在签收后 7 天内发起；超期需人工审批，最多 15 天。",
    "运费": "单笔满 99 元包邮，退货运费由买家承担。",
    "发票": "电子发票在确认收货后 24 小时内开具。",
}


def _build_tools() -> dict[str, Any]:
    """Day 12 的五个业务工具：三个只读 + 一个写操作 + 一个纯计算。

    工具描述写「何时用 / 何时不用」（Day 04 的四段式），因为今天的核心矛盾
    恰恰是"工具太多时描述互相干扰"——把边界写清楚，单 Agent 还能勉强撑住。
    """
    from langchain_core.tools import tool

    @tool(parse_docstring=True)
    def search_policy(query: str) -> str:
        """在公司政策知识库中检索售后条款。

        何时用：用户问退款政策、运费规则、发票开具等"规则类"问题。
        何时不用：需要具体订单的状态、金额、签收天数时（改用 get_order）。

        Args:
            query: 检索关键词，如"退款"、"运费"。
        """
        for key, value in POLICIES.items():
            if key in query:
                return value
        return "未找到相关政策。"

    @tool(parse_docstring=True)
    def get_order(order_id: str) -> str:
        """查询订单状态、金额与签收天数（只读）。

        何时用：用户提到具体订单号，需要事实数据时。
        何时不用：用户只问政策、不涉及具体订单时。

        Args:
            order_id: 订单号，形如 A1001。
        """
        data = ORDERS.get(order_id, {"order_id": order_id, "error": "订单不存在"})
        return json.dumps(data, ensure_ascii=False)

    @tool(parse_docstring=True)
    def refund_order(order_id: str, amount: float) -> str:
        """对指定订单发起退款（敏感写操作，会真实扣款）。

        何时用：用户明确要求退款，且已确认订单号与金额。
        何时不用：用户只在咨询政策，或订单还没核对清楚。

        Args:
            order_id: 订单号，形如 A1001。
            amount: 退款金额（元）。
        """
        return f"已为订单 {order_id} 退款 {amount:g} 元"

    @tool(parse_docstring=True)
    def write_reply(tone: str, content: str) -> str:
        """把结论润色成对客话术。

        何时用：已经拿到结论，需要统一口吻输出给客户时。
        何时不用：结论还没查清时（先查再说）。

        Args:
            tone: 语气，如 friendly / formal。
            content: 要润色的结论正文。
        """
        return f"[{tone}] {content}"

    @tool(parse_docstring=True)
    def calculate(expression: str) -> str:
        """计算四则运算表达式，例如 "299*0.85"。

        何时用：需要精确金额计算（折扣、差额）时。
        何时不用：心算就能确定的简单数字，或表达式含变量时。

        Args:
            expression: 只含数字与 + - * / ( ) 的算式。
        """
        return _safe_calc(expression)

    return {
        "search_policy": search_policy,
        "get_order": get_order,
        "refund_order": refund_order,
        "write_reply": write_reply,
        "calculate": calculate,
    }


_EXPR_PATTERN = re.compile(r"\d+(?:\.\d+)?(?:\s*[\+\-\*/]\s*\d+(?:\.\d+)?)+")


def _safe_calc(expression: str) -> str:
    """只允许四则运算的安全求值：白名单校验 + 空 builtins。

    课程 03 直接 ``eval`` 并注明"真实项目必须进沙箱（Day 22）"。这里再收一道：
    表达式必须整体匹配数字与 + - * / 与括号，其他字符一律拒绝。
    """
    expr = (expression or "").strip()
    if not re.fullmatch(r"[\d\.\s\+\-\*/\(\)%]+", expr or ""):
        return f"拒绝执行：表达式含白名单之外的字符（{expression!r}）"
    try:
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 - 已做字符白名单
    except Exception as exc:  # noqa: BLE001  计算失败也是模型要看的"观察结果"
        return f"计算失败：{type(exc).__name__}: {exc}"
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def _extract_expression(text: str) -> str:
    """从派单文本里抠出第一个带运算符的算式（"顺便算 299*0.85" → "299*0.85"）。"""
    match = _EXPR_PATTERN.search(text or "")
    return match.group(0) if match else ""


def _order_id_of(text: str) -> str:
    """从文本里抠订单号，抠不到就用演示默认值 A1001。"""
    match = re.search(r"[A-Z]\d{4}", text or "")
    return match.group(0) if match else "A1001"


def _amount_of(text: str) -> float:
    """从派单文本里抠退款金额（"退 299" → 299），抠不到就按订单金额。"""
    match = re.search(r"(?:退|退款)\s*(\d+(?:\.\d+)?)", text or "")
    if match:
        return float(match.group(1))
    order = ORDERS.get(_order_id_of(text)) or {}
    return float(order.get("amount") or 0)


# ============================================================
# 1. 专家子图：私有工具集 + 步数刹车（课程 02 的 create_react_agent 手写版）
# ============================================================


def _build_expert(
    llm: Any,
    tools: list[Any],
    role: str,
    prompt: str,
    *,
    max_steps: int = 3,
) -> Any:
    """搭一个"专家"ReAct 子图：model ⇄ tools，步数到上限就停。

    这就是 Day 10 那条回头边，只是把 system prompt 和工具集换成了该专家的私有版本。
    ``tools`` 为空时直接不绑工具——writer 想查也查不到（坑 2 的物理隔离）。
    """
    from langchain_core.messages import SystemMessage, ToolMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    tool_map = {item.name: item for item in tools}
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    class ExpertState(TypedDict):
        messages: Annotated[list, add_messages]
        steps: Annotated[int, operator.add]

    async def model_node(state: ExpertState) -> dict[str, Any]:
        response = await llm_with_tools.ainvoke([SystemMessage(content=prompt), *state["messages"]])
        return {"messages": [response], "steps": 1}

    async def tools_node(state: ExpertState) -> dict[str, Any]:
        last = state["messages"][-1]
        outputs: list[Any] = []
        for call in list(getattr(last, "tool_calls", None) or []):
            name = str(call.get("name") or "")
            fn = tool_map.get(name)
            if fn is None:
                content = f"未知工具 {name}，可用：{sorted(tool_map)}"
            else:
                try:
                    content = str(await fn.ainvoke(dict(call.get("args") or {})))
                except Exception as exc:  # noqa: BLE001  错误也包成 ToolMessage，配对不能断
                    content = f"工具执行失败：{type(exc).__name__}: {exc}"
            outputs.append(ToolMessage(content=content, tool_call_id=str(call.get("id") or "")))
        return {"messages": outputs}

    def after_model(state: ExpertState) -> str:
        """有 tool_calls 且预算没用完就继续转；否则收工（语义收敛 + 步数刹车）。"""
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        if int(state.get("steps") or 0) >= max_steps:
            return END
        return "tools"

    builder = StateGraph(ExpertState)
    builder.add_node("model", model_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", after_model, {"tools": "tools", END: END})
    builder.add_edge("tools", "model")
    return builder.compile(name=f"expert-{role}")


def _expert_conclusion(messages: list[Any]) -> str:
    """取专家子图的最终结论：最后一条不带 tool_calls 的 AI 消息。"""
    for message in reversed(messages or []):
        if getattr(message, "type", "") == "ai" and not getattr(message, "tool_calls", None):
            text = _text(message)
            if text:
                return text
    return "（专家在步数预算内没有给出结论）"


def _make_runner(agent: Any, role: str, counter: dict[str, int]) -> Callable[..., Awaitable[str]]:
    """把专家子图包成 "看历史 + 接派单 → 回一句结论" 的 runnable。

    关键就是这一层：**私有执行**（专家内部可以跑多轮工具）**摘要回传**
    （只把最后一段结论交回团队，中间推理留在子图里）。
    """

    async def run(messages: list[Any], assignment: str) -> str:
        from langchain_core.messages import HumanMessage

        brief = HumanMessage(content=f"【主管派单】{assignment}\n只完成这件事，完成后直接给结论。")
        out = await agent.ainvoke({"messages": [*messages, brief], "steps": 0})
        counter["expert_calls"] = counter.get("expert_calls", 0) + int(out.get("steps") or 0)
        return _expert_conclusion(out.get("messages") or [])

    return run


def _expert_node(
    role: str,
    runner: Callable[..., Awaitable[str]],
    *,
    back_to: str | None = "supervisor",
) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    """专家节点：执行 → 只回传 AIMessage(name=角色) + findings + 审计事件。

    ``back_to=None`` 表示该节点不返回 Command，改由调用方加静态出边
    （只有终点 writer 这么用，避免"边 + Command"混用，坑 5）。
    """

    async def node(state: dict[str, Any]) -> Any:
        from langchain_core.messages import AIMessage
        from langgraph.types import Command

        started = time.perf_counter()
        label = str(state.get("branch") or role)
        try:
            answer = await runner(state.get("messages") or [], str(state.get("assignment") or ""))
        except Exception as exc:  # noqa: BLE001  专家挂了也要把控制权交回主管
            answer = f"（专家执行失败：{type(exc).__name__}: {exc}）"
        elapsed = int((time.perf_counter() - started) * 1000)
        patch = {
            "messages": [AIMessage(content=answer, name=label)],
            "findings": [f"[{label}] {answer}"],
            "events": [
                {
                    "node": label,
                    "kind": "expert",
                    "detail": answer[:300],
                    "elapsed_ms": elapsed,
                }
            ],
        }
        if back_to is None:
            return patch
        return Command(goto=back_to, update=patch)

    return node


# ============================================================
# 2. 主管：结构化路由 + 预算 + 去重守卫（课程 02 / 03 + 练习 1）
# ============================================================


class Route(BaseModel):
    """主管唯一被允许输出的东西。

    故意**没有 content 字段**——这就是"主管人格靠 schema 卡出来"的落地方式：
    它想说一句完整答复都没有地方放（坑 4 的修法）。
    """

    next: Literal["researcher", "coder", "writer", "FINISH"] = Field(description="下一个节点")
    task: str = Field(description="给这位专家的一句话任务说明")
    reason: str = Field(description="为什么派给他，一句话（便于排查乒乓）")


def _heuristic_route(state: dict[str, Any], roster: list[str]) -> dict[str, Any]:
    """兜底路由：结构化输出整体不可用时的关键词策略。

    真实项目不该长期依赖它——它保证的是"实验台在 thinking 模型上仍然能跑完"，
    并把"主管降级到关键词"这件事显式标进事件里，而不是假装一切正常。
    """
    findings = " ".join(state.get("findings") or [])
    question = ""
    for message in state.get("messages") or []:
        if getattr(message, "type", "") == "human":
            question = _text(message)
            break

    if "researcher" in roster and "[researcher]" not in findings:
        return {"next": "researcher", "task": f"查清事实与政策：{question[:60]}", "reason": "还没有任何结论"}
    if (
        "coder" in roster
        and "[coder]" not in findings
        and re.search(r"\d+\s*[\+\-\*/]\s*\d+|退款|退\s*\d+", question)
    ):
        return {"next": "coder", "task": "按用户要求执行计算并落地退款", "reason": "还有数字要算 / 退款要执行"}
    if "writer" in roster and "[writer]" not in findings:
        return {"next": "writer", "task": "把已有结论写成对客答复", "reason": "事实已清楚，需要成稿"}
    return {"next": "FINISH", "task": "", "reason": "已有可用答复"}


def _make_decider(
    llm: Any, roster: list[str], prompt: str, counter: dict[str, int]
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """构造主管的决策函数：with_structured_output 三级降级 → 关键词兜底。"""

    async def decide(state: dict[str, Any]) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        done = state.get("findings") or ["（暂无）"]
        messages = [
            SystemMessage(content=prompt),
            *(state.get("messages") or []),
            HumanMessage(content="已完成事项：\n" + "\n".join(done) + "\n\n下一步派谁？只输出 json 对象。"),
        ]
        for method in ("default", "function_calling", "json_mode"):
            try:
                runner = (
                    llm.with_structured_output(Route)
                    if method == "default"
                    else llm.with_structured_output(Route, method=method)
                )
                counter["supervisor_calls"] = counter.get("supervisor_calls", 0) + 1
                result = await runner.ainvoke(messages)
                data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
                nxt = str(data.get("next") or "").strip()
                if nxt not in roster and nxt != "FINISH":
                    continue  # 模型编了个不存在的人：当作这次下发失败，继续降级
                return {
                    "next": nxt,
                    "task": str(data.get("task") or "").strip(),
                    "reason": str(data.get("reason") or "").strip(),
                    "method": method,
                }
            except Exception:  # noqa: BLE001  降级失败也是课堂素材
                continue
        counter["supervisor_calls"] = counter.get("supervisor_calls", 0) + 1
        fallback = _heuristic_route(state, roster)
        fallback["method"] = "heuristic"
        return fallback

    return decide


def _build_team_graph(
    *,
    decide: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    nodes: dict[str, Callable[..., Any]],
    roster: list[str],
    max_handoff: int = DEFAULT_MAX_HANDOFF,
    dedupe: bool = False,
    dedupe_threshold: float = 0.75,
    finish_to_writer: bool = False,
    extra_edges: list[tuple[str, str]] | None = None,
    checkpointer: Any = None,
) -> Any:
    """组装「主管 + 专家」图：主管用 Command 派单，专家用 Command 回跳。

    ``extra_edges`` 只给终点用（``("writer", "END")``）——用 Command 的节点绝不能
    再加静态出边（坑 5）。``dedupe`` 是练习 1 的去重守卫。
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.types import Command

    writer_role = "writer" if "writer" in nodes else None

    class TeamState(TypedDict):
        messages: Annotated[list, add_messages]
        next: str
        assignment: str
        findings: Annotated[list[str], operator.add]
        steps: Annotated[int, operator.add]
        dispatches: Annotated[list[dict[str, Any]], operator.add]
        events: Annotated[list[dict[str, Any]], operator.add]
        code_output: str
        thread_id: str
        terminated: bool

    async def supervisor_node(state: TeamState) -> Any:
        steps = int(state.get("steps") or 0)

        # ⓪ 已经强制收过尾：最小图的 writer 会回跳主管，这里必须直接结束，
        #    否则「预算耗尽 → writer → 主管 → 预算耗尽」会变成死循环（实测踩过）。
        if state.get("terminated"):
            return Command(
                goto=END,
                update={
                    "next": "FINISH",
                    "events": [
                        {"node": "supervisor", "kind": "finish", "detail": "已强制收尾 → END"}
                    ],
                },
            )

        # ① 预算终止：交接次数用完 → 强制收尾（降级交付，而不是抛异常）
        if max_handoff and steps >= max_handoff:
            target = writer_role or END
            return Command(
                goto=target,
                update={
                    "next": writer_role or "FINISH",
                    "assignment": DEGRADED_ASSIGNMENT,
                    "terminated": True,
                    "events": [
                        {
                            "node": "supervisor",
                            "kind": "budget",
                            "detail": f"交接预算耗尽（{steps}/{max_handoff}）→ "
                            + ("writer 降级交付" if writer_role else "直接收尾"),
                        }
                    ],
                },
            )

        route = await decide(state)
        proposed = str(route.get("next") or "")
        task = str(route.get("task") or "").strip()

        # ② 语义终止：FINISH。完整版让它走 writer，保证每条路径都以成稿结束
        if proposed == "FINISH":
            if finish_to_writer and writer_role:
                return Command(
                    goto=writer_role,
                    update={
                        "next": writer_role,
                        "assignment": FINISH_ASSIGNMENT,
                        "events": [
                            {"node": "supervisor", "kind": "finish", "detail": f"FINISH → {writer_role} 收尾"}
                        ],
                    },
                )
            return Command(
                goto=END,
                update={
                    "next": "FINISH",
                    "events": [{"node": "supervisor", "kind": "finish", "detail": "FINISH → END"}],
                },
            )

        # ③ 模型编了不存在的人：当成这次决策失败，直接收尾
        if proposed not in nodes:
            target = writer_role or END
            return Command(
                goto=target,
                update={
                    "next": writer_role or "FINISH",
                    "assignment": DEGRADED_ASSIGNMENT,
                    "terminated": True,
                    "events": [
                        {
                            "node": "supervisor",
                            "kind": "budget",
                            "detail": f"主管给出了不存在的专家 {proposed!r} → 强制收尾",
                        }
                    ],
                },
            )

        # ④ 练习 1 去重守卫：连续两次派给同一人且任务高度相似 → 判乒乓，转 writer 汇总
        dispatches = list(state.get("dispatches") or [])
        last = dispatches[-1] if dispatches else None
        if dedupe and writer_role and last and last.get("role") == proposed:
            ratio = difflib.SequenceMatcher(
                None, str(last.get("task") or ""), task
            ).ratio()
            if ratio >= dedupe_threshold:
                return Command(
                    goto=writer_role,
                    update={
                        "next": writer_role,
                        "assignment": "两轮派单高度相似，判定为乒乓。请基于现有结论给用户一个诚实的阶段答复，"
                        "并说明哪一步没做完。",
                        "terminated": True,
                        "events": [
                            {
                                "node": "supervisor",
                                "kind": "dedupe",
                                "detail": f"去重守卫命中：连续派给 {proposed} 且任务相似度 {ratio:.2f} → 跳过本次派单",
                            }
                        ],
                    },
                )

        # ⑤ 正常派单：跳转 + 打补丁一次完成（assignment / 账本 / 预算计数）
        return Command(
            goto=proposed,
            update={
                "next": proposed,
                "assignment": task,
                "steps": 1,
                "dispatches": [
                    {"role": proposed, "task": task, "reason": str(route.get("reason") or "")}
                ],
                "events": [
                    {
                        "node": "supervisor",
                        "kind": "dispatch",
                        "detail": f"→ {proposed}｜{task}",
                        "method": str(route.get("method") or ""),
                        "reason": str(route.get("reason") or ""),
                    }
                ],
            },
        )

    builder = StateGraph(TeamState)
    builder.add_node("supervisor", supervisor_node)
    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_edge(START, "supervisor")
    for source, target in extra_edges or []:
        builder.add_edge(source, END if target == "END" else target)
    return builder.compile(checkpointer=checkpointer, name="day12-team")


# ============================================================
# 3. 01：先复现"单 Agent 的失败"（三堵墙）
# ============================================================


async def _run_tool_loop(
    llm: Any, tools: list[Any], messages: list[Any], max_steps: int
) -> dict[str, Any]:
    """跑一个最小 Function Calling 循环，把每一步都记成可展示的轨迹。

    与 Day 01/05 的循环同构，只是不再 print：决策 / 工具调用 / 观察结果三段分开记，
    这样前端能一眼看出"它到底调没调 refund_order"。
    """
    from langchain_core.messages import ToolMessage

    tool_map = {item.name: item for item in tools}
    llm_with_tools = llm.bind_tools(tools)
    history = list(messages)
    steps: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    outputs: list[dict[str, str]] = []
    final_answer = ""
    llm_calls = 0

    for index in range(1, max_steps + 1):
        response = await llm_with_tools.ainvoke(history)
        llm_calls += 1
        history.append(response)
        calls = list(getattr(response, "tool_calls", None) or [])
        steps.append(
            {
                "index": index,
                "kind": "decision",
                "content": _text(response),
                "tool_calls": [
                    {"name": str(call.get("name") or ""), "args": _jsonable(call.get("args") or {})}
                    for call in calls
                ],
            }
        )
        if not calls:
            final_answer = _text(response)
            break
        for call in calls:
            name = str(call.get("name") or "")
            counts[name] = counts.get(name, 0) + 1
            fn = tool_map.get(name)
            if fn is None:
                content = f"未知工具 {name}，可用：{sorted(tool_map)}"
            else:
                try:
                    content = str(await fn.ainvoke(dict(call.get("args") or {})))
                except Exception as exc:  # noqa: BLE001  异常也包成观察结果
                    content = f"工具执行失败：{type(exc).__name__}: {exc}"
            outputs.append({"tool": name, "content": content})
            steps.append({"index": index, "kind": "tool", "tool": name, "content": content})
            history.append(
                ToolMessage(content=content, tool_call_id=str(call.get("id") or ""))
            )
    else:
        final_answer = "（达到步数上限仍未收敛）"

    return {
        "steps": steps,
        "tool_counts": counts,
        "tool_outputs": outputs,
        "final_answer": final_answer,
        "llm_calls": llm_calls,
        "messages": history,
    }


# 声称"已经做完"的词；带条件/将来时态的句子不算（"回复确认后提交退款申请" 不是假收敛）
_REFUND_CLAIMS = ("已提交退款", "已为您退款", "退款已提交", "已退款", "已提交退款申请")
_CONDITION_WORDS = ("将", "需要", "需在", "待您", "待确认", "若", "如需", "未", "无法", "不能", "回复确认")


def _claims_done(answer: str) -> bool:
    """判断回答里有没有"声称已完成"的句子（逐句判定，避免条件句误伤）。"""
    for sentence in re.split(r"[。！？\n]", answer or ""):
        if any(word in sentence for word in _REFUND_CLAIMS) and not any(
            word in sentence for word in _CONDITION_WORDS
        ):
            return True
    return False


def _assess_issues(
    *, toolset: str, counts: dict[str, int], outputs: list[dict[str, str]], final_answer: str
) -> list[str]:
    """把课程 3.1 的三个"看着还行"的问题变成机器可查的结论。"""
    issues: list[str] = []
    answer = final_answer or ""

    # 问题 1：假收敛——回答声称退款完成，但 refund_order 一次都没调
    claims_refund = _claims_done(answer)
    if claims_refund and counts.get("refund_order", 0) == 0:
        issues.append(
            "假收敛：回答里声称『已提交退款』，但 refund_order 一次都没调用——编造了执行结果（Day 05）。"
        )

    # 问题 2：越权——写操作在没有任何审批闸门的情况下直接执行
    if counts.get("refund_order", 0) > 0:
        issues.append(
            "越权：refund_order 在没有审批闸门的情况下直接扣了款——权限没有分级（Day 04 确认门 / Day 11 interrupt）。"
        )

    # 问题 3：工具返回值被丢弃——write_reply 产出的话术没进最终回答
    reply_outputs = [row["content"] for row in outputs if row["tool"] == "write_reply"]
    if reply_outputs:
        payload = reply_outputs[-1].split("] ", 1)[-1]
        if payload and payload[:12] not in answer:
            issues.append(
                "工具返回值没人用：write_reply 产出了话术，最终回答却另写一遍——『写话术』和『回答用户』混为一谈。"
            )

    if toolset == "wide":
        issues.append(
            "工具膨胀：一个决策点上要同时区分 5 个工具（含 1 个写操作），描述之间开始互相干扰（Day 04 的 7±2）。"
        )
    return issues


async def run_single(db: AsyncSession, payload: Day12SingleRequest) -> dict[str, Any]:
    """课程 01：一个"什么都会"的 Agent，观察它怎么选错工具 / 假收敛 / 越权。

    ``toolset=wide`` 给全部 5 个工具，``narrow`` 只给 3 个只读工具（去掉 refund_order）：
    同一句问法、同一模型，工具集一收窄，越权与假收敛都会明显减少——这就是"该拆"的证据。
    """
    _require_langgraph()
    from langchain_core.messages import HumanMessage, SystemMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day12-single")
    tools = _build_tools()
    names = (
        ["search_policy", "get_order", "refund_order", "write_reply", "calculate"]
        if payload.toolset == "wide"
        else ["search_policy", "get_order", "write_reply"]
    )
    selected = [tools[name] for name in names]

    started = time.perf_counter()
    loop = await _run_tool_loop(
        llm,
        selected,
        [SystemMessage(content=SINGLE_SYSTEM), HumanMessage(content=payload.question)],
        payload.max_steps,
    )
    issues = _assess_issues(
        toolset=payload.toolset,
        counts=loop["tool_counts"],
        outputs=loop["tool_outputs"],
        final_answer=loop["final_answer"],
    )

    return {
        "question": payload.question,
        "toolset": payload.toolset,
        "tools_available": names,
        "steps": loop["steps"],
        "tool_counts": loop["tool_counts"],
        "final_answer": loop["final_answer"],
        "issues": issues,
        "llm_calls": loop["llm_calls"],
        "note": (
            "问题不在模型，在你让它一个人干四个部门的活。"
            "把同一句问法换成 narrow（去掉退款工具）再跑一次，对比 issues 会少很多。"
        ),
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ============================================================
# 4. 02：最小可跑的 Supervisor
# ============================================================


def _initial_team_state(question: str, *, thread_id: str = "") -> dict[str, Any]:
    """团队图的初始 state：账本层全部显式初始化（reducer 字段给空 list）。"""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=question)],
        "next": "",
        "assignment": "",
        "findings": [],
        "steps": 0,
        "dispatches": [],
        "events": [],
        "code_output": "",
        "thread_id": thread_id,
        "terminated": False,
    }


def _team_summary(state: dict[str, Any], counter: dict[str, int], graph: Any) -> dict[str, Any]:
    """把团队图的最终 state 摘成前端要的形状（四张卡片：答复 / 派单 / 结论 / 事件）。"""
    messages = state.get("messages") or []
    events = list(state.get("events") or [])
    return {
        "final_answer": _last_answer(messages) or "（模型没有产出最终回答）",
        "findings": list(state.get("findings") or []),
        "handoffs": list(state.get("dispatches") or []),
        "events": events,
        "messages": _jsonable(messages),
        "steps": int(state.get("steps") or 0),
        "dedupe_skips": len([item for item in events if item.get("kind") == "dedupe"]),
        "budget_hit": any(item.get("kind") == "budget" for item in events),
        "llm_calls": int(counter.get("supervisor_calls", 0)) + int(counter.get("expert_calls", 0)),
        "graph": graph,
    }


async def run_team(db: AsyncSession, payload: Day12SupervisorRequest) -> dict[str, Any]:
    """课程 02：1 个主管 + 2 个专家 + FINISH，跑通"派单—回传—再派单"。

    专家回传的每条消息都带 ``name``——主管在 messages 里能分清谁说的，
    ``findings`` 则拼进下一轮主管 prompt，避免重复派同一个活。
    """
    _require_langgraph()
    from langgraph.errors import GraphRecursionError

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day12-team")
    tools = _build_tools()
    counter: dict[str, int] = {}

    researcher = _build_expert(
        llm, [tools["search_policy"], tools["get_order"]], "researcher", RESEARCHER_PROMPT, max_steps=3
    )
    writer = _build_expert(llm, [], "writer", WRITER_PROMPT, max_steps=1)
    nodes = {
        "researcher": _expert_node("researcher", _make_runner(researcher, "researcher", counter)),
        # 最小版里 writer 也回传主管（课程 02 的写法），由主管判定 FINISH → END
        "writer": _expert_node(
            "writer", _make_runner(writer, "writer", counter), back_to="supervisor"
        ),
    }
    decide = _make_decider(llm, ROSTER_MIN, MIN_SUPERVISOR_PROMPT, counter)

    app = _build_team_graph(
        decide=decide,
        nodes=nodes,
        roster=ROSTER_MIN,
        max_handoff=payload.max_handoff,
        dedupe=payload.dedupe_guard,
        finish_to_writer=False,
    )

    started = time.perf_counter()
    degraded = ""
    state: dict[str, Any] = {}
    try:
        state = await app.ainvoke(
            _initial_team_state(payload.question),
            {"recursion_limit": payload.recursion_limit},
        )
    except GraphRecursionError as exc:
        # 图外降级：没有 checkpointer，进程里这轮 state 已经丢了，只能如实告知
        degraded = f"撞上 recursion_limit={payload.recursion_limit}：{exc}"
        state = _initial_team_state(payload.question)

    summary = _team_summary(state, counter, None)
    summary["graph"] = _graph_structure(
        app,
        labels={
            ("__start__", "supervisor"): "入口",
            ("supervisor", "researcher"): "Command(goto=researcher)",
            ("supervisor", "writer"): "Command(goto=writer)",
            ("supervisor", "__end__"): "FINISH",
            ("researcher", "supervisor"): "Command(goto=supervisor)",
            ("writer", "supervisor"): "Command(goto=supervisor)",
        },
    )
    return {
        "question": payload.question,
        "final_answer": summary["final_answer"],
        "findings": summary["findings"],
        "handoffs": summary["handoffs"],
        "events": summary["events"],
        "messages": summary["messages"],
        "steps": summary["steps"],
        "dedupe_skips": summary["dedupe_skips"],
        "budget_hit": summary["budget_hit"],
        "llm_calls": summary["llm_calls"],
        "graph": summary["graph"],
        "note": degraded
        or (
            "主管的工具列表是空的——物理上不可能自己调业务工具；"
            "每个专家回传的都是 AIMessage(name=角色) + 一句 findings。"
        ),
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ============================================================
# 5. 03：加审批、加预算、加存档（生产形态）
# ============================================================


def _coder_node() -> Callable[[dict[str, Any]], Awaitable[Any]]:
    """执行专家：数字计算 + 退款落地。

    写操作前挂 ``interrupt()``（Day 11 的确认门）：**interrupt 之前只做纯读**，
    真正 append 流水发生在拿到 resume 值之后——所以节点被重放也不会扣两次款。

    计算结果写 ``code_output``（私有层），messages 里只留一句结论，
    避免 writer 被迫读一大段中间过程（坑 3）。
    """

    async def node(state: dict[str, Any]) -> Any:
        from langchain_core.messages import AIMessage
        from langgraph.types import Command, interrupt

        assignment = str(state.get("assignment") or "")
        thread_id = str(state.get("thread_id") or "")
        order_id = _order_id_of(assignment)
        events: list[dict[str, Any]] = []
        findings: list[str] = []
        note = ""

        # ① 命中写操作：先问人（这里只有 interrupt，零副作用）
        if "退款" in assignment or "refund" in assignment.lower():
            decision = interrupt(
                {
                    "type": "confirm",
                    "action": "refund",
                    "tool": "refund_order",
                    "args": {"order_id": order_id, "amount": _amount_of(assignment)},
                    "prompt": f"即将为订单 {order_id} 执行退款，确认？",
                }
            )
            if isinstance(decision, dict):
                approved = bool(decision.get("approved", True))
                note = str(decision.get("note") or "")
                patched = decision.get("amount")
            else:
                approved = bool(decision)
                patched = None
            amount = float(patched) if patched is not None else _amount_of(assignment)
            events.append(
                {
                    "node": "coder",
                    "kind": "approval",
                    "approved": approved,
                    "detail": f"退款审批：{'放行' if approved else '驳回'}"
                    + (f"（人工改金额 {amount:g}）" if patched is not None else "")
                    + (f"｜{note}" if note else ""),
                }
            )
            if not approved:
                answer = "退款被人工驳回，未执行任何资金操作。"
                return Command(
                    goto="supervisor",
                    update={
                        "messages": [AIMessage(content=answer, name="coder")],
                        "findings": ["[coder] 退款被人工驳回，未执行"],
                        "events": events,
                    },
                )
            # ② 通过之后才产生副作用（重放安全）
            REFUND_LOG.append(
                {
                    "thread_id": thread_id,
                    "order_id": order_id,
                    "amount": amount,
                    "approved": True,
                    "note": note,
                }
            )
            findings.append(f"[coder] 已为订单 {order_id} 退款 {amount:g} 元")

        # ③ 纯计算：从派单里抠算式（没有就算不出，如实说明）
        expression = _extract_expression(assignment)
        if expression:
            result = _safe_calc(expression)
        else:
            result = "（派单里没有可计算的算式）"

        answer = f"计算结果：{result}" + (f"；{findings[0].split('] ', 1)[-1]}" if findings else "")
        findings.append(f"[coder] 计算 {expression or '（无算式）'} = {result}")
        events.append({"node": "coder", "kind": "expert", "detail": answer[:300]})
        return Command(
            goto="supervisor",
            update={
                "code_output": result,
                "messages": [AIMessage(content=answer, name="coder")],
                "findings": findings,
                "events": events,
            },
        )

    return node


def _build_full_app(
    llm: Any,
    *,
    max_handoff: int,
    counter: dict[str, int],
    checkpointer: Any | None = None,
) -> Any:
    """完整版图：supervisor + researcher + coder（带审批）+ writer（终点）。

    ``checkpointer`` 默认用进程级 MemorySaver（跨请求续跑）；单测可注入独立实例。
    """
    tools = _build_tools()
    researcher = _build_expert(
        llm, [tools["search_policy"], tools["get_order"]], "researcher", RESEARCHER_PROMPT, max_steps=3
    )
    writer = _build_expert(llm, [], "writer", WRITER_PROMPT, max_steps=1)
    nodes = {
        "researcher": _expert_node("researcher", _make_runner(researcher, "researcher", counter)),
        "coder": _coder_node(),
        # writer 是终点：不返回 Command，靠静态边 writer → END（坑 5 的正确用法）
        "writer": _expert_node("writer", _make_runner(writer, "writer", counter), back_to=None),
    }
    decide = _make_decider(llm, ROSTER_FULL, FULL_SUPERVISOR_PROMPT, counter)
    return _build_team_graph(
        decide=decide,
        nodes=nodes,
        roster=ROSTER_FULL,
        max_handoff=max_handoff,
        dedupe=False,
        finish_to_writer=True,
        extra_edges=[("writer", "END")],
        checkpointer=checkpointer if checkpointer is not None else _saver(),
    )


async def start_full(db: AsyncSession, payload: Day12FullRequest) -> dict[str, Any]:
    """课程 03 第一段：跑到退款审批处冻结（或被预算拦下直接收尾）。"""
    _require_langgraph()

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day12-full")
    counter: dict[str, int] = {}
    app = _build_full_app(llm, max_handoff=payload.max_handoff, counter=counter)

    thread_id = payload.thread_id or _thread_id("team")
    config_a = _config(thread_id)

    started = time.perf_counter()
    state = await app.ainvoke(
        _initial_team_state(payload.question, thread_id=thread_id),
        {**config_a, "recursion_limit": payload.recursion_limit},
    )
    snap = await app.aget_state(config_a)
    view = _snapshot_view(snap)

    interrupt_payload = None
    for task in view["tasks"]:
        if task["interrupts"]:
            interrupt_payload = task["interrupts"][0]
            break

    paused = bool(view["next"])
    summary = _team_summary(state, counter, None)
    return {
        "thread_id": thread_id,
        "paused": paused,
        "next": view["next"],
        "interrupt": interrupt_payload,
        "answer": "" if paused else summary["final_answer"],
        "findings": summary["findings"],
        "handoffs": summary["handoffs"],
        "events": summary["events"],
        "steps": summary["steps"],
        "budget_hit": summary["budget_hit"],
        "snapshot": view,
        "graph": _graph_structure(
            app,
            labels={
                ("__start__", "supervisor"): "入口",
                ("supervisor", "researcher"): "Command(goto=researcher)",
                ("supervisor", "coder"): "Command(goto=coder)（退款在这里弹审批）",
                ("supervisor", "writer"): "FINISH / 预算耗尽 → writer 收尾",
                ("writer", "__end__"): "★ 静态边：writer 是终点",
                ("researcher", "supervisor"): "Command(goto=supervisor)",
                ("coder", "supervisor"): "Command(goto=supervisor)",
            },
        ),
        "note": (
            "图停在 coder 的超步边界，state 已存档：进程可以死，thread_id 不变就能从这一帧继续。"
            if paused
            else "这次没有命中写操作，图直接跑完了（可以换成含『退款』的问法再试）。"
        ),
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


async def resume_full(db: AsyncSession, payload: Day12FullResumeRequest) -> dict[str, Any]:
    """课程 03 第二段：``Command(resume=决定)`` 从审批帧继续跑。

    ``decision`` 三种形态：``True`` 放行 / ``False`` 驳回 / ``{"approved": True, "amount": 199}``
    改参数后放行。审批节点被重放是安全的——它里面只有 ``interrupt()``。
    """
    _require_langgraph()
    from langgraph.types import Command

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day12-full-resume")
    counter: dict[str, int] = {}
    app = _build_full_app(llm, max_handoff=payload.max_handoff, counter=counter)
    config_a = _config(payload.thread_id)

    started = time.perf_counter()
    state = await app.ainvoke(
        Command(resume=payload.decision),
        {**config_a, "recursion_limit": payload.recursion_limit},
    )
    snap = await app.aget_state(config_a)
    view = _snapshot_view(snap)
    summary = _team_summary(state, counter, None)

    approvals = [item for item in summary["events"] if item.get("kind") == "approval"]
    approved = bool(approvals[-1].get("approved")) if approvals else False
    refunds = [row for row in REFUND_LOG if row.get("thread_id") == payload.thread_id]

    return {
        "thread_id": payload.thread_id,
        "approved": approved,
        "final_answer": summary["final_answer"],
        "refunds": _jsonable(refunds),
        "findings": summary["findings"],
        "handoffs": summary["handoffs"],
        "events": summary["events"],
        "steps": summary["steps"],
        "budget_hit": summary["budget_hit"],
        "next": view["next"],
        "note": (
            "审批节点被重放过，但它里面只有 interrupt，所以没有重复副作用；"
            "退款流水只在 resume 之后 append 一次（Day 11 的坑 1）。"
        ),
        "model": cfg.model_name,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


# ============================================================
# 6. 04：该并行的时候别排队（Send 扇出 + join 汇聚）
# ============================================================


def _build_parallel_app(llm: Any, order_id: str, counter: dict[str, int]) -> Any:
    """并行图：START →(Send 扇出)→ 政策 / 订单 两条调研 → join → writer → END。

    两条分支各挂**私有工具集**（一条只有 search_policy，一条只有 get_order），
    节点返回 dict（不返回 Command），汇聚靠静态边自动等待——与主管模式的写法
    刻意区分开：这里没有"谁下一步"的决策，只有确定的扇出与汇聚。
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.types import Send

    tools = _build_tools()
    policy_agent = _build_expert(
        llm, [tools["search_policy"]], "researcher_policy", RESEARCHER_PROMPT, max_steps=2
    )
    order_agent = _build_expert(
        llm, [tools["get_order"]], "researcher_order", RESEARCHER_PROMPT, max_steps=2
    )
    writer_agent = _build_expert(llm, [], "writer", WRITER_PROMPT, max_steps=1)

    class PState(TypedDict):
        messages: Annotated[list, add_messages]
        assignment: str
        branch: str
        findings: Annotated[list[str], operator.add]
        events: Annotated[list[dict[str, Any]], operator.add]

    def plan(state: PState) -> list[Any]:
        """扇出：调研与核对互不依赖 → 同时派两条（Send 的载荷是各分支的私有 state）。"""
        return [
            Send(
                "researcher_policy",
                {**state, "assignment": "查退款政策：签收后几天内可退、超期怎么办", "branch": "政策"},
            ),
            Send(
                "researcher_order",
                {**state, "assignment": f"查订单 {order_id} 的签收天数与金额", "branch": "订单"},
            ),
        ]

    # 两条分支共用同一个节点实现（用 branch 区分署名与工具集），返 dict → 静态边汇聚
    async def scout_node(state: PState) -> dict[str, Any]:
        from langchain_core.messages import AIMessage

        started = time.perf_counter()
        branch = str(state.get("branch") or "")
        agent = policy_agent if branch == "政策" else order_agent
        runner = _make_runner(agent, "researcher", counter)
        answer = await runner(state.get("messages") or [], str(state.get("assignment") or ""))
        elapsed = int((time.perf_counter() - started) * 1000)
        return {
            "messages": [AIMessage(content=answer, name=f"researcher:{branch}")],
            "findings": [f"[{branch}] {answer}"],
            "events": [
                {
                    "node": f"researcher:{branch}",
                    "kind": "expert",
                    "detail": answer[:300],
                    "elapsed_ms": elapsed,
                }
            ],
        }

    async def join_node(state: PState) -> dict[str, Any]:
        """汇聚点：两条分支的 findings 已由 reducer 合并，这里只负责打包给 writer。"""
        merged = "\n".join(state.get("findings") or [])
        return {
            "assignment": f"基于以下两条并行调研结论写答复，不要新增事实：\n{merged}",
            "events": [
                {
                    "node": "join",
                    "kind": "join",
                    "detail": f"合并 {len(state.get('findings') or [])} 条结论",
                }
            ],
        }

    writer_node = _expert_node("writer", _make_runner(writer_agent, "writer", counter), back_to=None)

    builder = StateGraph(PState)
    builder.add_node("researcher_policy", scout_node)
    builder.add_node("researcher_order", scout_node)
    builder.add_node("join", join_node)
    builder.add_node("writer", writer_node)
    builder.add_conditional_edges(START, plan, ["researcher_policy", "researcher_order"])
    builder.add_edge("researcher_policy", "join")
    builder.add_edge("researcher_order", "join")
    builder.add_edge("join", "writer")
    builder.add_edge("writer", END)
    return builder.compile(name="day12-parallel")


async def run_parallel(db: AsyncSession, payload: Day12ParallelRequest) -> dict[str, Any]:
    """练习 3：把 researcher 拆成"查政策 / 查订单"两条并行分支，join 后 writer 只调一次。

    ``Send`` 会把当前 state 复制给每个分支（各拿一份，互不覆盖），
    两条分支的 ``findings`` 靠 ``operator.add`` 在 join 处合并——这正是 Day 10
    讲的 reducer 在并行写时的价值。
    """
    _require_langgraph()
    from langchain_core.messages import HumanMessage

    cfg = await resolve_model(db, payload.model_config_id)
    llm = build_chat_model(cfg, temperature=0.0, run_name="day12-parallel")
    counter: dict[str, int] = {}
    app = _build_parallel_app(llm, _order_id_of(payload.question), counter)

    started = time.perf_counter()
    state = await app.ainvoke(
        {"messages": [HumanMessage(content=payload.question)], "findings": [], "events": []}
    )
    elapsed = int((time.perf_counter() - started) * 1000)

    events = list(state.get("events") or [])
    branches = [item for item in events if item.get("kind") == "expert"]
    writer_calls = len([item for item in branches if str(item.get("node")) == "writer"])
    messages = state.get("messages") or []

    return {
        "question": payload.question,
        "final_answer": _last_answer(messages) or "（模型没有产出最终回答）",
        "branches": branches,
        "findings": list(state.get("findings") or []),
        "writer_calls": writer_calls,
        "events": events,
        "messages": _jsonable(messages),
        "llm_calls": int(counter.get("expert_calls", 0)),
        "note": (
            "两条分支各拿一份 state 副本，findings 在 join 处由 operator.add 合并；"
            "writer 只被调用一次。代价是 token 直接翻倍——只有『确实独立 + 都耗时』才值得并行。"
        ),
        "graph": _graph_structure(
            app,
            labels={
                ("__start__", "researcher_policy"): "Send（扇出）",
                ("__start__", "researcher_order"): "Send（扇出）",
                ("researcher_policy", "join"): "静态边：自动等待",
                ("researcher_order", "join"): "静态边：自动等待",
                ("join", "writer"): "join 只打包，不决策",
                ("writer", "__end__"): "终点",
            },
        ),
        "model": cfg.model_name,
        "elapsed_ms": elapsed,
    }


# ============================================================
# 7. 静态概念数据接口
# ============================================================


async def mapping() -> dict[str, Any]:
    """Day 12 概念表：三堵墙 / 三角色 / 交接方式 / 状态三层 / 三层终止 / 五个坑 / 选型 / 成本。"""
    return {
        "walls": WALLS,
        "roles": ROLES,
        "handoff_modes": HANDOFF_MODES,
        "state_layers": STATE_LAYERS,
        "terminations": TERMINATIONS,
        "pitfalls": PITFALLS,
        "decision_table": DECISION_TABLE,
        "cost_table": COST_TABLE,
        "checklist": CHECKLIST,
        "presets": PRESETS,
        "install_hint": INSTALL_HINT,
        "langgraph_version": await asyncio.to_thread(_version, "langgraph"),
    }


__all__ = [
    "CHECKLIST",
    "COST_TABLE",
    "DECISION_TABLE",
    "DEFAULT_MAX_HANDOFF",
    "HANDOFF_MODES",
    "INSTALL_HINT",
    "PITFALLS",
    "PRESETS",
    "REFUND_LOG",
    "ROLES",
    "ROSTER_FULL",
    "ROSTER_MIN",
    "STATE_LAYERS",
    "TERMINATIONS",
    "WALLS",
    "mapping",
    "resume_full",
    "run_parallel",
    "run_single",
    "run_team",
    "start_full",
]
