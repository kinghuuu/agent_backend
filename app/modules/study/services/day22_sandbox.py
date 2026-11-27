"""Day 22 代码执行沙箱：让 Agent 会算、会跑、但跑不出笼子。

课程 day22 的七段脚本 + 三道练习在这里收成十个实验：

    为什么要沙箱    → ``why_demo``        （01：模型心算错 + exec 黑名单纸糊的）
    AST 筛查        → ``ast_demo``        （02：lint 不是 jail，四种绕过漏一半）
    L1 进程沙箱     → ``l1_demo``         （03：子进程 + rlimit + 独立 cwd + 组杀）
    网络与文件      → ``netfs_demo``      （04：软断网被 _socket 绕过，L1 的墙不存在）
    L2 容器         → ``container_demo``  （05：九个安全参数逐条对应一条威胁）
    代码循环        → ``loop_demo``       （06：写代码→执行→看报错→改代码）
    生产形态        → ``production_demo`` （07：画像 / 缓存 / 池化 / 审计 / 指标）
    接回 Day 21     → ``bridge_demo``     （练习 1：s3 换真执行器，stdout 字符串解析）
    压测矩阵        → ``pressure_demo``   （练习 2/3：十条攻击 × L1/L2 + 同类错误升级）

与课程示例的工程差异（都是为了在本仓库可跑、可测、可复现）：

- 课程 01 的 ``__import__('os').getcwd()`` 写在 ``__builtins__={}`` 的 exec 里其实会抛
  ``NameError``；逃逸链里的 ``print`` 同样不存在，必须从重建的 builtins 里取。本模块用
  安全探针复现**真实**行为并把这两处写进 ``corrections``，不照抄课程的错误输出；
- 课程 03 建议用 ``preexec_fn`` 设 rlimit，但 FastAPI/uvicorn 是多线程环境，官方明确说明
  它 fork 后不安全（可能死锁）。这里改为 runner 脚本：子进程在自己的进程里先设限再 exec；
- 内存限制用 ``RLIMIT_DATA`` 而不是 ``RLIMIT_AS``（后者让 numpy/pandas 导入即失败）；
  Windows 没有 ``resource``，显式跳过内存/CPU 硬限实验，只用 wall timeout 兜底；
- fork bomb 只在回放里展示：真跑会拖垮本机，生产用容器 ``--pids-limit`` 才是正解；
- 课程 05 的容器探针默认只生成命令 + 回放实测结果；``live=true`` 且 docker 可用时才真跑
  一条纯计算探针，不会挂宿主机目录、不会拉依赖、不会访问网络；
- 课程 06 的 coder 用真实模型；这里用脚本化回答 + 假 runner，零 token、毫秒级、可单测。
  错误回填保留课程要求的五要素（类型 / 消息 / 出错行 / 行号 / 失败前 stdout）；
- 所有接口默认不碰用户文件、不联网、不调模型；只有显式 ``mode=live`` 才起子进程，
  且子进程用白名单环境变量 + 临时 cwd + 超时 + 进程组清理。
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any

from app.modules.study.schemas.study import (
    Day22AstRequest,
    Day22BridgeRequest,
    Day22ContainerRequest,
    Day22L1Request,
    Day22LoopRequest,
    Day22NetfsRequest,
    Day22ProductionRequest,
    Day22PressureRequest,
    Day22WhyRequest,
)

POSIX = os.name == "posix"
MAX_OUTPUT = 8000
IMAGE = "python:3.11-slim"
SIGXCPU = getattr(signal, "SIGXCPU", -999)
SIGXFSZ = getattr(signal, "SIGXFSZ", -999)
# Windows 没有 SIGKILL 常量（getattr 为 None），组杀时用 9 兜底
SIGKILL = getattr(signal, "SIGKILL", 9)

# ============================================================
# 0. 静态概念数据（全部来自课程原文，个别处标明修正）
# ============================================================

WHY_CALC: list[dict[str, str]] = [
    {"task": "2 + 3", "accuracy": "几乎 100%", "reason": "训练里见过太多次，是记忆不是计算"},
    {"task": "1234 × 5678", "accuracy": "70%~90%", "reason": "见过类似模式能蒙对，但没有进位机制"},
    {"task": "1,234,567.89 × 0.0735", "accuracy": "崩", "reason": "千分位 + 小数 + 5 位有效数字，无训练分布覆盖"},
    {"task": "10 万行 CSV 聚合", "accuracy": "0%", "reason": "它连「10 万行」是什么都没法持有"},
]

DIVISION: list[dict[str, str]] = [
    {"layer": "旧分工", "who": "LLM 想 + LLM 算", "verdict": "心算错是静默失败：HTTP 200、格式漂亮、数字错"},
    {"layer": "新分工·写", "who": "LLM 写代码（run_python 是图灵完备的通用工具）", "verdict": "它只负责表达意图"},
    {"layer": "新分工·算", "who": "解释器执行，产出 stdout", "verdict": "精确表示、循环、变量都归 CPU"},
    {"layer": "新分工·边界", "who": "内核（namespace / cgroup / seccomp）", "verdict": "你不再控制它能做什么，只能限制它能碰到什么"},
]

ROUTING: list[dict[str, str]] = [
    {"scene": "两个已知数字比大小", "how": "纯代码判据（Day 21 的 expect / verify）", "why": "不需要 LLM"},
    {"scene": "单步四则运算", "how": "eval + 白名单 AST", "why": "轻量，但要走 AST 筛查"},
    {"scene": "多步计算 / 表格 / 统计", "how": "沙箱执行 pandas", "why": "今天的主场"},
    {"scene": "画图 / 生成文件", "how": "沙箱 + 产物回收", "why": "白名单后缀 + 大小上限"},
    {"scene": "装包 / 联网 / 长时间训练", "how": "默认拒绝", "why": "供应链投毒与资源耗尽的主入口"},
]

THREATS: list[dict[str, str]] = [
    {"no": "T1", "threat": "读敏感文件", "code": "open('.env') / os.environ", "gate": "文件系统隔离（只读挂载 / 容器）"},
    {"no": "T2", "threat": "外泄数据", "code": "socket / requests.post", "gate": "网络隔离（--network none）"},
    {"no": "T3", "threat": "篡改/删除数据", "code": "df.to_csv(原路径) / os.remove", "gate": "只读输入 + 独立可写目录"},
    {"no": "T4", "threat": "CPU 被占满", "code": "while True: pass", "gate": "RLIMIT_CPU + wall timeout"},
    {"no": "T5", "threat": "内存被吃光", "code": "[0]*10**9", "gate": "cgroup memory.max / RLIMIT_DATA"},
    {"no": "T6", "threat": "进程爆炸", "code": "os.fork() 循环", "gate": "RLIMIT_NPROC / --pids-limit"},
    {"no": "T7", "threat": "逃逸到宿主", "code": "内核漏洞 / --privileged", "gate": "容器 / microVM + 最小权限（唯一挡不住的一类）"},
]

LEVELS: list[dict[str, str]] = [
    {"level": "L0", "impl": "同进程 exec()", "cold": "0ms", "pandas": "能", "escape": "极低（等于没有）", "cost": "0", "when": "只跑自己写的、无外部输入的代码"},
    {"level": "L1", "impl": "子进程 + rlimit + 独立 cwd", "cold": "~20ms", "pandas": "能", "escape": "中：无命名空间，能读全盘、能联网", "cost": "极低", "when": "内部可信代码 / 教学 / 单机工具"},
    {"level": "L2", "impl": "容器（namespace + cgroup + seccomp）", "cold": "300ms~1s", "pandas": "能", "escape": "高：需内核 0day", "cost": "低", "when": "生产默认"},
    {"level": "L3", "impl": "microVM / gVisor", "cold": "125ms~300ms", "pandas": "能（gVisor 慢 10~50%）", "escape": "极高", "cost": "中", "when": "多租户 SaaS、不可信第三方代码"},
    {"level": "L4", "impl": "WASM（Pyodide）", "cold": "~50ms（预热 ~5ms）", "pandas": "有但纯 wasm，约 1/3 性能", "escape": "几乎不可能（无 syscall）", "cost": "极低", "when": "浏览器端纯计算"},
]

DECISION_TREE: list[dict[str, str]] = [
    {"source": "我自己写的固定脚本", "level": "L1 够了", "note": "没有外部输入就没有注入面"},
    {"source": "模型生成，输入只有内部结构化数据", "level": "L2（默认）", "note": "模型代码本身不可信"},
    {"source": "模型生成，输入含用户自由文本", "level": "L2 + 断网 + 只读挂载", "note": "输入一定被污染"},
    {"source": "多租户 / 客户上传代码", "level": "L3", "note": "要防彼此与内核漏洞"},
    {"source": "只要算数，不要文件不要网", "level": "L4", "note": "最快最省"},
]

GATES: list[dict[str, str]] = [
    {"no": "①", "gate": "静态筛查 AST", "blocks": "老实写错的 import os、手滑的 open('.env')", "cannot": "字符串拼接 / getattr / __subclasses__ / 编码后的 exec"},
    {"no": "②", "gate": "进程边界", "blocks": "崩溃带走主进程、cwd 污染、孤儿进程（配合组杀）", "cannot": "越权读文件（L1）、T7 逃逸"},
    {"no": "③", "gate": "资源配额", "blocks": "T4/T5/T6 三类耗尽", "cannot": "低频慢速滥用（每秒 fork 一次跑一天）"},
    {"no": "④", "gate": "网络与文件系统", "blocks": "T1/T2/T3", "cannot": "通过已允许的输出目录外泄"},
    {"no": "⑤", "gate": "输出与回收", "blocks": "日志爆盘、孤儿进程", "cannot": "算出有害但正确的结论（那是意图层的事）"},
]

CONTAINER_ARGS: list[dict[str, str]] = [
    {"arg": "--network none", "blocks": "T2", "without": "一行 requests.post 把 .env 发走"},
    {"arg": "--read-only", "blocks": "T3", "without": "能改容器内 python 库，下次执行被污染"},
    {"arg": "--tmpfs /tmp:rw,noexec,nosuid,size=64m", "blocks": "T3/T7", "without": "能在 /tmp 落地并执行二进制"},
    {"arg": "--memory 512m --memory-swap 512m", "blocks": "T5", "without": "只限 memory 不限 swap，换页到磁盘继续吃"},
    {"arg": "--cpus 0.5", "blocks": "T4", "without": "跑满 CPU，同机请求 p99 暴涨"},
    {"arg": "--pids-limit 64", "blocks": "T6", "without": "fork bomb 拖死整台机器"},
    {"arg": "--cap-drop=ALL", "blocks": "T7", "without": "默认 14 个 cap，含 CAP_NET_RAW"},
    {"arg": "--security-opt=no-new-privileges", "blocks": "T7", "without": "镜像里的 setuid 程序可提权"},
    {"arg": "--user 65534:65534", "blocks": "T7", "without": "容器内 root + 内核漏洞 = 宿主 root"},
    {"arg": "-v /in:ro / -v /out:rw", "blocks": "T1/T3", "without": "输入被改，源数据没了；挂载边界就是暴露边界"},
]

LIMITS: list[dict[str, str]] = [
    {"resource": "CPU 时间", "means": "RLIMIT_CPU（SIGXCPU）", "note": "只统计占用 CPU 的秒数，管不住 sleep / 等锁"},
    {"resource": "墙钟时间", "means": "communicate(timeout=)", "note": "防「不干活但占着并发槽」，两层都要有"},
    {"resource": "内存", "means": "RLIMIT_DATA 或 cgroup memory.max", "note": "别用 RLIMIT_AS：pandas 导入会报 static TLS block"},
    {"resource": "进程数", "means": "RLIMIT_NPROC / --pids-limit", "note": "容器里的 cgroup 更可靠"},
    {"resource": "单文件大小", "means": "RLIMIT_FSIZE + 产物白名单", "note": "防写爆磁盘、防可执行文件落地"},
    {"resource": "输出长度", "means": "头 1/4 + 尾 3/4 截断", "note": "尾部留 traceback；给模型的观察结果再压到 2KB"},
    {"resource": "stdin", "means": "DEVNULL", "note": "防 input() 永远卡住"},
]

METRICS: list[dict[str, str]] = [
    {"metric": "timeout_rate", "healthy": "< 3%", "abnormal": ">10% 说明没教它写快代码，或超时太紧"},
    {"metric": "oom_rate", "healthy": "< 1%", "abnormal": "模型爱用 iterrows / 全表转 list"},
    {"metric": "ast_deny_rate", "healthy": "< 5%", "abnormal": "突然升高 = 有人在试探，或注入在生效（最早信号）"},
    {"metric": "p95_duration_ms", "healthy": "与 profile 匹配", "abnormal": "脱节说明池化/缓存失效"},
]

PITFALLS: list[dict[str, str]] = [
    {"no": "1", "title": "把 AST 当安全边界", "symptom": "禁了 import os 和双下划线就以为安全，三条绕过就出去", "fix": "AST 只降噪；边界靠命名空间"},
    {"no": "2", "title": "RLIMIT_AS 让 pandas 起不来", "symptom": "256MB 报 cannot allocate memory in static TLS block", "fix": "用 RLIMIT_DATA 或 cgroup memory.max"},
    {"no": "3", "title": "只 kill(pid) 不杀进程组", "symptom": "multiprocessing 的 worker 变孤儿，占满机器一夜", "fix": "start_new_session + killpg；容器用 docker rm -f"},
    {"no": "4", "title": "忘了 wall clock", "symptom": "RLIMIT_CPU 管不住 sleep(3600)，并发槽被占一小时", "fix": "CPU 限 + wall timeout + stdin=DEVNULL，三层都要"},
    {"no": "5", "title": "输出不截断", "symptom": "print(df.to_string()) 把 80MB 塞进日志和下一轮 prompt", "fix": "头 1/4 + 尾 3/4，单次 8KB，给模型 2KB"},
    {"no": "6", "title": "跑通了就以为安全", "symptom": "算出正确但有害的结论、把隐私打印给不该看的人", "fix": "沙箱管能不能做，不管该不该做（意图层 + 出站审计）"},
]

CHEATSHEET: list[dict[str, str]] = [
    {"topic": "为什么不让它算", "content": "token 化后无变量、无循环、无精度", "key": "换分工，不是改 prompt"},
    {"topic": "三层分工", "content": "LLM 写程序 / 解释器算 / 内核管边界", "key": "谁也别越界"},
    {"topic": "沙箱目标", "content": "降低爆炸半径", "key": "炸不到你，不是不能做坏事"},
    {"topic": "不可信代码", "content": "模型生成 = 不可信 = 至少 L2", "key": "输入一定含用户文本"},
    {"topic": "五道闸", "content": "AST / 进程 / 资源 / 网络文件 / 输出回收", "key": "每道都要答出挡不住什么"},
    {"topic": "AST 定位", "content": "lint，不是 jail", "key": "拦手滑不拦攻击"},
    {"topic": "CPU 限", "content": "RLIMIT_CPU（SIGXCPU）", "key": "只管计算不管 sleep"},
    {"topic": "内存限", "content": "RLIMIT_DATA / cgroup", "key": "别用 RLIMIT_AS"},
    {"topic": "墙钟", "content": "communicate(timeout=) + DEVNULL", "key": "防 sleep 与 input"},
    {"topic": "进程回收", "content": "start_new_session + killpg", "key": "容器用 docker rm -f"},
    {"topic": "管道", "content": "必须 communicate()", "key": "64KB 写满即假超时"},
    {"topic": "容器必加", "content": "none / read-only / cap-drop / nobody / pids / memory&swap", "key": "少一个漏一个洞"},
    {"topic": "挂载边界", "content": "挂进去什么就暴露什么", "key": "别挂 .env 和 docker.sock"},
    {"topic": "seccomp 挡不住", "content": "内核漏洞、侧信道", "key": "要 gVisor / Firecracker"},
    {"topic": "代码循环", "content": "退出码 + traceback 就是 ground truth", "key": "不需要 LLM judge"},
    {"topic": "错误回填", "content": "类型/消息/出错行/行号/失败前 stdout", "key": "最后一项最容易漏"},
    {"topic": "超时反馈", "content": "换算法，不是再试一次", "key": "同类两次升硬约束"},
    {"topic": "循环上限", "content": "3 次，超了回 replanner", "key": "是计划问题不是执行问题"},
    {"topic": "缓存", "content": "代码 hash + 输入 hash；只缓存确定性", "key": "有 random/now/网络别缓存"},
    {"topic": "池化", "content": "污染的容器直接丢弃", "key": "否则跨请求泄漏"},
    {"topic": "最值钱指标", "content": "ast_deny_rate", "key": "突然升高就是攻击信号"},
]

AUDIT_FIELDS: list[str] = [
    "run_id", "tenant_id", "user_id", "thread_id",
    "code_hash", "code_len", "imports",
    "exit_code", "hit_limit", "timed_out",
    "duration_ms", "stdout_bytes", "truncated",
    "artifacts", "ast_denied",
    "image", "mem_mb", "cpu_s", "network",
]

TIMEOUT_PROFILES: list[dict[str, Any]] = [
    {"name": "quick_calc", "wall": 3, "mem_mb": 256, "cpu_s": 2, "desc": "单步计算 / 小表"},
    {"name": "data_analysis", "wall": 30, "mem_mb": 1024, "cpu_s": 25, "desc": "多步统计"},
    {"name": "plot", "wall": 15, "mem_mb": 512, "cpu_s": 10, "desc": "画图 + 产物回收"},
    {"name": "reject", "wall": 0, "mem_mb": 0, "cpu_s": 0, "desc": "装包 / 训练 / 爬虫，直接拒"},
]

# runner 脚本：在子进程内部先设限，再 exec 用户代码。
# 不用 preexec_fn —— 它在 fork 后、exec 前的多线程不安全窗口里运行（FastAPI 就是多线程）。
_RUNNER_SOURCE = '''
"""L1 沙箱 runner：在子进程里设资源限制，然后执行用户代码。"""
import os
import sys

def limit(cpu_s, mem_mb, fsize_mb):
    """POSIX 上设硬限；Windows 没有 resource，只留 wall timeout 兜底。"""
    try:
        import resource
    except ImportError:
        return "无 resource：仅 wall timeout"
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 1))
    resource.setrlimit(resource.RLIMIT_DATA, (mem_mb * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    return "RLIMIT_CPU/DATA/FSIZE/NOFILE/CORE"

if __name__ == "__main__":
    path, cpu_s, mem_mb = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    mode = limit(cpu_s, mem_mb, 8)
    sys.stdin = open(os.devnull)
    code = open(path, encoding="utf-8").read()
    namespace = {"__name__": "__main__", "__file__": path, "SANDBOX_LIMITS": mode}
    try:
        exec(compile(code, path, "exec"), namespace)
    except SystemExit:
        pass
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(1)
'''

# ============================================================
# 1. 概念数据接口
# ============================================================


def _version(name: str) -> str:
    """读取依赖版本；缺包时返回空串，不影响静态概念表。"""
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return ""


def mapping() -> dict[str, Any]:
    """返回 Day 22 全部静态概念数据（不调模型、不起容器、不写文件）。"""
    return {
        "why_calc": WHY_CALC,
        "division": DIVISION,
        "routing": ROUTING,
        "threats": THREATS,
        "levels": LEVELS,
        "decision_tree": DECISION_TREE,
        "gates": GATES,
        "container_args": CONTAINER_ARGS,
        "limits": LIMITS,
        "metrics": METRICS,
        "pitfalls": PITFALLS,
        "cheatsheet": CHEATSHEET,
        "audit_fields": AUDIT_FIELDS,
        "timeout_profiles": TIMEOUT_PROFILES,
        "environment": {
            "platform": f"{sys.platform} / Python {sys.version.split()[0]}",
            "resource": "RLIMIT 可用" if POSIX else "无 resource，仅 wall timeout",
            "docker": "可用" if shutil.which("docker") else "不可用（容器实验只回放）",
            "python": _version("pydantic"),
        },
    }


# ============================================================
# 2. 实验 01：两类失败（课程 01）
# ============================================================

_EXACT_TAX = round(1234567.89 * 0.0735, 2)
_WRONG_ANSWER = """税额计算如下：
1,234,567.89 × 7.35% = 1,234,567.89 × 0.0735 = 90,740.24 元
所以税额为 90,740.24 元。"""


def why_demo(payload: Day22WhyRequest) -> dict[str, Any]:
    """两种失败各跑一次：心算的静默错误 + exec 黑名单的逃逸链（安全探针）。"""
    safe_escape = '''
c = ().__class__.__base__.__subclasses__()
f = [x for x in c if x.__name__ == "catch_warnings"][0]
b = f.__init__.__globals__["__builtins__"]
b["print"]("拿回 __import__:", b["callable"](b["__import__"]))
b["print"]("当前进程 pid > 0:", b["__import__"]("os").getpid() > 0)
'''
    cases = [
        {
            "name": "以为清空 builtins 就安全：直接 __import__",
            "code": "__import__('os').getcwd()",
            "globals": '{"__builtins__": {}}',
            "actual": "NameError: name '__import__' is not defined",
            "escaped": False,
            "lesson": "课程称这行能取到 cwd，实际 CPython 里连 __import__ 都没有；它恰好说明黑名单记账不可靠。",
        },
        {
            "name": "链式取 builtins（不经危险函数字面量）",
            "code": safe_escape.strip(),
            "globals": '{"__builtins__": {}}',
            "actual": "拿回 __import__: True / 当前进程 pid > 0: True",
            "escaped": True,
            "lesson": "().__class__.__base__.__subclasses__() 找到 catch_warnings，沿 __globals__ 把真 builtins 拿回来。",
        },
    ]
    # 真的执行第二条安全探针：只取 pid 布尔值，不读文件、不联网、不做任何副作用
    # 重定向 stdout，避免教学探针把后端日志刷脏（builtins 里的 print 绕过了局部捕获）
    import contextlib
    import io

    buffer = io.StringIO()
    try:
        namespace: dict[str, Any] = {"__builtins__": {}}
        with contextlib.redirect_stdout(buffer):
            exec(compile(safe_escape, "<day22-safe-probe>", "exec"), namespace)
        probe = f"实际执行：逃逸链成立（{buffer.getvalue().strip().replace(chr(10), ' / ')}）"
    except BaseException as exc:  # 环境差异时不影响教学结论
        probe = f"实际执行失败：{type(exc).__name__}: {exc}"
    return {
        "question": payload.question,
        "exact_value": _EXACT_TAX,
        "llm_answer": _WRONG_ANSWER,
        "llm_wrong": "90,740.24" in _WRONG_ANSWER,
        "exec_cases": cases,
        "corrections": [
            "课程实验 B 第一行 __import__('os').getcwd() 在 __builtins__={} 下会 NameError，不是能取到 cwd。",
            "课程逃逸链里的 print 同样不在空 builtins 里，必须用 b['print'](...)；本文的安全探针按真实行为修正。",
        ],
        "notes": [
            "心算错是静默失败：HTTP 200、格式漂亮、数字错，规则与监控都拦不住。",
            f"正确答案是 {_EXACT_TAX:.2f} 元；差 0.5 元正是最危险的地方——人眼一眼看不出来。",
            probe,
            "exec + 清空 builtins + 黑名单 = 心理安慰；真正的边界是命名空间，不是过滤器。",
        ],
    }


# ============================================================
# 3. 实验 02：AST 筛查与三种绕过（课程 02）
# ============================================================

DENY_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "ctypes", "importlib",
    "pty", "multiprocessing", "requests", "urllib", "http", "ftplib",
    "telnetlib", "pickle", "marshal",
}
DENY_CALLS = {
    "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "input",
}
REVIEW_CALLS = {"open", "print"}


class Guard(ast.NodeVisitor):
    """AST 访客：记录被禁模块、被禁调用与需要审计的调用。"""

    def __init__(self) -> None:
        self.deny: list[str] = []
        self.review: list[str] = []
        self.imports: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        """逐个 import 检查根模块名；`import os.path` 也算 os。"""
        for alias in node.names:
            root = alias.name.split(".")[0]
            self.imports.append(alias.name)
            if root in DENY_MODULES:
                self.deny.append(f"L{node.lineno}: import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """`from os import environ` 与 `import os` 同等对待。"""
        root = (node.module or "").split(".")[0]
        self.imports.append(node.module or "")
        if root in DENY_MODULES:
            self.deny.append(f"L{node.lineno}: from {node.module} import ...")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """按调用名查黑名单/审计名单；只识别字面量函数名，拼出来的名字看不见。"""
        func = node.func
        name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else None)
        if name in DENY_CALLS:
            self.deny.append(f"L{node.lineno}: 调用 {name}()")
        elif name in REVIEW_CALLS:
            self.review.append(f"L{node.lineno}: 调用 {name}()")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """拦所有 dunder 属性访问，包括 __class__ / __globals__ / __subclasses__。"""
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.deny.append(f"L{node.lineno}: 访问 {node.attr}")
        self.generic_visit(node)


def screen(code: str) -> tuple[bool, list[str], list[str], list[str]]:
    """静态筛查：返回 (是否放行, 拦截原因, 需审计记录, 导入列表)。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"语法错误：{exc}"], [], []
    guard = Guard()
    guard.visit(tree)
    return not guard.deny, guard.deny, guard.review, guard.imports


_NORMAL_CASES: list[tuple[str, str, str]] = [
    ("normal", "正常 pandas", "import pandas as pd\ndf = pd.read_csv('in.csv')\nprint(df.mean())"),
    ("write", "手滑写文件", "import pandas as pd\ndf.to_csv('/data/orders.csv')"),
    ("import_os", "import os", "import os\nos.remove('x')"),
    ("env", "读环境变量", "import os\nprint(os.environ['OPENAI_API_KEY'])"),
    ("dunder", "dunder 逃逸", "print(().__class__.__base__.__subclasses__())"),
]

_BYPASS_CASES: list[tuple[str, str]] = [
    ("字符串拼接 __import__", "m = __builtins__['__imp' + 'ort__']\nm('os').system('id')"),
    ("getattr 拼函数名", "f = getattr(__builtins__, 'ev' + 'al')"),
    # 课程原样写的是 exec(bytes.fromhex(...))，但 exec 是字面量调用，会被 DENY_CALLS 拦下；
    # 要复现「编码后的 exec 能绕」，连调用名都得拼出来 —— 这才是真实的绕过姿势。
    ("hex 编码 exec（调用名也拼出来）", "b = __builtins__\nf = b['ex' + 'ec']\nf(bytes.fromhex('7072696e74283129').decode())"),
    ("链式取 builtins", "x = [c for c in ().__class__.__base__.__subclasses__() if c.__name__ == 'catch_warnings'][0]"),
]


def ast_demo(payload: Day22AstRequest) -> dict[str, Any]:
    """筛查课程五个样本 + 四种绕过，把「哪条漏了」摆到台面上。"""
    selected = [c for c in _NORMAL_CASES if payload.case in ("all", c[0])]
    cases: list[dict[str, Any]] = []
    for key, name, code in selected:
        ok, deny, review, imports = screen(code)
        cases.append({
            "group": "sample", "key": key, "name": name, "code": code,
            "allowed": ok, "deny": deny, "review": review, "imports": imports,
        })
    if payload.case in ("all", "bypass"):
        for name, code in _BYPASS_CASES:
            ok, deny, review, _ = screen(code)
            cases.append({
                "group": "bypass", "key": name, "name": name, "code": code,
                "allowed": ok, "deny": deny, "review": review, "imports": [],
            })
    write_case = next((c for c in cases if c["key"] == "write"), None)
    return {
        "cases": cases,
        "bypass_leaked": sum(c["group"] == "bypass" and c["allowed"] for c in cases),
        "write_file_blocked": write_case["allowed"] is False if write_case else False,
        "notes": [
            "df.to_csv('/data/orders.csv') 没被拦：to_csv 是普通方法名，AST 看不到参数语义 —— 要靠只读挂载。",
            "四种绕过里两种被放过（字符串拼接、拼调用名的 hex exec）；getattr 和 dunder 字面量只是恰好命中规则。",
            "课程把 hex 那条写成 exec(bytes.fromhex(...))，但 exec 是字面量调用会被 DENY_CALLS 拦下；这里改成「连调用名也拼出来」才是真绕过。",
            "AST 是 lint 不是 jail：给模型即时反馈、拦 80% 手滑、记录能力清单，别用它决定能不能跑。",
        ],
    }


# ============================================================
# 4. 实验 03：L1 子进程沙箱（课程 03）
# ============================================================


@dataclass
class CodeResult:
    """一次代码执行的可观察结果：退出码、命中哪条限制、截断后的输出。"""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    duration_ms: int = 0
    timed_out: bool = False
    truncated: bool = False
    hit_limit: str = ""
    skipped: str = ""

    def as_dict(self) -> dict[str, Any]:
        """转成接口可直接返回的结构。"""
        return {
            "ok": self.ok, "stdout": self.stdout, "stderr": self.stderr,
            "exit_code": self.exit_code, "duration_ms": self.duration_ms,
            "timed_out": self.timed_out, "truncated": self.truncated,
            "hit_limit": self.hit_limit, "skipped": self.skipped,
        }


def truncate(text: str, limit: int = MAX_OUTPUT) -> tuple[str, bool]:
    """头 1/4 + 尾 3/4 截断：尾部通常是 traceback，比头部更有用。"""
    if len(text) <= limit:
        return text, False
    head = text[: limit // 4]
    tail = text[-limit * 3 // 4 :]
    return f"{head}\n...[截断 {len(text) - limit} 字符]...\n{tail}", True


def _kill_group(process: subprocess.Popen[bytes]) -> None:
    """超时后杀整组：只杀直接子进程会留下 fork/multiprocessing 的孤儿。"""
    try:
        if POSIX:
            os.killpg(os.getpgid(process.pid), SIGKILL)
        else:
            process.kill()  # Windows 无通用的 killpg；容器边界更可靠
    except (ProcessLookupError, OSError):
        pass


def run_python_live(
    code: str,
    wall_timeout: int = 3,
    mem_mb: int = 256,
    cpu_s: int = 2,
    extra_env: dict[str, str] | None = None,
) -> CodeResult:
    """真起一个子进程跑代码：白名单环境 + 临时 cwd + rlimit + 超时组杀。"""
    start = time.time()
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": "",
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update(extra_env or {})
    with tempfile.TemporaryDirectory(prefix="day22_sbx_") as workdir:
        runner = os.path.join(workdir, "runner.py")
        target = os.path.join(workdir, "user_code.py")
        with open(runner, "w", encoding="utf-8") as handle:
            handle.write(_RUNNER_SOURCE)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(code)
        flags: dict[str, Any] = {
            "cwd": workdir, "env": env,
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE, "stderr": subprocess.PIPE,
        }
        if POSIX:
            flags["start_new_session"] = True
        else:
            flags["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", runner, target, str(cpu_s), str(mem_mb)], **flags
        )
        hit = ""
        try:
            raw_out, raw_err = process.communicate(timeout=wall_timeout)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            raw_out, raw_err = process.communicate()
            hit = "timeout"
        out, cut_out = truncate(raw_out.decode("utf-8", "replace"))
        err, cut_err = truncate(raw_err.decode("utf-8", "replace"))
        if not hit:
            if process.returncode == -SIGXCPU:
                hit = "cpu"
            elif process.returncode == -SIGXFSZ:
                hit = "fsize"
            elif process.returncode == -SIGKILL and "MemoryError" in err:
                hit = "mem"
        return CodeResult(
            ok=process.returncode == 0, stdout=out, stderr=err,
            exit_code=process.returncode,
            duration_ms=int((time.time() - start) * 1000),
            timed_out=hit == "timeout", truncated=cut_out or cut_err, hit_limit=hit,
        )


_L1_CASES: list[dict[str, str]] = [
    {"key": "normal", "name": "正常计算", "code": "print(1234567.89 * 0.0735)"},
    {"key": "cpu", "name": "CPU 死循环", "code": "while True: pass"},
    {"key": "sleep", "name": "sleep 骗时间", "code": "import time; time.sleep(60)"},
    {"key": "mem", "name": "内存炸弹", "code": "x = []\nwhile True: x.append('a' * 10**6)"},
    {
        "key": "fork", "name": "fork bomb（限量演示）",
        "code": "import os\nfor _ in range(100):\n    try: os.fork()\n    except OSError as exc: print('fork 失败:', exc)",
    },
    {"key": "env", "name": "读环境变量（L1 挡不住）", "code": "import os\nprint(os.environ.get('DEMO_SECRET', '<空>'))"},
]

_L1_SIMULATED: dict[str, dict[str, Any]] = {
    "normal": {"ok": True, "exit_code": 0, "duration_ms": 412, "hit_limit": "", "stdout": "90740.743915\n", "stderr": ""},
    "cpu": {"ok": False, "exit_code": -SIGXCPU if POSIX else -9, "duration_ms": 2138, "hit_limit": "cpu", "stdout": "", "stderr": ""},
    "sleep": {"ok": False, "exit_code": -9, "duration_ms": 3001, "hit_limit": "timeout", "stdout": "", "stderr": ""},
    "mem": {"ok": False, "exit_code": -9, "duration_ms": 1876, "hit_limit": "mem", "stdout": "", "stderr": "MemoryError\n"},
    "fork": {"ok": True, "exit_code": 0, "duration_ms": 623, "hit_limit": "", "stdout": "fork 失败: [Errno 11] Resource temporarily unavailable\n（×97 次）", "stderr": ""},
    "env": {"ok": True, "exit_code": 0, "duration_ms": 385, "hit_limit": "", "stdout": "sk-demo-canary\n", "stderr": ""},
}


def l1_demo(payload: Day22L1Request) -> dict[str, Any]:
    """回放或真跑 L1 沙箱；fork bomb 与 Windows 内存炸弹只回放，避免拖垮本机。"""
    cases = [c for c in _L1_CASES if payload.case in ("all", c["key"])]
    if payload.custom_code:
        ok, deny, review, _ = screen(payload.custom_code)
        if not ok:
            return {
                "platform": f"{sys.platform} / Python {sys.version.split()[0]}",
                "mode": payload.mode,
                "cases": [{
                    "name": "自定义代码", "code": payload.custom_code, "ok": False,
                    "exit_code": -1, "hit_limit": "ast", "stdout": "",
                    "stderr": "AST 筛查未通过：" + "；".join(deny),
                    "skipped": "", "duration_ms": 0,
                }],
                "limits": {"ast": "拦下，未进入子进程"},
                "notes": ["AST 只做降噪；它拦下的是明显手滑，不构成安全边界。"],
            }
        cases = [{"key": "custom", "name": "自定义代码", "code": payload.custom_code}]

    rows: list[dict[str, Any]] = []
    for item in cases:
        key = item["key"]
        base = {"name": item["name"], "code": item["code"]}
        skipped = ""
        if key == "fork" and payload.mode == "live":
            skipped = "真 fork 会拖垮本机：生产请用容器 --pids-limit，这里只回放"
        elif key == "mem" and payload.mode == "live" and not POSIX:
            skipped = "Windows 无 RLIMIT_DATA，真跑可能吃光内存，这里只回放"
        if skipped:
            row = {**base, **_L1_SIMULATED[key], "skipped": skipped}
        elif payload.mode == "simulate":
            row = {**base, **_L1_SIMULATED[key], "skipped": ""}
        else:
            result = run_python_live(
                item["code"], wall_timeout=payload.wall_timeout,
                mem_mb=payload.mem_mb, cpu_s=payload.cpu_s,
                extra_env={"DEMO_SECRET": "sk-demo-canary"},
            )
            row = {**base, **result.as_dict()}
        rows.append(row)

    return {
        "platform": f"{sys.platform} / Python {sys.version.split()[0]}",
        "mode": payload.mode,
        "cases": rows,
        "limits": {
            "cpu": "RLIMIT_CPU → SIGXCPU" if POSIX else "无 RLIMIT_CPU，仅 wall timeout",
            "memory": "RLIMIT_DATA（避开 RLIMIT_AS 坑）" if POSIX else "无 RLIMIT_DATA，已跳过内存实验",
            "wall": f"communicate(timeout={payload.wall_timeout})",
            "stdin": "DEVNULL",
            "process_group": "start_new_session + killpg" if POSIX else "CREATE_NEW_PROCESS_GROUP（子树可能残留）",
        },
        "notes": [
            "exit_code -24 是 SIGXCPU，-9 是 SIGKILL；把退出码映射成 hit_limit，模型才知道超时该换算法。",
            "RLIMIT_CPU 只算占用 CPU 的秒数，sleep(60) 得靠 wall timeout；两层都要有。",
            "最后一条证明 L1 挡不住读环境变量：子进程继承你的用户、环境、全盘权限。",
        ],
    }


# ============================================================
# 5. 实验 04：网络与文件系统 —— L1 的墙在哪（课程 04）
# ============================================================

NET_GUARD_CODE = '''import socket, builtins

class _Blocked(Exception):
    pass

def _die(*a, **kw):
    raise _Blocked("沙箱内禁止网络访问")

class _FakeSocket:
    def __init__(self, *a, **kw): _die()
    def __getattr__(self, n): _die()

socket.socket = _FakeSocket
socket.create_connection = _die
socket.getaddrinfo = _die

builtins.__import_orig__ = builtins.__import__

def _guarded_import(name, *a, **kw):
    if name.split(".")[0] in {"socket", "ssl", "http", "urllib", "requests", "ftplib"}:
        raise _Blocked(f"禁止导入网络模块 {name}")
    return builtins.__import_orig__(name, *a, **kw)

builtins.__import__ = _guarded_import
'''

_NET_PROBES: list[dict[str, str]] = [
    {"key": "socket", "name": "软断网（走 socket 模块）", "code": "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=1)"},
    {"key": "raw_socket", "name": "软断网的绕过（_socket）", "code": "import _socket\ns = _socket.socket()\nprint('拿到了 raw socket:', s.fileno())"},
    {"key": "host_read", "name": "读宿主机文件", "code": "import os\nprint(open(os.environ['CANARY_PATH']).read().strip())"},
    {"key": "host_write", "name": "写宿主机文件", "code": "import os\nopen(os.environ['CANARY_PATH'] + '.written', 'w').write('x')\nprint('写成功')"},
]

_NET_SIMULATED: dict[str, dict[str, Any]] = {
    "socket": {"ok": False, "last_line": "Blocked: 禁止导入网络模块 socket", "wall": "不存在：软断网只改 Python 层"},
    "raw_socket": {"ok": True, "last_line": "拿到了 raw socket: 3", "wall": "破了：_socket 是 C 扩展，绕过 patch"},
    "host_read": {"ok": True, "last_line": "docker-desktop（宿主机 hostname 被读到）", "wall": "不存在：L1 无命名空间"},
    "host_write": {"ok": True, "last_line": "写成功（写进了宿主机 /tmp）", "wall": "不存在：L1 无只读挂载"},
}


def netfs_demo(payload: Day22NetfsRequest) -> dict[str, Any]:
    """软断网 + 宿主文件读写探针：四条里两条墙根本不存在。"""
    selected = [p for p in _NET_PROBES if payload.probe in ("all", p["key"])]
    probes: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="day22_netfs_") as canary_dir:
        canary = os.path.join(canary_dir, "canary.txt")
        with open(canary, "w", encoding="utf-8") as handle:
            handle.write("HOST-CANARY-OK")
        for item in selected:
            base = {"name": item["name"], "code": item["code"]}
            if payload.mode == "simulate":
                row = {**base, **_NET_SIMULATED[item["key"]], "mode": "回放课程实测"}
            else:
                result = run_python_live(
                    f"{NET_GUARD_CODE}\n{item['code']}", wall_timeout=3,
                    extra_env={"CANARY_PATH": canary},
                )
                text = (result.stdout or result.stderr).strip().splitlines()
                row = {
                    **base, "ok": result.ok,
                    "last_line": text[-1][:160] if text else "(无输出)",
                    "wall": "实测：当前平台下的真实行为",
                    "mode": "live",
                    "exit_code": result.exit_code,
                }
            probes.append(row)
    return {
        "guard_code": NET_GUARD_CODE,
        "probes": probes,
        "broken": sum(
            p["ok"] and ("破了" in p["wall"] or "不存在" in p["wall"]) for p in probes
        ),
        "notes": [
            "软断网只替换 Python 的 socket 模块；ctypes 调 syscall 或 import _socket 就能绕过去。",
            "L1 没有命名空间，宿主机全盘可读可写；「输入只读」只是约定，用户代码可以 chmod 再改。",
            "该上容器了：再加 seccomp/chroot 都要 root，L1 补不出真正的墙。",
        ],
    }


# ============================================================
# 6. 实验 05：容器（课程 05）
# ============================================================

DOCKER_ARGS = [
    "--rm", "--network", "none", "--read-only",
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
    "--memory", "512m", "--memory-swap", "512m",
    "--cpus", "0.5", "--pids-limit", "64",
    "--cap-drop=ALL", "--security-opt=no-new-privileges",
    "--user", "65534:65534",
]

_CONTAINER_PROBES: list[dict[str, Any]] = [
    {"name": "读 .env", "code": "print(open('/in/.env').read())", "ok": True, "exit_code": 0, "hit": "-", "last": "OPENAI_API_KEY=sk-secret（主动挂进去的假密钥）"},
    {"name": "读 /etc/shadow", "code": "print(open('/etc/shadow').read())", "ok": False, "exit_code": 1, "hit": "-", "last": "PermissionError: [Errno 13] Permission denied"},
    {"name": "网络", "code": "import socket; socket.create_connection(('1.1.1.1', 80), timeout=2)", "ok": False, "exit_code": 1, "hit": "-", "last": "OSError: [Errno 101] Network is unreachable"},
    {"name": "写根目录", "code": "open('/x','w').write('y')", "ok": False, "exit_code": 1, "hit": "-", "last": "OSError: [Errno 30] Read-only file system: '/x'"},
    {"name": "正常 pandas", "code": "import pandas as pd; print(pd.__version__)", "ok": True, "exit_code": 0, "hit": "-", "last": "2.2.1"},
    {"name": "内存炸弹", "code": "x=[]\nwhile True: x.append(b'a'*10**6)", "ok": False, "exit_code": 137, "hit": "mem", "last": "(无输出)"},
]


def _docker_command(payload: Day22ContainerRequest) -> str:
    """按请求参数生成完整命令；内存与 swap 必须同值，否则限制等于没有。"""
    args = [a.replace("512m", f"{payload.mem_mb}m") for a in DOCKER_ARGS]
    args[args.index("0.5")] = str(payload.cpus)
    args[args.index("64")] = str(payload.pids_limit)
    mounts = ["-v", "$WORK:/in:ro", "-v", "$OUT:/out:rw", "-w", "/out"]
    if payload.mount_env:
        mounts.extend(["-v", "$ENV_FILE:/in/.env:ro"])
    return "docker run " + " ".join([*args, *mounts, IMAGE, "python", "/in/user_code.py"])


def _live_probe(payload: Day22ContainerRequest) -> dict[str, Any]:
    """docker 可用时真跑一条纯计算探针：不挂宿主目录、不联网、不装包。"""
    if not shutil.which("docker"):
        return {"status": "skipped", "reason": "PATH 上没有 docker，容器实验只回放"}
    args = [a.replace("512m", f"{payload.mem_mb}m") for a in DOCKER_ARGS]
    args[args.index("0.5")] = str(payload.cpus)
    args[args.index("64")] = str(payload.pids_limit)
    cmd = ["docker", "run", *args, IMAGE, "python", "-c", "print(1234567.89 * 0.0735)"]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=payload.wall_timeout + 10)
        return {
            "status": "ok" if done.returncode == 0 else "failed",
            "exit_code": done.returncode,
            "stdout": done.stdout.decode("utf-8", "replace")[:MAX_OUTPUT],
            "stderr": done.stderr.decode("utf-8", "replace")[:MAX_OUTPUT],
        }
    except subprocess.TimeoutExpired:
        return {"status": "failed", "reason": f"超过 {payload.wall_timeout + 10}s，可能正在拉镜像"}
    except OSError as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def container_demo(payload: Day22ContainerRequest) -> dict[str, Any]:
    """生成安全命令 + 回放六条探针，突出「挂载边界 = 暴露边界」。"""
    return {
        "command": _docker_command(payload),
        "args": CONTAINER_ARGS,
        "probes": _CONTAINER_PROBES,
        "docker_available": bool(shutil.which("docker")),
        "live_result": _live_probe(payload) if payload.live else None,
        "notes": [
            "第一条是刻意陷阱：.env 被读到，因为是主动挂进去的；生产只能挂本次任务需要的那一个数据文件。",
            "后五条全部正确拦住：PermissionError / Network unreachable / Read-only FS / OOM kill。",
            "seccomp 挡的是 syscall 编号，挡不住内核漏洞与侧信道；多租户要 gVisor 或 Firecracker。",
        ],
    }


# ============================================================
# 7. 实验 06：写代码 → 执行 → 看报错 → 改代码（课程 06）
# ============================================================


@dataclass
class FakeRun:
    """假 runner：按课程实测回填退出码、stdout、stderr 与命中的限制。"""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: int = 0
    hit_limit: str = ""


@dataclass
class LoopScript:
    """一段 coder 剧本：模型输出 + 对应的一次执行结果。"""

    code: str
    run: FakeRun


def extract_code(text: str) -> str:
    """从模型回答里剥出 ```python 围栏，拿不到就当全文是代码。"""
    match = re.search(r"```(?:python)?\s*(.+?)```", text or "", re.S)
    return (match.group(1) if match else text or "").strip()


def error_digest(run: FakeRun, code: str) -> dict[str, Any]:
    """把 40 行 traceback 压成五要素：类型 / 消息 / 出错行 / 行号 / 失败前 stdout。"""
    if run.hit_limit:
        msg = "执行超时（>5s），可能是算法复杂度太高或死循环" if run.hit_limit == "timeout" else f"命中限制 {run.hit_limit}"
        return {"type": run.hit_limit, "msg": msg, "line": None, "code_line": ""}
    err = run.stderr or ""
    line_match = re.search(r'File "[^"]*user_code\.py", line (\d+)', err)
    lines = [line for line in err.strip().splitlines() if line.strip()]
    exception = ""
    for line in reversed(lines):
        if re.match(r"^\w+(Error|Exception|Warning):", line.strip()):
            exception = line.strip()
            break
    code_line = ""
    if line_match:
        number = int(line_match.group(1))
        source = code.splitlines()
        if 0 < number <= len(source):
            code_line = source[number - 1].strip()
    return {
        "type": exception.split(":")[0] if exception else "Unknown",
        "msg": exception or (lines[-1] if lines else ""),
        "line": int(line_match.group(1)) if line_match else None,
        "code_line": code_line,
    }


def feedback_message(run: FakeRun, code: str, attempt: int, digest: dict[str, Any], strategy: str = "") -> str:
    """生成回填给模型的提示：最小错误 + 失败前 stdout（最容易漏的免费上下文）+ 换策略指令。"""
    parts = [
        f"执行失败（第 {attempt} 次）。",
        f"异常类型：{digest['type']}",
        f"错误信息：{digest['msg']}",
    ]
    if digest.get("line"):
        parts.append(f"出错行（第 {digest['line']} 行）：{digest['code_line']}")
    if run.stdout.strip():
        parts.append(f"失败前的输出：\n{run.stdout.strip()[:600]}")
    if strategy:
        parts.append(strategy)
    parts.append("请输出修正后的完整代码。")
    return "\n".join(parts)


_PANDAS_V1 = """```python
import pandas as pd
df = pd.read_csv('/in/data.csv')
print(df.columns.tolist())
total = df.append(df).groupby('city')['amount'].sum()
print(total)
```"""
_PANDAS_V2 = """```python
import pandas as pd
df = pd.read_csv('/in/data.csv')
total = pd.concat([df, df]).groupby('city')['amount'].sum()
print(total.to_dict())
```"""
_KEY_V1 = """```python
import pandas as pd
df = pd.read_csv('/in/data.csv')
print(df.columns.tolist())
print(df.groupby('cities')['amount'].sum())
```"""
_KEY_V2 = """```python
import pandas as pd
df = pd.read_csv('/in/data.csv')
print(df.groupby('city')['amount'].sum().to_dict())
```"""
_TIMEOUT_V1 = """```python
import pandas as pd
df = pd.read_csv('/in/data.csv')
for i, row in df.iterrows():
    result[i] = row['amount'] * 0.0735
print(result)
```"""
_TIMEOUT_V2 = """```python
import pandas as pd
df = pd.read_csv('/in/data.csv')
print((df['amount'] * 0.0735).sum())
```"""
_SAME_V1 = """```python
import pandas as pd
df = pd.read_csv('/in/data.csv')
print(df.append(df).head())
```"""
_SAME_V3 = """```python
import pandas as pd
df = pd.read_csv('/in/data.csv')
print(dir(df))
print(pd.concat([df, df]).head())
```"""

_PANDAS_RUN1 = FakeRun(
    ok=False, exit_code=1, duration_ms=780,
    stdout="['order_id', 'amount', 'city']\n",
    stderr=(
        'Traceback (most recent call last):\n'
        '  File "/in/user_code.py", line 4, in <module>\n'
        '    total = df.append(df).groupby(\'city\')[\'amount\'].sum()\n'
        "AttributeError: 'DataFrame' object has no attribute 'append'\n"
    ),
)
_PANDAS_RUN2 = FakeRun(ok=True, stdout="{'北京': 2580, '上海': 1980, '广州': 860}\n", duration_ms=612)
_KEY_RUN1 = FakeRun(
    ok=False, exit_code=1, stdout="['order_id', 'amount', 'city']\n",
    stderr='  File "/in/user_code.py", line 4, in <module>\nKeyError: \'cities\'\n',
)
_TIMEOUT_RUN1 = FakeRun(ok=False, exit_code=-9, duration_ms=5001, hit_limit="timeout")


def _script(scenario: str, escalate: bool) -> list[LoopScript]:
    """四套剧本：标准修复、失败前 stdout 救场、超时换算法、同类错误升级。"""
    if scenario == "pandas_append":
        return [
            LoopScript(_PANDAS_V1, _PANDAS_RUN1),
            LoopScript(_PANDAS_V2, _PANDAS_RUN2),
        ]
    if scenario == "key_error":
        return [
            LoopScript(_KEY_V1, _KEY_RUN1),
            LoopScript(_KEY_V2, FakeRun(ok=True, stdout="{'北京': 2580}\n")),
        ]
    if scenario == "timeout":
        scripts = [LoopScript(_TIMEOUT_V1, _TIMEOUT_RUN1)]
        if escalate:
            # 同类超时第二次：仍按原方案跑一次，让反馈里出现「换算法」的硬约束
            scripts.append(LoopScript(
                _TIMEOUT_V1,
                FakeRun(ok=False, exit_code=-9, duration_ms=5002, hit_limit="timeout"),
            ))
        scripts.append(LoopScript(_TIMEOUT_V2, FakeRun(ok=True, stdout="16695.0\n")))
        return scripts
    # same_error：前两次是同一个 AttributeError，第三次是否换策略由 escalate 决定
    third = _SAME_V3 if escalate else _SAME_V1
    third_run = (
        FakeRun(ok=True, stdout="  empty\n0  1\n") if escalate
        else FakeRun(ok=False, exit_code=1, stderr="AttributeError: 'DataFrame' object has no attribute 'append'\n")
    )
    return [
        LoopScript(_SAME_V1, _PANDAS_RUN1),
        LoopScript(_SAME_V1, FakeRun(ok=False, exit_code=1, stderr="AttributeError: 'DataFrame' object has no attribute 'append'\n")),
        LoopScript(third, third_run),
    ]


def loop_demo(payload: Day22LoopRequest) -> dict[str, Any]:
    """脚本化 coder + 假 runner：演示错误五要素回填与「同类两次换策略」。"""
    scripts = _script(payload.scenario, payload.escalate)
    attempts: list[dict[str, Any]] = []
    feedback: list[str] = []
    strategy_switches: list[str] = []
    seen: dict[str, int] = {}
    answer: str | None = None
    used = 0

    for index, script in enumerate(scripts[: payload.max_tries], start=1):
        used = index
        code = extract_code(script.code)
        digest = error_digest(script.run, code)
        strategy = ""
        error_type = digest["type"]
        if not script.run.ok:
            seen[error_type] = seen.get(error_type, 0) + 1
            if payload.escalate and seen[error_type] == 2:
                if error_type == "AttributeError":
                    strategy = "同类错误第二次：先 print(dir(obj)) 确认可用 API，不要再猜方法名。"
                elif error_type == "timeout":
                    strategy = "同类错误第二次：改用向量化（groupby / 直接列运算），禁止 iterrows。"
                elif error_type == "KeyError":
                    strategy = "同类错误第二次：先 print(df.columns) 看清真实列名。"
                else:
                    strategy = "同类错误第二次：换一条实现路径，不要重试同一个算法。"
                strategy_switches.append(strategy)
        message = (
            "" if script.run.ok
            else feedback_message(script.run, code, index, digest, strategy)
        )
        attempts.append({
            "attempt": index, "code": code, "ok": script.run.ok,
            "exit_code": script.run.exit_code, "duration_ms": script.run.duration_ms,
            "hit_limit": script.run.hit_limit,
            "stdout": script.run.stdout[:600], "stderr": script.run.stderr[:600],
            "digest": digest,
        })
        if message:
            feedback.append(message)
        if script.run.ok:
            answer = script.run.stdout.strip()
            break

    return {
        "attempts": attempts,
        "feedback": feedback,
        "answer": answer,
        "tries": used,
        "strategy_switches": strategy_switches,
        "notes": [
            "退出码 + traceback 是内核给的 ground truth：没有解释空间，所以不需要 LLM judge。",
            "失败前的 stdout 是免费上下文：看到 ['order_id','amount','city'] 就能一次改对 KeyError。",
            "超时反馈要写「改成向量化」而不是「再试一次」；同类错误第二次就该换策略。",
            "3 次仍失败说明目标本身可能不成立，回 Day 21 的 replanner，不是继续重试。",
        ],
    }


# ============================================================
# 8. 练习 1：把 Day 21 的 s3 换成真执行器
# ============================================================


def _parse_number(text: str) -> tuple[str, float]:
    """stdout 是字符串：这一步是 Day 02 结构化输出在沙箱场景的复现。"""
    cleaned = text.strip().splitlines()
    last = cleaned[-1] if cleaned else ""
    return "str", float(last)


def bridge_demo(payload: Day22BridgeRequest) -> dict[str, Any]:
    """生成三行代码 → L1 执行 → 解析 stdout → 交给 Day 21 的机器判据。"""
    from app.modules.study.services.day21_planning import PlanStep, verify

    code = (
        f"available = {payload.available}\n"
        f"days = float({payload.days})\n"
        f"per_day = {payload.per_day}\n"
        "total = available + days * per_day\n"
        "print(total)\n"
    )
    if payload.mode == "live":
        result = run_python_live(code, wall_timeout=3)
        stdout = result.stdout
        if not result.ok:
            stdout = result.stderr
    else:
        total = payload.available + float(payload.days) * payload.per_day
        stdout = f"{total}\n"
    raw_type, parsed = _parse_number(stdout)
    step = PlanStep(
        id="s3", goal="算可供总量", done_criteria="total = 库存 + 天数×日产能",
        expect={"path": "total", "op": payload.expect_op, "value": payload.expect_value},
    )
    passed, reason = verify(step, {"total": parsed})
    return {
        "code": code,
        "stdout": stdout,
        "raw_type": raw_type,
        "parsed_type": type(parsed).__name__,
        "parsed_value": parsed,
        "expect": step.expect,
        "passed": passed,
        "reason": reason,
        "notes": [
            "Day 21 的 s3 原本让 LLM 心算，现在由解释器算、stdout 给出结果。",
            "stdout 是字符串，必须先 float() 再喂 verify —— 这就是 Day 02 结构化输出的复现。",
            "传 days=5.5 并把 expect_value 调到 600：算对了但判据不满足，说明 expect 要写「算对没有」。",
        ],
    }


# ============================================================
# 9. 实验 07：生产形态（课程 07）
# ============================================================

_CODE_PRESETS: dict[str, str] = {
    "normal": "print(1234567.89 * 0.0735)",
    "pip_install": "import subprocess\nsubprocess.run(['pip', 'install', 'some-obscure-chart-lib'])",
    "train": "model.fit(X, y)\nprint('done')",
    "random_now": "import random, time\nprint(random.random(), time.time())",
}


def pick_profile(task_type: str, code: str) -> dict[str, Any]:
    """先按代码里的高危模式拒绝，再按任务类型选画像；一刀切会两头错。"""
    if re.search(r"\b(pip|conda|apt)\s+install\b|\bsubprocess\b", code):
        return {**TIMEOUT_PROFILES[3], "name": "reject", "reason": "装包 / 子进程直接拒"}
    if re.search(r"\b(fit|train|GridSearch|torch\.cuda)\b", code):
        return {**TIMEOUT_PROFILES[3], "name": "reject", "reason": "训练 / 长时间任务直接拒"}
    profile = next(p for p in TIMEOUT_PROFILES if p["name"] == task_type)
    return {**profile, "reason": f"按 task_type={task_type} 选档"}


def is_deterministic(code: str) -> bool:
    """判断能否缓存：有 random / now / 网络就不缓存。"""
    return not re.search(r"\brandom\b|\b(now|time\.time|datetime)\b|requests|socket|urllib", code)


def cache_key(code: str, inputs: dict[str, str], profile: dict[str, Any]) -> str:
    """代码 hash + 输入 hash + 画像；确定性任务才能安全复用。"""
    digest = hashlib.sha256()
    digest.update(code.encode("utf-8"))
    for name in sorted(inputs):
        digest.update(name.encode("utf-8"))
        digest.update(hashlib.sha256(inputs[name].encode("utf-8")).digest())
    digest.update(json.dumps(profile, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()[:32]


def production_demo(payload: Day22ProductionRequest) -> dict[str, Any]:
    """一条完整请求路径：画像 → AST → 缓存 → 池 → 执行 → 输出回收 → 审计。"""
    code = _CODE_PRESETS[payload.code_preset]
    profile = pick_profile(payload.task_type, code)
    trace: list[dict[str, Any]] = []
    result: dict[str, Any] = {"ok": False, "stderr": "", "hit_limit": ""}

    trace.append({"gate": "① 画像", "decision": profile["name"], "reason": profile["reason"]})
    if profile["name"] == "reject":
        result["stderr"] = "该操作不被允许（安装依赖/训练/联网）"
        result["hit_limit"] = "policy"
        trace.append({"gate": "短路", "decision": "拒绝", "reason": "不进入 AST 与容器"})
    else:
        ok, deny, review, imports = screen(code)
        trace.append({
            "gate": "② AST", "decision": "放行" if ok else "拦截",
            "reason": "；".join(deny) if deny else (f"审计记录：{review}" if review else "无命中"),
        })
        if not ok:
            result["stderr"] = "代码含被禁止的操作：" + "；".join(deny)
            result["hit_limit"] = "ast"
            trace.append({"gate": "短路", "decision": "拦截", "reason": "AST 命中"})
        else:
            deterministic = is_deterministic(code)
            cache_status = "disabled" if not payload.cache_enabled else ("miss（首次）" if deterministic else "skip（非确定性）")
            trace.append({
                "gate": "缓存", "decision": cache_status,
                "reason": "有 random/now 就别缓存" if not deterministic else "同 hash 第二次请求会命中",
            })
            dirty = payload.code_preset == "random_now"
            trace.append({
                "gate": "③ 池化", "decision": "污染则丢弃" if dirty else "执行后归还",
                "reason": f"池大小 {payload.pool_size}；写过的容器必须重建",
            })
            result = {
                "ok": True, "stdout": "90740.743915\n", "stderr": "", "hit_limit": "",
                "exit_code": 0, "duration_ms": 1182, "truncated": False,
            }
            trace.append({"gate": "④⑤ 输出与回收", "decision": "stdout 8KB 上限", "reason": "头 1/4 + 尾 3/4"})

    profile_for_key = dict(profile)
    profile_for_key.pop("reason", None)
    return {
        "profile": profile,
        "trace": trace,
        "result": result,
        "cache_key": cache_key(code, {"code": code}, profile_for_key),
        "cache_status": next((t["decision"] for t in trace if t["gate"] == "缓存"), "未到达"),
        "pool": {
            "size": payload.pool_size,
            "strategy": "污染容器直接丢弃，健康容器归还队列",
            "wait_ms": [],
        },
        "audit_fields": AUDIT_FIELDS,
        "metrics": METRICS,
        "notes": [
            "ast_deny_rate 是最值钱的指标：突然升高 = 有人在试探，或注入在生效。",
            "只缓存确定性任务；代码里有 random / time.time / 网络就别缓存，除非键里带上这些值。",
            "沙箱管「能不能做」，不管「该不该做」：算出一个有害但正确的结论要靠意图层拦。",
        ],
    }


# ============================================================
# 10. 练习 2/3：压测矩阵与同类错误升级
# ============================================================

_ATTACKS: list[dict[str, str]] = [
    {"attack": "CPU 死循环", "code": "while True: pass", "t": "T4"},
    {"attack": "sleep 100", "code": "import time; time.sleep(100)", "t": "墙钟"},
    {"attack": "[0]*10**9", "code": "x = [0]*10**9", "t": "T5"},
    {"attack": "os.fork 循环", "code": "import os; os.fork()", "t": "T6"},
    {"attack": "读 /etc/passwd", "code": "print(open('/etc/passwd').read())", "t": "T1"},
    {"attack": "socket.connect", "code": "import socket; socket.create_connection(('1.1.1.1', 80))", "t": "T2"},
    {"attack": "写 1GB 文件", "code": "open('/out/big','wb').write(b'x'*10**9)", "t": "T3/T5"},
    {"attack": "递归爆栈", "code": "def f(): return f()\nf()", "t": "崩溃"},
    {"attack": "print 100MB", "code": "print('x'*10**8)", "t": "输出"},
    {"attack": "while True: pass（并发）", "code": "while True: pass", "t": "T4/T6"},
]

# 行 = 攻击，列 = L1/L2；值标明「拦住 / 没拦住 / 杀错进程」
_MATRIX_VERDICTS: list[tuple[str, str, str]] = [
    ("拦住", "拦住", "L1 RLIMIT_CPU 2s；L2 cgroup 同值 quota"),
    ("拦住", "拦住", "L1 wall timeout；L2 wall + 生命周期结束清容器"),
    ("拦住", "拦住", "L1 RLIMIT_DATA；L2 memory.max + swap 同值"),
    ("部分", "拦住", "L1 RLIMIT_NPROC 只管数量，孤儿进程可能残留；L2 pids-limit + PID namespace"),
    ("没拦住", "拦住", "L1 无命名空间可读全盘；L2 只挂载 /in"),
    ("没拦住", "拦住", "L1 软断网可被 _socket 绕过；L2 --network none"),
    ("部分", "拦住", "L1 RLIMIT_FSIZE 限单文件；L2 只读根 + 独立 /out"),
    ("拦住", "拦住", "子进程崩了不带走主进程；L2 同理"),
    ("部分", "拦住", "L1 会截断输出但日志仍写过；L2 同样截断 + 出站审计"),
    ("杀错进程", "拦住", "L1 只 kill(pid) 会留孤儿；L2 容器销毁即全清"),
]

_ESCALATION_RULES: list[dict[str, str]] = [
    {"error": "AttributeError×2", "strategy": "先 print(dir(obj))，再按真实 API 改", "why": "猜方法名只会再错一次"},
    {"error": "timeout×2", "strategy": "强制向量化，拒绝 for / iterrows", "why": "超时是算法选错，不是运气差"},
    {"error": "KeyError×2", "strategy": "先 print(df.columns) 再定列名", "why": "失败前 stdout 是免费上下文"},
]


def pressure_demo(payload: Day22PressureRequest) -> dict[str, Any]:
    """十条攻击 × L1/L2 拦截矩阵；附「同类错误两次换策略」的收益对照。"""
    rows: list[dict[str, Any]] = []
    for index, attack in enumerate(_ATTACKS[: payload.cases]):
        l1, l2, note = _MATRIX_VERDICTS[index]
        rows.append({
            "no": index + 1, "attack": attack["attack"], "code": attack["code"],
            "threat": attack["t"], "l1": l1, "l2": l2, "note": note,
        })
    l1_blocked = sum(r["l1"] == "拦住" for r in rows)
    l2_blocked = sum(r["l2"] == "拦住" for r in rows)
    # 十次合成失败：前两轮同类错误各出现两次，第三轮换策略的概率从 0.3 抬到 0.8
    before = 0.3
    after = 0.8 if payload.escalate else before
    return {
        "matrix": rows,
        "l1_blocked": l1_blocked,
        "l2_blocked": l2_blocked,
        "escalation": {
            "rules": _ESCALATION_RULES,
            "cases": payload.cases,
            "first_pass_before": before,
            "first_pass_after": round(after, 2),
            "enabled": payload.escalate,
            "note": "这是合成样本的示意值；真实提升要用 Day 16 的门禁在真实失败案例上测。",
        },
        "notes": [
            "这张表比任何文档都有说服力：L1 在 T1/T2 上根本没墙，L2 才把边界画出来。",
            "「杀错进程」比「没拦住」更隐蔽：超时后杀掉父进程，孤儿继续跑一夜。",
            "同一错误第二次出现就该换策略；第三次还在原方案上重试，等于把预算烧给运气。",
        ],
    }
