"""Day 22 单测：代码执行沙箱（分工 / AST / L1 / 网络文件 / L2 / 代码循环 / 生产 / 压测）。

运行：``python -m pytest tests/test_day22_sandbox.py -p no:cacheprovider -q``

四个原则：

1. **默认不碰系统**：除显式 ``mode=live`` 的两条安全探针外，全部是回放/纯逻辑，
   不联网、不挂宿主目录、不起 docker、不 fork bomb；
2. **课程修正必须有断言**：直接 `__import__` 报 NameError、逃逸链能拿回 builtins、
   AST 的 hex 样本要拼调用名才真绕过，三条都断言实际行为；
3. **「挡不住」要可数**：手滑写文件放行、绕过漏 2 条、L1 网络/文件墙破 3 条、L1 只拦 4/10；
4. **live 只跑安全代码**：正常计算与 sleep 超时，整数结果可复算，超时后不留孤儿。
"""

import pytest
from pydantic import ValidationError

from app.modules.study.schemas.study import (
    Day22AstRequest,
    Day22BridgeRequest,
    Day22ContainerRequest,
    Day22LoopRequest,
    Day22NetfsRequest,
    Day22PressureRequest,
    Day22ProductionRequest,
    Day22WhyRequest,
    Day22L1Request,
)
from app.modules.study.services import day22_sandbox as d22


# ---------- 概念表与课程修正 ----------


def test_mapping_covers_all_concepts() -> None:
    """概念表覆盖 §1–§15 的全部表格，静态行没有空单元格。"""
    data = d22.mapping()
    assert len(data["why_calc"]) == 4
    assert len(data["division"]) == 4
    assert len(data["threats"]) == 7
    assert [row["no"] for row in data["threats"]] == [f"T{i}" for i in range(1, 8)]
    assert [row["level"] for row in data["levels"]] == ["L0", "L1", "L2", "L3", "L4"]
    assert len(data["gates"]) == 5
    assert len(data["container_args"]) == 10
    assert len(data["pitfalls"]) == 6
    assert len(data["cheatsheet"]) >= 20
    assert "exit_code" in data["audit_fields"] and "ast_denied" in data["audit_fields"]
    assert [p["name"] for p in data["timeout_profiles"]] == ["quick_calc", "data_analysis", "plot", "reject"]
    assert data["environment"]["platform"]
    for key in ("threats", "levels", "gates", "container_args", "pitfalls", "cheatsheet", "limits", "metrics"):
        for row in data[key]:
            assert all(str(value).strip() for value in row.values())


def test_why_demo_records_real_cpython_behavior() -> None:
    """心算错 0.5 元；直接 __import__ 报 NameError；逃逸链确实能拿回 builtins。"""
    result = d22.why_demo(Day22WhyRequest())
    assert result["exact_value"] == 90740.74
    assert result["llm_wrong"] is True
    assert "90,740.24" in result["llm_answer"]
    direct, chained = result["exec_cases"]
    assert "NameError" in direct["actual"] and not direct["escaped"]
    assert chained["escaped"] is True
    assert len(result["corrections"]) == 2
    assert "实际执行：逃逸链成立" in result["notes"][2]


# ---------- 实验 02：AST 筛查 ----------


def test_ast_screen_blocks_denies_but_not_semantics() -> None:
    """模块黑名单一拦一个准；to_csv 覆盖源文件这种语义危害完全看不见。"""
    ok, deny, review, imports = d22.screen("import os\nprint(os.environ['KEY'])")
    # os.environ 是普通属性访问，AST 只看得到 import os；print 进审计记录
    assert ok is False and len(deny) == 1 and "L1: import os" in deny
    assert any("print()" in item for item in review)
    assert imports == ["os"]
    ok, _, _, _ = d22.screen("import pandas as pd\ndf.to_csv('/data/orders.csv')")
    assert ok is True
    ok, deny, _, _ = d22.screen("print(().__class__.__base__.__subclasses__())")
    assert ok is False and any("__subclasses__" in item for item in deny)


@pytest.mark.parametrize("case_key", ["import_os", "env", "dunder"])
def test_ast_demo_denies_obvious_samples(case_key: str) -> None:
    """明显手滑样本全部拦下。"""
    result = d22.ast_demo(Day22AstRequest(case=case_key))
    assert len(result["cases"]) == 1
    assert result["cases"][0]["allowed"] is False
    assert result["cases"][0]["deny"]


def test_ast_demo_exposes_its_own_blind_spots() -> None:
    """第一个关键结论：手滑写文件没拦住；第二个：四种绕过漏了两条。"""
    result = d22.ast_demo(Day22AstRequest())
    write = next(c for c in result["cases"] if c["key"] == "write")
    assert write["allowed"] is True
    assert result["write_file_blocked"] is False
    assert result["bypass_leaked"] == 2
    leaked = [c["name"] for c in result["cases"] if c["group"] == "bypass" and c["allowed"]]
    assert any("字符串拼接" in name for name in leaked)
    assert any("hex" in name for name in leaked)
    assert "课程把 hex 那条写成" in " ".join(result["notes"])


# ---------- 实验 03：L1 子进程 ----------


def test_l1_simulated_matrix_maps_exit_codes() -> None:
    """退出码映射成 hit_limit：CPU / 超时 / 内存 / 读环境各有结论。"""
    result = d22.l1_demo(Day22L1Request())
    cases = {c["name"]: c for c in result["cases"]}
    assert cases["正常计算"]["stdout"].strip() == "90740.743915"
    assert cases["CPU 死循环"]["hit_limit"] == "cpu"
    assert cases["sleep 骗时间"]["hit_limit"] == "timeout"
    assert cases["内存炸弹"]["hit_limit"] == "mem"
    assert cases["读环境变量（L1 挡不住）"]["ok"] is True
    assert "sk-demo-canary" in cases["读环境变量（L1 挡不住）"]["stdout"]
    assert result["limits"]["stdin"] == "DEVNULL"


def test_l1_rejects_custom_code_at_ast_gate() -> None:
    """自定义代码先过 AST；import os 不进子进程。"""
    result = d22.l1_demo(Day22L1Request(custom_code="import os\nprint(os.getcwd())", mode="live"))
    assert result["cases"][0]["hit_limit"] == "ast"
    assert result["cases"][0]["ok"] is False
    assert "import os" in result["cases"][0]["stderr"]


def test_l1_live_runs_safe_code_and_kills_timeout() -> None:
    """真起子进程：正常计算可复算，sleep 由 wall timeout 杀掉，fork bomb 永不真跑。"""
    normal = d22.l1_demo(Day22L1Request(case="normal", mode="live", wall_timeout=3))
    row = normal["cases"][0]
    assert row["ok"] is True
    assert row["exit_code"] == 0
    assert float(row["stdout"].strip()) == pytest.approx(1234567.89 * 0.0735, abs=1e-6)

    sleeping = d22.l1_demo(Day22L1Request(case="sleep", mode="live", wall_timeout=1))
    assert sleeping["cases"][0]["hit_limit"] == "timeout"
    assert sleeping["cases"][0]["timed_out"] is True

    fork = d22.l1_demo(Day22L1Request(case="fork", mode="live"))
    assert fork["cases"][0]["skipped"]
    assert fork["cases"][0]["ok"] is True  # 只回放，不真 fork


def test_truncate_keeps_tail_traceback() -> None:
    """头 1/4 + 尾 3/4：尾部 traceback 必须保留，且标注被截断的字数。"""
    text = "HEAD" + "x" * 100 + "TAIL-ERROR"
    cut, flag = d22.truncate(text, 40)
    assert flag is True
    assert "TAIL-ERROR" in cut
    assert "截断" in cut


# ---------- 实验 04：网络与文件系统 ----------


def test_netfs_shows_broken_walls() -> None:
    """四条探针三条墙不存在/被绕过；只有老实走 socket 模块的被软断网拦下。"""
    result = d22.netfs_demo(Day22NetfsRequest())
    assert result["broken"] == 3
    by_name = {p["name"]: p for p in result["probes"]}
    assert by_name["软断网（走 socket 模块）"]["ok"] is False
    assert by_name["软断网的绕过（_socket）"]["ok"] is True
    assert by_name["读宿主机文件"]["ok"] is True
    assert by_name["写宿主机文件"]["ok"] is True
    assert "builtins.__import__" in result["guard_code"]


def test_netfs_live_socket_guard_is_real() -> None:
    """live 探针：在子进程里 patch 后 import socket 真的报 Blocked。"""
    result = d22.netfs_demo(Day22NetfsRequest(probe="socket", mode="live"))
    assert result["probes"][0]["mode"] == "live"
    assert result["probes"][0]["ok"] is False
    assert "禁止导入网络模块" in result["probes"][0]["last_line"]


# ---------- 实验 05：容器 ----------


def test_container_command_has_every_hardening_flag() -> None:
    """九个安全参数逐条生成；memory 与 swap 必须同值。"""
    result = d22.container_demo(Day22ContainerRequest(mem_mb=256, cpus=0.25, pids_limit=32))
    cmd = result["command"]
    for flag in ("--network none", "--read-only", "--tmpfs", "--cap-drop=ALL", "no-new-privileges", "65534:65534", "-v $WORK:/in:ro", "-v $OUT:/out:rw"):
        assert flag in cmd
    assert cmd.count("256m") == 2
    assert "--cpus 0.25" in cmd and "--pids-limit 32" in cmd
    assert [row["arg"] for row in result["args"]][0] == "--network none"
    assert result["live_result"] is None


def test_container_mount_boundary_is_exposure_boundary() -> None:
    """勾上 mount_env 才会多一条 .env 挂载；探针明确标出这是主动挂进去的。"""
    plain = d22.container_demo(Day22ContainerRequest())
    with_env = d22.container_demo(Day22ContainerRequest(mount_env=True))
    assert ".env" not in plain["command"]
    assert "$ENV_FILE:/in/.env:ro" in with_env["command"]
    assert "主动挂进去" in with_env["probes"][0]["last"]


# ---------- 实验 06：代码循环 ----------


def test_loop_pandas_append_rewrite() -> None:
    """pandas 2.0 删了 append：第一轮 AttributeError，第二轮 concat 通过。"""
    result = d22.loop_demo(Day22LoopRequest())
    assert result["tries"] == 2
    assert result["attempts"][0]["digest"]["type"] == "AttributeError"
    assert result["attempts"][0]["digest"]["line"] == 4
    assert "df.append" in result["attempts"][0]["digest"]["code_line"]
    assert result["answer"] == "{'北京': 2580, '上海': 1980, '广州': 860}"
    assert "失败前的输出" in result["feedback"][0]
    assert "['order_id', 'amount', 'city']" in result["feedback"][0]


def test_loop_key_error_feedback_includes_columns() -> None:
    """失败前 stdout 的列名让模型一次改对，不需要猜。"""
    result = d22.loop_demo(Day22LoopRequest(scenario="key_error"))
    assert result["attempts"][0]["digest"]["type"] == "KeyError"
    assert "['order_id', 'amount', 'city']" in result["feedback"][0]
    assert result["answer"] == "{'北京': 2580}"


def test_loop_timeout_needs_strategy_not_retry() -> None:
    """超时给「换算法」提示：开 escalate 后策略提示进 feedback 并换向量化代码。"""
    result = d22.loop_demo(Day22LoopRequest(scenario="timeout", escalate=True))
    assert result["attempts"][0]["hit_limit"] == "timeout"
    assert "向量化" in result["strategy_switches"][0]
    assert "iterrows" in result["attempts"][1]["code"]  # 第二次仍在原方案上跑
    assert "iterrows" not in result["attempts"][2]["code"]
    assert result["tries"] == 3 and result["answer"] == "16695.0"


def test_loop_same_error_escalation_changes_outcome() -> None:
    """练习 3：同一错误重复两次，第三次不换策略必然失败，换策略才成功。"""
    stuck = d22.loop_demo(Day22LoopRequest(scenario="same_error", escalate=False))
    improved = d22.loop_demo(Day22LoopRequest(scenario="same_error", escalate=True))
    assert stuck["answer"] is None and stuck["tries"] == 3
    assert improved["answer"] and improved["tries"] == 3
    assert any("dir(obj)" in s for s in improved["strategy_switches"])
    assert "print(dir(df))" in improved["attempts"][2]["code"]


def test_extract_code_handles_fence_and_plain_text() -> None:
    """模型可能包 ```python 围栏，也可能直接吐代码。"""
    assert d22.extract_code("```python\nprint(1)\n```") == "print(1)"
    assert d22.extract_code("print(2)") == "print(2)"


# ---------- 练习 1：接回 Day 21 ----------


def test_bridge_parses_stdout_before_verify() -> None:
    """stdout 是字符串：必须先 float，再交给 Day 21 的机器判据。"""
    result = d22.bridge_demo(Day22BridgeRequest())
    assert result["parsed_value"] == 520.0
    assert result["raw_type"] == "str"
    assert result["parsed_type"] == "float"
    assert result["passed"] is True
    assert "total = available + days * per_day" in result["code"]


def test_bridge_float_days_and_expect_flip_verdict() -> None:
    """5.5 天 → 560；把判据抬到 600 就失败，说明 expect 判「算对没有」之外的东西会误杀。"""
    result = d22.bridge_demo(Day22BridgeRequest(days=5.5, expect_value=600))
    assert result["parsed_value"] == 560.0
    assert result["passed"] is False
    assert ">= 600" in result["reason"]


def test_bridge_live_executor_matches_simulation() -> None:
    """mode=live 用真子进程执行三行代码，结果与回放一致。"""
    result = d22.bridge_demo(Day22BridgeRequest(mode="live"))
    assert result["stdout"].strip() == "520.0"
    assert result["parsed_value"] == 520.0
    assert result["passed"] is True


# ---------- 实验 07：生产形态 ----------


@pytest.mark.parametrize("preset", ["pip_install", "train"])
def test_production_rejects_install_and_training(preset: str) -> None:
    """装包与训练在画像阶段就被拒，不进 AST 与容器。"""
    result = d22.production_demo(Day22ProductionRequest(code_preset=preset))
    assert result["profile"]["name"] == "reject"
    assert result["result"]["hit_limit"] == "policy"
    assert any(step["gate"] == "短路" for step in result["trace"])


def test_production_skips_cache_for_nondeterministic_code() -> None:
    """有 random / time.time 就不缓存；池化对写过的容器直接丢弃。"""
    result = d22.production_demo(Day22ProductionRequest(code_preset="random_now"))
    assert result["cache_status"] == "skip（非确定性）"
    assert any("污染则丢弃" in step["decision"] for step in result["trace"])
    assert len(result["cache_key"]) == 32


def test_cache_key_changes_with_code_inputs_and_profile() -> None:
    """代码 / 输入 / 画像任一变化，键都要变；同输入两次必须一致。"""
    profile = {"wall": 3, "mem_mb": 256}
    a = d22.cache_key("print(1)", {"x": "1"}, profile)
    assert a == d22.cache_key("print(1)", {"x": "1"}, profile)
    assert a != d22.cache_key("print(2)", {"x": "1"}, profile)
    assert a != d22.cache_key("print(1)", {"x": "2"}, profile)
    assert a != d22.cache_key("print(1)", {"x": "1"}, {**profile, "wall": 5})


def test_pick_profile_uses_code_pattern_first() -> None:
    """代码里出现 subprocess/fit 直接拒，不能被 task_type=quick_calc 掩盖。"""
    assert d22.pick_profile("quick_calc", "import subprocess")["name"] == "reject"
    assert d22.pick_profile("data_analysis", "model.fit(X, y)")["name"] == "reject"
    assert d22.pick_profile("quick_calc", "print(1+1)")["name"] == "quick_calc"


# ---------- 练习 2：压测矩阵 ----------


def test_pressure_matrix_counts_l1_gaps() -> None:
    """十条攻击 L1 只全拦 4 条，L2 十条全拦；两条明确标「没拦住」。"""
    result = d22.pressure_demo(Day22PressureRequest())
    assert len(result["matrix"]) == 10
    assert result["l1_blocked"] == 4
    assert result["l2_blocked"] == 10
    assert sum(r["l1"] == "没拦住" for r in result["matrix"]) == 2
    assert any(r["l1"] == "杀错进程" for r in result["matrix"])
    assert all(r["l2"] == "拦住" for r in result["matrix"])


def test_pressure_escalation_improves_first_pass() -> None:
    """练习 3：开升级后首次通过率从 0.3 抬到 0.8，三条规则覆盖三类高频错误。"""
    off = d22.pressure_demo(Day22PressureRequest(escalate=False))["escalation"]
    on = d22.pressure_demo(Day22PressureRequest(escalate=True))["escalation"]
    assert off["first_pass_before"] == off["first_pass_after"] == 0.3
    assert on["first_pass_after"] == 0.8
    assert {rule["error"] for rule in on["rules"]} == {"AttributeError×2", "timeout×2", "KeyError×2"}
    assert "合成样本" in on["note"]


# ---------- Schema 边界 ----------


def test_request_schemas_reject_out_of_range_values() -> None:
    """超时/内存/样本数都有边界，避免前端误传把机器拖死。"""
    with pytest.raises(ValidationError):
        Day22L1Request(wall_timeout=0)
    with pytest.raises(ValidationError):
        Day22L1Request(mem_mb=99999)
    with pytest.raises(ValidationError):
        Day22PressureRequest(cases=100)
    with pytest.raises(ValidationError):
        Day22BridgeRequest(days=-1)
