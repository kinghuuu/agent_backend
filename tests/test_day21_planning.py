"""Day 21 单测：规划能力（失败复现 / 计划校验 / L1 / L2 / L3 / 图编排 / 评估门禁）。

运行：``python -m pytest tests/test_day21_planning.py -p no:cacheprovider -q``

四个原则：

1. **不调模型、不写文件**：工具事实来自请求参数，推理步用可复算公式，图用真实
   LangGraph 但节点全是纯函数；整份单测秒级跑完；
2. **反面对照必须有断言**：非法计划必须被编译期拦下、篡改已完成步必须抛异常、
   重复补丁必须熔断、自我表扬必须提前结束；
3. **数学可复算**：``库存 + (交期-节假日)×日产能`` 在所有实验里必须一致，
   断言具体数字而不是「大概对」；
4. **判定要能翻转**：同一份场景改 holidays / expect_mode / approve 必须换结论，
   否则那些规则只是摆设。
"""

import pytest
from pydantic import ValidationError

from app.modules.study.schemas.study import (
    Day21EvalRequest,
    Day21ExecuteRequest,
    Day21FailuresRequest,
    Day21GraphRequest,
    Day21ReflexionRequest,
    Day21ReplanRequest,
    Day21SchemaRequest,
)
from app.modules.study.services import day21_planning as day21


# ---------- 概念表与公共校验器 ----------


def test_mapping_covers_course_sections() -> None:
    """概念表覆盖课程 §1–§14 的要点，且静态行没有空单元格。"""
    data = day21.mapping()
    assert len(data["failure_types"]) == 4
    assert [row["no"] for row in data["failure_types"][:3]] == ["A", "B", "C"]
    assert len(data["react_vs_plan"]) == 8
    assert len(data["plan_fields"]) == 8
    assert len(data["invariants"]) == 3
    assert len(data["plan_forms"]) == 3
    assert [row["action"] for row in data["replan_actions"]] == ["continue", "revise", "finish", "abort"]
    assert len(data["critique_causes"]) == 6
    assert len(data["metrics"]) == 4
    assert len(data["pitfalls"]) == 6
    assert len(data["cheatsheet"]) >= 14
    assert data["versions"]["pydantic"]
    for key in ("plan_fields", "invariants", "plan_forms", "pitfalls", "cheatsheet", "metrics"):
        for row in data[key]:
            assert all(str(value).strip() for value in row.values())


def test_verify_uses_machine_rule_not_self_report() -> None:
    """规则优先：缺字段 / 类型不匹配 / 判据写错都不能算通过。"""
    step = day21.PlanStep(
        id="s1", goal="查可用库存", tool="get_stock", done_criteria="拿到 available",
        expect={"path": "available", "op": ">=", "value": 0},
    )
    assert day21.verify(step, {"available": 120})[0] is True
    assert day21.verify(step, {"stock": 120})[0] is False
    assert day21.verify(step, {"available": "很多"})[0] is False
    assert day21.verify(step, {"error": "工具超时"})[0] is False
    skipped = day21.PlanStep(id="s2", goal="给出结论", done_criteria="给出明确结论", skip_verify=True)
    assert day21.verify(skipped, {})[0] is True


def test_facts_and_feasible_answer_are_recomputable() -> None:
    """默认场景 120+5×80=520 可接；加一天节假日就变成 440 不能接。"""
    base = day21._facts(Day21FailuresRequest())
    assert base["expected_total"] == 520
    assert day21._feasible_answer(base).startswith("可以接")
    holiday = day21._facts(Day21FailuresRequest(holidays=1))
    assert holiday["expected_total"] == 440
    assert day21._feasible_answer(holiday).startswith("不能接")


# ---------- 实验 01：失败复现 ----------


def test_failures_reproduce_three_wrong_answers() -> None:
    """同一问题三种错法，全部 200 + 监控全绿。"""
    result = day21.failures_demo(Day21FailuresRequest())
    assert [case["symptom"] for case in result["cases"]] == ["premature", "monolithic", "no_backtrack"]
    assert result["summary"]["wrong_count"] == 3
    assert result["summary"]["all_green"] is True
    assert result["correct_answer"].startswith("可以接")
    for case in result["cases"]:
        assert case["verdict"] == "错误"
        assert case["monitor"]["http_status"] == 200
        assert case["monitor"]["slo_alert"] is False
        assert case["missing_facts"]


def test_failures_follow_facts_when_scenario_changes() -> None:
    """改节假日后正确答案翻转，说明剧本不是写死的字符串。"""
    result = day21.failures_demo(Day21FailuresRequest(holidays=1))
    assert result["correct_answer"].startswith("不能接")


def test_failures_can_select_one_symptom() -> None:
    """按症状筛选时只跑一条轨迹，便于课堂逐个讲解。"""
    result = day21.failures_demo(Day21FailuresRequest(symptom="no_backtrack"))
    assert len(result["cases"]) == 1
    assert result["cases"][0]["symptom"] == "no_backtrack"


# ---------- 实验 02：计划结构与不变量 ----------


@pytest.mark.parametrize(
    "case",
    ["cycle", "missing_expect", "unknown_dep", "duplicate_id", "self_dep", "too_long"],
)
def test_invalid_plans_are_rejected_before_execution(case: str) -> None:
    """六类非法计划必须在编译期被拒，不能等到跑起来才炸。"""
    result = day21.schema_demo(Day21SchemaRequest(case=case))
    assert result["accepted"] is False
    assert result["errors"]
    assert any(not check["ok"] for check in result["checks"])
    assert result["topo_levels"] == []


def test_valid_plan_is_leveled_and_ready() -> None:
    """合法计划的 s1/s2 同层可并行，s3 等两者完成。"""
    result = day21.schema_demo(Day21SchemaRequest(case="valid"))
    assert result["accepted"] is True
    assert result["topo_levels"] == [["s1", "s2"], ["s3"]]
    assert result["ready_steps"] == ["s1", "s2"]
    assert all(check["ok"] for check in result["checks"])


def test_plan_model_enforces_invariants_directly() -> None:
    """除了 demo 包装，Plan 模型本身也必须拒绝环与缺判据。"""
    with pytest.raises(ValidationError, match="依赖成环"):
        day21.Plan(**day21._schema_raw("cycle"))
    with pytest.raises(ValidationError, match="会假完成"):
        day21.Plan(**day21._schema_raw("missing_expect"))


# ---------- 实验 03：L1 计划先行 ----------


def test_l1_good_plan_fixes_premature_and_monolithic() -> None:
    """计划清单治漏探查，工具事实治硬扛；并行层先跑 s1/s2。"""
    result = day21.execute_demo(Day21ExecuteRequest())
    assert result["failed_at"] is None
    assert result["answer"].startswith("可以接")
    assert result["levels"] == [["s1", "s2"], ["s3"]]
    assert [step["sid"] for step in result["trace"]] == ["s1", "s2", "s3"]
    assert all(step["passed"] for step in result["trace"])
    assert result["trace"][2]["context"] == {"s1": {"available": 120}, "s2": {"per_day": 80}}
    assert "C" in "".join(result["unfixed_symptoms"])


def test_l1_cannot_backtrack_from_external_only_plan() -> None:
    """只查补货周期的计划在 s2 校验失败后停住 —— L1 的天花板。"""
    result = day21.execute_demo(Day21ExecuteRequest(plan_kind="external_only"))
    assert result["failed_at"] == "s2"
    assert result["answer"] is None
    assert [step["sid"] for step in result["trace"]] == ["s1", "s2"]
    assert result["trace"][-1]["passed"] is False


def test_business_expect_kills_a_correct_answer() -> None:
    """把验收判据写成 >=500，会把手算出 440 的正确答案判成失败。"""
    result = day21.execute_demo(Day21ExecuteRequest(plan_kind="business_expect", holidays=1))
    assert result["failed_at"] == "s3"
    assert result["trace"][-1]["output"] == {"total": 440}
    assert "业务结论" in "".join(result["unfixed_symptoms"])


# ---------- 实验 04：L2 重规划 ----------


def test_replan_revise_adds_self_production_branch() -> None:
    """外采失败后补查产能、改结论步，失败步仍留在账本里。"""
    result = day21.replan_demo(Day21ReplanRequest())
    assert result["guard"].startswith("revise")
    assert result["answer"].startswith("可以接")
    assert [p["version"] for p in result["plans"]] == [1, 2]
    assert [p["op"] for p in result["patches"]] == ["add", "modify"]
    failed = [s for s in result["plans"][-1]["steps"] if s["id"] == "s2"]
    assert failed[0]["status"] == "failed"
    assert [s["id"] for s in result["plans"][-1]["steps"]] == ["s1", "s2", "s3", "s4"]


def test_replan_other_actions_are_mutually_exclusive() -> None:
    """finish / abort 是两条独立出口，不该继续改计划。"""
    finish = day21.replan_demo(Day21ReplanRequest(action="finish"))
    abort = day21.replan_demo(Day21ReplanRequest(action="abort"))
    assert finish["guard"].startswith("finish")
    assert finish["answer"].startswith("不能接")
    assert abort["guard"].startswith("abort")
    assert abort["answer"].startswith("无法完成")
    assert len(finish["plans"]) == len(abort["plans"]) == 1


def test_repeat_patch_is_cut_off() -> None:
    """同一补丁第二次出现就熔断，不能让 replanner 无限绕圈。"""
    result = day21.replan_demo(Day21ReplanRequest(action="repeat"))
    assert "死循环" in result["guard"]
    assert result["answer"] is None


def test_tampering_done_step_is_rejected() -> None:
    """已完成步骤冻结：放宽 s1 判据的尝试必须被账本不变量拒绝。"""
    result = day21.replan_demo(Day21ReplanRequest(tamper_done=True))
    assert result["integrity_check"]["attempted"] is True
    assert result["integrity_check"]["rejected"] is True
    assert "历史不可改写" in result["integrity_check"]["reason"]
    assert result["answer"] is None


def test_apply_patch_rejects_modifying_failed_step_too() -> None:
    """失败步的结论也是事实：s2 failed 同样不能改。"""
    plan = day21.Plan(**day21._plan_raw("external_only", day21._facts(Day21FailuresRequest())))
    plan.steps[1].status = "failed"
    patch = day21.StepPatch(
        op="modify", step_id="s2", reason="想把失败步改掉",
        step=day21.PlanStep(
            id="s2", goal="查补货周期", tool="get_lead_time", done_criteria="拿到 days",
            expect={"path": "days", "op": "<=", "value": 99},
        ),
    )
    with pytest.raises(day21.PlanIntegrityError, match="历史不可改写"):
        day21.apply_patch(plan, [patch])


def test_replan_skips_when_no_deviation() -> None:
    """补货周期满足交期时没必要重规划：没有信息量就不花钱。"""
    result = day21.replan_demo(Day21ReplanRequest(lead_time=3))
    assert "没有必要为没发生的偏离" in result["guard"]
    assert result["answer"] is not None
    assert len(result["plans"]) == 1


# ---------- 实验 05：L3 反思 ----------


def test_reflexion_repairs_ignored_holiday_constraint() -> None:
    """第一次满产算错 → 归因 ignored_constraint → 第二次按有效天数算对。"""
    result = day21.reflexion_demo(Day21ReflexionRequest(holidays=1))
    assert result["attempts"][0]["output"] == {"total": 520}
    assert result["attempts"][1]["output"] == {"total": 440}
    assert result["critiques"][0]["root_cause"] == "ignored_constraint"
    assert "4" in result["critiques"][0]["fix_instruction"]
    assert result["verdict"].startswith("通过")
    assert result["answer"].startswith("不能接")


def test_outcome_expect_creates_false_finish() -> None:
    """验收判据写成 >=500 时，第一次满产算错 520 反而「通过」= 假完成。"""
    result = day21.reflexion_demo(Day21ReflexionRequest(holidays=1, expect_mode="outcome"))
    assert result["verdict"].startswith("假完成")
    assert result["attempts"][0]["output"] == {"total": 520}
    assert result["attempts"][0]["passed"] is True
    assert result["expect"]["op"] == ">="
    assert result["answer"].startswith("不能接")


def test_outcome_expect_can_also_kill_a_correct_answer() -> None:
    """换一组事实让两次尝试都不达标，就变成另一种失败：正确答案被误杀。"""
    result = day21.reflexion_demo(Day21ReflexionRequest(
        stock=0, capacity=80, days=5, holidays=1, order_qty=500,
        expect_mode="outcome",
    ))
    assert result["verdict"].startswith("假失败")
    assert result["answer"].startswith("不能接")


def test_self_praise_critique_ends_early() -> None:
    """root_cause=none 的批评让重试提前结束，且不会写出教训。"""
    result = day21.reflexion_demo(Day21ReflexionRequest(holidays=1, critique_style="self_praise"))
    assert len(result["attempts"]) == 1
    assert result["critiques"][0]["root_cause"] == "none"
    assert result["lesson"] == ""
    assert "自我表扬" in result["verdict"]


def test_lesson_escalates_to_hard_constraint() -> None:
    """同类错误两次升级成硬约束；清单去重保序。"""
    memory = {"ignored_constraint": ["第一次：扣掉节假日"]}
    assert day21.build_lesson(memory, "ignored_constraint").startswith("【历史教训】")
    memory["ignored_constraint"].append("第二次：按有效天数算")
    lesson = day21.build_lesson(memory, "ignored_constraint")
    assert lesson.startswith("【必须遵守的硬约束】")
    assert day21.build_lesson(memory, "bad_calc") == ""

    result = day21.reflexion_demo(Day21ReflexionRequest(holidays=1, prior_same_errors=2))
    assert "【必须遵守的硬约束】" in result["attempts"][0]["lesson_injected"]
    assert result["attempts"][0]["lesson_injected"].count("历史教训") == 2


def test_vague_fix_instruction_is_rejected() -> None:
    """「要更仔细」这类空话必须在 Critique 构造时就被拒（none 不在此列）。"""
    with pytest.raises(ValidationError, match="必须指向具体对象"):
        day21.Critique(
            what_went_wrong="这次计算没有扣除节假日导致总量偏高",
            root_cause="ignored_constraint",
            fix_instruction="以后要更加仔细地检查约束",
            confidence=0.9,
        )
    okay = day21.Critique(
        what_went_wrong="这次计算没有扣除节假日导致总量偏高",
        root_cause="none", fix_instruction="无需修改", confidence=0.1,
    )
    assert okay.root_cause == "none"


# ---------- 实验 06：图编排 ----------


def test_graph_normal_path_finishes_without_interrupt() -> None:
    """一路顺风时不该付中断审批的钱。"""
    result = day21.graph_demo(Day21GraphRequest(scenario="normal"))
    assert result["interrupt_payload"] is None
    assert result["approved"] is None
    assert result["answer"].startswith("可以接")
    assert result["audit"][-1] == "finish"


def test_graph_patch_path_waits_for_human_approval() -> None:
    """偏离后暂停在计划补丁上；批准才升 v2 并算出正确答案。"""
    result = day21.graph_demo(Day21GraphRequest(scenario="patch", approve=True))
    assert result["interrupt_payload"]["type"] == "plan_patch"
    assert len(result["interrupt_payload"]["patches"]) == 2
    assert result["approved"] is True
    assert result["plan"]["version"] == 2
    assert result["answer"].startswith("可以接")
    assert any("replan v2" in line for line in result["audit"])


def test_graph_rejected_patch_goes_abort() -> None:
    """拒绝计划变更就停在这里，不能偷偷继续执行。"""
    result = day21.graph_demo(Day21GraphRequest(scenario="patch", approve=False))
    assert result["approved"] is False
    assert result["answer"] == "计划变更被拒绝"
    assert result["plan"]["version"] == 1
    assert any("rejected" in line for line in result["audit"])


def test_graph_reflect_runs_before_replan() -> None:
    """reflexion_left>0 时先反思重试，仍失败才走重规划。"""
    result = day21.graph_demo(Day21GraphRequest(reflexion_left=1))
    assert any("reflect s2" in line for line in result["audit"])
    assert any("replan v2" in line for line in result["audit"])
    assert result["answer"].startswith("可以接")


# ---------- 实验 07：评估门禁 ----------


def test_eval_compares_four_schemes_on_same_dataset() -> None:
    """同一份 25 条数据：A→B 跃升最大，C 首次通过率继续涨。"""
    result = day21.eval_demo(Day21EvalRequest())
    assert result["dataset_size"] == 25
    assert [row["scheme"] for row in result["summary"]] == ["react", "l1", "l2", "l3"]
    rates = [row["correct_rate"] for row in result["summary"]]
    assert rates[0] < rates[1] < rates[2] == rates[3]
    first_pass = [row["first_pass_rate"] for row in result["summary"]]
    assert first_pass[1] < first_pass[2] < first_pass[3]
    assert result["summary"][1]["plan_validity"] < result["summary"][2]["plan_validity"]


def test_eval_gate_can_flip_with_threshold() -> None:
    """门禁阈值改严后结论必须翻转，否则门禁只是摆设。"""
    relaxed = day21.eval_demo(Day21EvalRequest(scheme="l1"))
    strict = day21.eval_demo(Day21EvalRequest(
        scheme="l1", gate_first_pass=0.99, gate_validity=0.99
    ))
    assert relaxed["gate"]["passed"] is False  # B 档首次通过率 18/25 = 72%，默认 0.75 门禁不过
    assert "first_pass_rate" in relaxed["gate"]["failed"]
    assert strict["gate"]["passed"] is False
    passing = day21.eval_demo(Day21EvalRequest(scheme="l2"))
    assert passing["gate"]["passed"] is True


def test_eval_one_separates_valid_correct_and_false_finish() -> None:
    """假完成 = validator 判过但人工判错，不能只看正确率。"""
    row = day21.eval_one("l1", 20)
    assert row["valid"] is True
    assert row["correct"] is False
    assert row["false_finish"] is True
    row = day21.eval_one("l2", 9)
    assert row["correct"] is True
    assert row["false_finish"] is False
    summary = day21.summarize([{"correct": True, "valid": True, "steps": 3.2, "replans": 0, "false_finish": False}], 3.2)
    assert summary["step_efficiency"] == 1.0
    assert summary["first_pass_rate"] == 1.0
