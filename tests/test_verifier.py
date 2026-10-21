"""交付验证器单元测试（不依赖 LLM / Jupyter kernel）。"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.verifier import (
    build_submission_self_check,
    build_submission_self_check_code,
    format_pre_submit_feedback,
    format_skill_check_feedback,
    format_verification_feedback,
    make_prediction_contract,
    run_skill_checks,
    verify_artifacts,
)


def _write_predictions(path: Path, n: int = 5) -> None:
    pd.DataFrame({
        "id": list(range(n)),
        "Churn": [0.1, 0.8, 0.3, 0.9, 0.2][:n],
    }).to_csv(path, index=False)


def _labels(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame({
        "id": list(range(n)),
        "Churn": [0, 1, 0, 1, 0][:n],
    })


def _state(final_report: str = "完成") -> dict:
    return {"final_report": final_report, "code_history": []}


def test_contract_passes_when_artifacts_are_valid(tmp_path):
    pred = tmp_path / "pred.csv"
    report = tmp_path / "report.md"
    labels = tmp_path / "labels.csv"
    _write_predictions(pred)
    _labels().to_csv(labels, index=False)
    report.write_text("分析报告" * 100, encoding="utf-8")

    contract = make_prediction_contract(
        tmp_path,
        pred,
        report,
        target_column="Churn",
        expected_rows=5,
        labels_path=labels,
        metric="roc_auc",
        min_metric=0.5,
    )
    result = verify_artifacts(contract, _state())

    assert result["passed"], result
    assert result["metrics"]["value"] == pytest.approx(1.0)
    assert all(check["passed"] for check in result["checks"])


def test_missing_prediction_file_fails(tmp_path):
    contract = make_prediction_contract(
        tmp_path,
        tmp_path / "missing.csv",
        target_column="Churn",
    )
    result = verify_artifacts(contract, _state())

    assert not result["passed"]
    assert any(
        check["name"].startswith("file_exists") and not check["passed"]
        for check in result["checks"]
    )


def test_wrong_columns_fail(tmp_path):
    pred = tmp_path / "pred.csv"
    pd.DataFrame({"id": [1, 2], "wrong": [0.1, 0.2]}).to_csv(pred, index=False)
    contract = make_prediction_contract(
        tmp_path,
        pred,
        target_column="Churn",
    )
    result = verify_artifacts(contract, _state())

    assert not result["passed"]
    assert any("缺少列" in check["detail"] for check in result["checks"])


def test_row_count_mismatch_fails(tmp_path):
    pred = tmp_path / "pred.csv"
    _write_predictions(pred, n=3)
    contract = make_prediction_contract(
        tmp_path,
        pred,
        target_column="Churn",
        expected_rows=5,
    )
    result = verify_artifacts(contract, _state())

    assert not result["passed"]
    assert any(
        check["name"].startswith("row_count") and not check["passed"]
        for check in result["checks"]
    )


def test_probability_range_fails(tmp_path):
    pred = tmp_path / "pred.csv"
    pd.DataFrame({"id": [1, 2], "Churn": [1.5, 0.2]}).to_csv(pred, index=False)
    contract = make_prediction_contract(
        tmp_path,
        pred,
        target_column="Churn",
        probability=True,
    )
    result = verify_artifacts(contract, _state())

    assert not result["passed"]
    assert any(
        check["name"].startswith("probability_range") and not check["passed"]
        for check in result["checks"]
    )


def test_group_validation_is_checked(tmp_path):
    pred = tmp_path / "pred.csv"
    _write_predictions(pred)
    contract = make_prediction_contract(
        tmp_path,
        pred,
        target_column="Churn",
        group_column="prompt_id",
    )

    without_group = verify_artifacts(contract, _state())
    assert not without_group["passed"]
    assert any(
        check["name"] == "group_validation" and not check["passed"]
        for check in without_group["checks"]
    )

    with_group = verify_artifacts(
        {**contract},
        {
            "final_report": "完成",
            "code_history": ["from sklearn.model_selection import StratifiedGroupKFold\n"],
        },
    )
    assert with_group["passed"], with_group


def test_feedback_lists_only_failures(tmp_path):
    contract = make_prediction_contract(
        tmp_path,
        tmp_path / "missing.csv",
        target_column="Churn",
    )
    result = verify_artifacts(contract, _state())
    feedback = format_verification_feedback(result)

    assert "[验证失败]" in feedback
    assert "file_exists" in feedback
    assert "必须写文件" in feedback


def test_feedback_includes_executable_self_check(tmp_path):
    pred = tmp_path / "pred.csv"
    report = tmp_path / "report.md"
    contract = make_prediction_contract(
        tmp_path,
        pred,
        report,
        target_column="Churn",
        expected_rows=5,
        probability=True,
    )
    result = verify_artifacts(contract, _state())
    feedback = format_verification_feedback(result, contract)

    assert "SELF-CHECK PASSED" in feedback
    assert "expected_rows = 5" in feedback
    assert "between(0, 1)" in feedback
    assert "len(report_text.strip())" in feedback


def test_self_check_omits_row_assert_when_not_specified(tmp_path):
    contract = make_prediction_contract(
        tmp_path,
        tmp_path / "pred.csv",
        target_column="Churn",
    )
    snippet = build_submission_self_check(contract)
    assert "expected_rows" in snippet
    assert "行数错误" not in snippet


def test_raw_self_check_code_has_no_markdown_fence(tmp_path):
    contract = make_prediction_contract(
        tmp_path,
        tmp_path / "pred.csv",
        target_column="Churn",
        expected_rows=5,
    )
    raw = build_submission_self_check_code(contract)
    fenced = build_submission_self_check(contract)

    assert not raw.startswith("```")
    assert "SELF-CHECK PASSED" in raw
    assert "expected_rows = 5" in raw
    assert fenced.startswith("```python")
    assert raw in fenced


def test_pre_submit_feedback_includes_error_and_check(tmp_path):
    contract = make_prediction_contract(
        tmp_path,
        tmp_path / "pred.csv",
        target_column="Churn",
        expected_rows=5,
    )

    class Result:
        error = "AssertionError: 预测文件不存在"
        text = ""

    feedback = format_pre_submit_feedback(Result(), contract)
    assert "[提交前自检未通过]" in feedback
    assert "预测文件不存在" in feedback
    assert "SELF-CHECK PASSED" in feedback


def test_skill_group_cv_check(tmp_path):
    contract = make_prediction_contract(
        tmp_path,
        tmp_path / "pred.csv",
        target_column="Churn",
        group_column="group_id",
    )
    state = {
        "artifact_contract": contract,
        "active_skill_keys": ["group_cv"],
        "code_history": ["model.fit(X, y)"],
    }
    checks = run_skill_checks(state)
    assert not checks[0]["passed"]
    assert "GroupKFold" in checks[0]["detail"]

    state["code_history"].append(
        "from sklearn.model_selection import StratifiedGroupKFold"
    )
    checks = run_skill_checks(state)
    assert checks[0]["passed"]


def test_skill_forbidden_features_check(tmp_path):
    contract = make_prediction_contract(
        tmp_path,
        tmp_path / "pred.csv",
        target_column="y",
        forbidden_features=["duration"],
    )
    state = {
        "artifact_contract": contract,
        "active_skill_keys": ["forbidden_features"],
        "code_history": ["X = df[['age', 'duration']]"],
    }
    checks = run_skill_checks(state)
    assert not checks[0]["passed"]

    state["code_history"] = ["X = df.drop(columns=['duration'])"]
    checks = run_skill_checks(state)
    assert checks[0]["passed"]


def test_forbidden_features_uses_manifest(tmp_path):
    pred = tmp_path / "pred.csv"
    pd.DataFrame({"id": [1], "y": [0.5]}).to_csv(pred, index=False)
    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(
        json.dumps({"features": ["age", "call_length"]}),
        encoding="utf-8",
    )
    contract = make_prediction_contract(
        tmp_path,
        pred,
        target_column="y",
        forbidden_features=["call_length"],
        feature_manifest_path=manifest,
    )
    state = {
        "artifact_contract": contract,
        "active_skill_keys": ["forbidden_features"],
        "code_history": [],
    }
    checks = run_skill_checks(state)
    assert not checks[0]["passed"]
    assert "特征清单" in checks[0]["detail"]

    result = verify_artifacts(contract, {"final_report": "done", "code_history": []})
    assert not result["passed"]
    assert any(
        check["name"] == "forbidden_features" and not check["passed"]
        for check in result["checks"]
    )


def test_self_check_includes_manifest(tmp_path):
    contract = make_prediction_contract(
        tmp_path,
        tmp_path / "pred.csv",
        target_column="y",
        feature_manifest_path=tmp_path / "feature_manifest.json",
    )
    raw = build_submission_self_check_code(contract)
    assert "manifest_path" in raw
    assert "features" in raw


def test_skill_probability_check(tmp_path):
    pred = tmp_path / "pred.csv"
    pd.DataFrame({"id": [1, 2, 3], "Churn": [0, 1, 0]}).to_csv(pred, index=False)
    contract = make_prediction_contract(
        tmp_path,
        pred,
        target_column="Churn",
        probability=True,
    )
    state = {
        "artifact_contract": contract,
        "active_skill_keys": ["probability_not_hard_labels"],
        "code_history": [],
    }
    checks = run_skill_checks(state)
    assert not checks[0]["passed"]

    pd.DataFrame({"id": [1, 2, 3], "Churn": [0.1, 0.8, 0.3]}).to_csv(pred, index=False)
    checks = run_skill_checks(state)
    assert checks[0]["passed"]


def test_skill_feedback_lists_only_failures(tmp_path):
    checks = [
        {"name": "skill:group_cv", "passed": False, "detail": "未使用分组交叉验证"},
        {"name": "skill:forbidden_features", "passed": True, "detail": "OK"},
    ]
    feedback = format_skill_check_feedback(checks)
    assert "[技能检查未通过]" in feedback
    assert "skill:group_cv" in feedback
    assert "skill:forbidden_features" not in feedback


def test_no_contract_passes_for_backward_compatibility():
    result = verify_artifacts({}, _state())
    assert result["passed"]
    assert result["checks"][0]["name"] == "contract"
