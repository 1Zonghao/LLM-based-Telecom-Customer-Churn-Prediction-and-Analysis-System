"""交付验证器单元测试（不依赖 LLM / Jupyter kernel）。"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.verifier import (
    build_submission_self_check,
    format_verification_feedback,
    make_prediction_contract,
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
    assert "os.path.getsize(report_path)" in feedback


def test_self_check_omits_row_assert_when_not_specified(tmp_path):
    contract = make_prediction_contract(
        tmp_path,
        tmp_path / "pred.csv",
        target_column="Churn",
    )
    snippet = build_submission_self_check(contract)
    assert "expected_rows" in snippet
    assert "行数错误" not in snippet


def test_no_contract_passes_for_backward_compatibility():
    result = verify_artifacts({}, _state())
    assert result["passed"]
    assert result["checks"][0]["name"] == "contract"
