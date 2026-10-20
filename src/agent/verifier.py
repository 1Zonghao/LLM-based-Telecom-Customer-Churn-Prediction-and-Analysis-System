"""交付物契约与验证器。

设计目标：把“智能体说完成了”变成“交付物真的存在且合格”。
所有检查都返回结构化结果，既能阻断流程，也能作为反馈交给 LLM 修复。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    roc_auc_score,
)

MIN_REPORT_CHARS = 200


def make_prediction_contract(
    output_dir: str | Path,
    prediction_path: str | Path,
    report_path: str | Path | None = None,
    id_column: str = "id",
    target_column: str = "target",
    expected_rows: int | None = None,
    probability: bool = True,
    labels_path: str | Path | None = None,
    metric: str | None = None,
    min_metric: float | None = None,
    group_column: str | None = None,
    max_verification_retries: int = 2,
    require_final_report: bool = True,
) -> dict:
    """构造一个“预测任务”的标准交付物契约。"""
    required_files: list[dict] = [
        {
            "path": str(prediction_path),
            "format": "csv",
            "columns": [id_column, target_column],
            "id_column": id_column,
            "target_column": target_column,
            "expected_rows": expected_rows,
            "probability": probability,
        }
    ]
    if report_path is not None:
        required_files.append({
            "path": str(report_path),
            "format": "text",
            "min_chars": MIN_REPORT_CHARS,
        })
    return {
        "output_dir": str(output_dir),
        "required_files": required_files,
        "id_column": id_column,
        "target_column": target_column,
        "labels_path": str(labels_path) if labels_path else None,
        "metric": metric,
        "min_metric": min_metric,
        "group_column": group_column,
        "max_verification_retries": max_verification_retries,
        "require_final_report": require_final_report,
    }


def verify_artifacts(contract: dict | None, state: dict) -> dict:
    """按契约检查交付物，返回结构化验证结果。

    没有契约时必须通过，保证旧调用方（UI、batch_test）不被破坏。
    """
    if not contract or not contract.get("required_files"):
        return {
            "passed": True,
            "checks": [{
                "name": "contract",
                "passed": True,
                "detail": "未提供交付物契约，跳过验证",
            }],
            "artifacts": {},
            "metrics": {},
        }

    checks: list[dict] = []
    artifacts: dict[str, Any] = {}
    output_dir = Path(contract.get("output_dir", ".")).resolve()

    for spec in contract.get("required_files", []):
        path = Path(spec["path"])
        if not path.is_absolute():
            path = output_dir / path
        fmt = spec.get("format") or path.suffix.lstrip(".").lower()
        label = path.name

        exists = path.exists() and path.is_file() and path.stat().st_size > 0
        checks.append({
            "name": f"file_exists:{label}",
            "passed": exists,
            "detail": f"{path} {'存在' if exists else '不存在或为空'}",
        })
        if not exists:
            continue

        if fmt == "csv":
            _check_csv(path, spec, checks, artifacts)
        elif fmt in ("text", "md", "markdown"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            min_chars = int(spec.get("min_chars", MIN_REPORT_CHARS))
            passed = len(text.strip()) >= min_chars
            checks.append({
                "name": f"report_content:{label}",
                "passed": passed,
                "detail": f"报告长度 {len(text.strip())} 字符（要求 >= {min_chars}）",
            })
            artifacts[label] = {"chars": len(text)}

    metrics = _check_metric(contract, checks)
    _check_group_validation(contract, state, checks)

    require_final_report = contract.get("require_final_report", False)
    if require_final_report:
        has_report = bool(str(state.get("final_report", "")).strip())
        checks.append({
            "name": "final_report",
            "passed": has_report,
            "detail": "state.final_report " + ("非空" if has_report else "为空"),
        })

    return {
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
        "artifacts": artifacts,
        "metrics": metrics,
    }


def _check_csv(path: Path, spec: dict, checks: list[dict], artifacts: dict) -> None:
    label = path.name
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        checks.append({
            "name": f"csv_parse:{label}",
            "passed": False,
            "detail": f"CSV 解析失败: {type(exc).__name__}: {exc}",
        })
        return

    checks.append({
        "name": f"csv_parse:{label}",
        "passed": True,
        "detail": f"解析成功，{len(df)} 行 × {len(df.columns)} 列",
    })

    required_columns = spec.get("columns", [])
    missing = [c for c in required_columns if c not in df.columns]
    checks.append({
        "name": f"columns:{label}",
        "passed": not missing,
        "detail": "列齐全" if not missing else f"缺少列: {missing}",
    })
    if missing:
        return

    expected_rows = spec.get("expected_rows")
    if expected_rows is not None:
        passed = len(df) == int(expected_rows)
        checks.append({
            "name": f"row_count:{label}",
            "passed": passed,
            "detail": f"实际 {len(df)} 行，要求 {expected_rows} 行",
        })

    id_column = spec.get("id_column", "id")
    if id_column in df.columns:
        duplicates = int(df[id_column].duplicated().sum())
        checks.append({
            "name": f"unique_id:{label}",
            "passed": duplicates == 0,
            "detail": f"重复 id {duplicates} 个",
        })

    target_column = spec.get("target_column")
    if target_column and target_column in df.columns:
        missing_values = int(df[target_column].isna().sum())
        checks.append({
            "name": f"no_missing_predictions:{label}",
            "passed": missing_values == 0,
            "detail": f"预测列缺失值 {missing_values} 个",
        })
        if spec.get("probability"):
            numeric = pd.to_numeric(df[target_column], errors="coerce")
            valid = numeric.notna() & (numeric >= 0) & (numeric <= 1)
            checks.append({
                "name": f"probability_range:{label}",
                "passed": bool(valid.all()),
                "detail": "预测值均在 [0,1]" if valid.all() else "存在非概率值或越界值",
            })

    artifacts[label] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
    }


def _check_metric(contract: dict, checks: list[dict]) -> dict:
    labels_path = contract.get("labels_path")
    metric = contract.get("metric")
    if not labels_path or not metric:
        return {}

    pred_spec = next(
        (s for s in contract.get("required_files", []) if s.get("format") == "csv"),
        None,
    )
    if pred_spec is None:
        checks.append({
            "name": "metric",
            "passed": False,
            "detail": "契约要求计算指标，但没有找到 CSV 预测文件",
        })
        return {}

    try:
        output_dir = Path(contract.get("output_dir", ".")).resolve()
        labels_file = Path(labels_path)
        if not labels_file.is_absolute():
            labels_file = output_dir / labels_file
        pred_file = Path(pred_spec["path"])
        if not pred_file.is_absolute():
            pred_file = output_dir / pred_file
        labels = pd.read_csv(labels_file)
        predictions = pd.read_csv(pred_file)
        id_column = contract.get("id_column", "id")
        target_column = contract.get("target_column")
        if target_column not in labels.columns or target_column not in predictions.columns:
            raise ValueError(
                f"标签或预测缺少目标列 {target_column!r}"
            )

        labels[target_column] = _to_numeric_target(labels[target_column])
        pred_target = predictions[target_column]
        if pd.api.types.is_numeric_dtype(pred_target):
            # 概率预测：检测是否为 0/1 硬标签
            unique_values = set(pd.unique(pred_target.dropna()))
            if unique_values <= {0, 1} and metric in ("roc_auc", "auc"):
                raise ValueError("AUC 需要概率值，但预测列看起来是 0/1 硬标签")

        merged = labels[[id_column, target_column]].merge(
            predictions[[id_column, target_column]],
            on=id_column,
            how="inner",
            suffixes=("_true", "_pred"),
            validate="one_to_one",
        )
        if len(merged) != len(labels):
            raise ValueError(
                f"标签与预测 id 不完全对齐：标签 {len(labels)} 行，匹配 {len(merged)} 行"
            )

        y_true = merged[f"{target_column}_true"]
        y_pred = merged[f"{target_column}_pred"]
        score = _compute_metric(y_true, y_pred, metric)
        checks.append({
            "name": "metric",
            "passed": contract.get("min_metric") is None or score >= contract["min_metric"],
            "detail": (
                f"{metric} = {score:.6f}"
                + (f"（要求 >= {contract['min_metric']}）" if contract.get("min_metric") is not None else "")
            ),
        })
        return {"name": metric, "value": round(float(score), 6)}
    except Exception as exc:
        checks.append({
            "name": "metric",
            "passed": False,
            "detail": f"指标计算失败: {type(exc).__name__}: {exc}",
        })
        return {}


def _check_group_validation(contract: dict, state: dict, checks: list[dict]) -> None:
    """有分组列时，检查代码里是否使用了分组交叉验证。"""
    group_column = contract.get("group_column")
    if not group_column:
        return
    code = "\n".join(state.get("code_history", []))
    markers = ("GroupKFold", "StratifiedGroupKFold", "groups=", "group_ids")
    passed = any(marker in code for marker in markers)
    checks.append({
        "name": "group_validation",
        "passed": passed,
        "detail": (
            "检测到分组交叉验证"
            if passed
            else f"契约要求按 {group_column!r} 分组，但代码中未发现 GroupKFold/StratifiedGroupKFold/groups="
        ),
    })


def _compute_metric(y_true: pd.Series, y_pred: pd.Series, metric: str) -> float:
    metric = metric.lower()
    if metric in ("roc_auc", "auc"):
        return float(roc_auc_score(y_true, pd.to_numeric(y_pred, errors="coerce")))
    if metric in ("f1", "f1_score"):
        return float(f1_score(y_true, pd.to_numeric(y_pred, errors="coerce") >= 0.5))
    if metric == "recall":
        return float(recall_score(y_true, pd.to_numeric(y_pred, errors="coerce") >= 0.5))
    if metric == "accuracy":
        return float(accuracy_score(y_true, pd.to_numeric(y_pred, errors="coerce") >= 0.5))
    if metric in ("balanced_accuracy", "balanced_accuracy_score"):
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    raise ValueError(f"不支持的指标: {metric}")


def _to_numeric_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    mapped = series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    if mapped.notna().all():
        return mapped.astype(int)
    return pd.to_numeric(series, errors="coerce")


def format_verification_feedback(result: dict) -> str:
    """把验证失败结果转成给 LLM 的修复指令。"""
    lines = ["[验证失败] 交付物检查未通过："]
    for check in result.get("checks", []):
        if not check.get("passed"):
            lines.append(f"- {check['name']}: {check.get('detail', '')}")
    lines.append("")
    lines.append(
        "请先修复以上问题，实际重新生成交付物，并再次输出最终报告。"
        "不要只解释方案，必须写文件。"
    )
    return "\n".join(lines)


def format_verification_report(result: dict) -> str:
    """把验证结果转成简短的最终报告（用于重试耗尽时）。"""
    lines = ["### 交付验证未通过", ""]
    for check in result.get("checks", []):
        status = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- **{status}** `{check['name']}` — {check.get('detail', '')}")
    metrics = result.get("metrics") or {}
    if metrics:
        lines.append("")
        lines.append(f"指标：`{metrics.get('name')}` = {metrics.get('value')}")
    lines.append("")
    lines.append("智能体在最大重试次数内未能生成合格的交付物，请人工检查。")
    return "\n".join(lines)
