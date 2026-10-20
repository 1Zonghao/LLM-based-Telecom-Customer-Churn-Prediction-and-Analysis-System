"""Kaggle Playground S6E3 离线评测脚本。

提供四个子命令：
    prepare   从官方 train.csv 生成固定的分层训练/留出集
    baseline  训练一个确定性的 XGBoost 基线并评分（不依赖 LLM）
    agent     运行自主数据科学智能体，并对其留出集预测评分（需要 LLM_API_KEY）
    score     对已有的预测文件评分

用法:
    python src/competition_eval.py prepare
    python src/competition_eval.py baseline
    python src/competition_eval.py agent --max-iter 20
    python src/competition_eval.py score --pred outputs/competition_eval/agent_predictions.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmarks" / "kaggle_s6e3"
EVAL_DIR = PROJECT_ROOT / "data" / "benchmarks" / "kaggle_s6e3_eval"
OUT_DIR = PROJECT_ROOT / "outputs" / "competition_eval"

DEV_TRAIN = EVAL_DIR / "dev_train.csv"
HOLDOUT_FEATURES = EVAL_DIR / "holdout_features.csv"
HOLDOUT_LABELS = EVAL_DIR / "holdout_labels.csv"
AGENT_PRED = OUT_DIR / "agent_predictions.csv"
AGENT_REPORT = OUT_DIR / "agent_report.md"
BASELINE_PRED = OUT_DIR / "baseline_predictions.csv"


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

def _map_target(series: pd.Series) -> pd.Series:
    """把 Yes/No 或 0/1 统一映射为 0/1。"""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    return (
        series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype(int)
    )


def _normalize_id(series: pd.Series) -> pd.Series:
    """统一 id 类型，避免 int/float/string 造成的合并失败。"""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("int64").astype(str)
    text = series.astype(str).str.strip()
    numeric_like = pd.to_numeric(text, errors="coerce")
    if numeric_like.notna().all() and (numeric_like % 1 == 0).all():
        return numeric_like.astype("int64").astype(str)
    return text


def cmd_prepare(args: argparse.Namespace) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    if not train_path.exists():
        raise SystemExit(f"找不到 {train_path}，请先下载 S6E3 数据")

    train = pd.read_csv(train_path)
    if "Churn" not in train.columns:
        raise SystemExit("train.csv 中缺少 Churn 列")
    train["Churn"] = _map_target(train["Churn"])

    dev_idx, hold_idx = train_test_split(
        train.index,
        test_size=args.holdout_size,
        random_state=args.seed,
        stratify=train["Churn"],
    )
    dev = train.loc[dev_idx].reset_index(drop=True)
    hold = train.loc[hold_idx].reset_index(drop=True)

    dev.to_csv(DEV_TRAIN, index=False)
    hold.drop(columns=["Churn"]).to_csv(HOLDOUT_FEATURES, index=False)
    hold[["id", "Churn"]].to_csv(HOLDOUT_LABELS, index=False)

    test = pd.read_csv(test_path)
    summary = {
        "source_train_rows": int(len(train)),
        "source_test_rows": int(len(test)),
        "churn_rate": round(float(train["Churn"].mean()), 4),
        "dev_rows": int(len(dev)),
        "holdout_rows": int(len(hold)),
        "holdout_churn_rate": round(float(hold["Churn"].mean()), 4),
        "features": [c for c in train.columns if c not in ("id", "Churn")],
        "seed": args.seed,
    }
    _write_json(OUT_DIR / "split_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 评分
# ---------------------------------------------------------------------------

def _normalize_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in pred.columns}
    if "id" not in cols:
        raise ValueError(f"预测文件缺少 id 列，实际列: {list(pred.columns)}")
    id_col = cols["id"]
    target_col = None
    for name in ("churn", "prediction", "probability", "proba", "target"):
        if name in cols:
            target_col = cols[name]
            break
    if target_col is None:
        # 回退：取除 id 外的第一列
        others = [c for c in pred.columns if c != id_col]
        if not others:
            raise ValueError("预测文件除 id 外没有预测列")
        target_col = others[0]
    out = pred[[id_col, target_col]].copy()
    out.columns = ["id", "prediction"]
    return out


def score_predictions(pred_path: Path, label_path: Path = HOLDOUT_LABELS) -> dict:
    pred = _normalize_predictions(pd.read_csv(pred_path))
    labels = pd.read_csv(label_path)
    labels["Churn"] = _map_target(labels["Churn"])
    pred["id"] = _normalize_id(pred["id"])
    labels["id"] = _normalize_id(labels["id"])

    merged = labels.merge(pred, on="id", how="left", validate="one_to_one")
    missing = int(merged["prediction"].isna().sum())
    if missing:
        raise ValueError(f"有 {missing} 个留出样本没有预测值")

    y_true = merged["Churn"].to_numpy()
    raw = merged["prediction"].to_numpy()
    is_probability = np.issubdtype(raw.dtype, np.number) and raw.min() >= 0 and raw.max() <= 1
    if not is_probability:
        # 硬标签：AUC 无意义，只做阈值指标
        y_pred = raw.astype(int)
        metrics = {"auc": None, "average_precision": None, "is_probability": False}
    else:
        y_prob = raw.astype(float)
        metrics = {
            "auc": round(float(roc_auc_score(y_true, y_prob)), 6),
            "average_precision": round(float(average_precision_score(y_true, y_prob)), 6),
            "is_probability": True,
        }
        y_pred = (y_prob >= 0.5).astype(int)
        metrics.update(_threshold_sweep(y_true, y_prob))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics.update({
        "threshold_0_5": {
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
        },
        "n_samples": int(len(merged)),
        "positive_rate": round(float(y_true.mean()), 6),
        "missing_predictions": missing,
    })
    return metrics


def _threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """在留出集上扫描阈值，报告最优 F1 与 Recall>=0.75 下的最优 F1。

    注意：阈值是在留出集上调的，属于乐观上界；正式报告需注明。
    """
    best_f1 = {"threshold": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    best_f1_r75 = {"threshold": 0.0, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in np.linspace(0.01, 0.99, 197):
        y_pred = (y_prob >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        row = {
            "threshold": round(float(t), 4),
            "f1": round(float(f1), 6),
            "precision": round(float(p), 6),
            "recall": round(float(r), 6),
        }
        if f1 > best_f1["f1"]:
            best_f1 = row
        if r >= 0.75 and f1 > best_f1_r75["f1"]:
            best_f1_r75 = row
    return {
        "best_f1_on_holdout": best_f1,
        "best_f1_at_recall_ge_0_75_on_holdout": best_f1_r75,
    }


def cmd_score(args: argparse.Namespace) -> None:
    metrics = score_predictions(Path(args.pred))
    _write_json(OUT_DIR / "score.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 确定性基线
# ---------------------------------------------------------------------------

def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = [c for c in df.columns if c not in ("id", "Churn")]
    categorical = [c for c in features if df[c].dtype == object]
    numeric = [c for c in features if c not in categorical]
    return categorical, numeric


def train_baseline(seed: int = 42) -> dict:
    dev = pd.read_csv(DEV_TRAIN)
    holdout = pd.read_csv(HOLDOUT_FEATURES)

    categorical, numeric = _feature_columns(dev)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical),
            ("num", "passthrough", numeric),
        ]
    )
    X_dev = preprocessor.fit_transform(dev[categorical + numeric])
    X_hold = preprocessor.transform(holdout[categorical + numeric])
    y_dev = dev["Churn"].to_numpy()

    neg, pos = int((y_dev == 0).sum()), int((y_dev == 1).sum())
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / max(pos, 1),
        tree_method="hist",
        eval_metric="auc",
        n_jobs=-1,
        random_state=seed,
    )

    started = time.time()
    model.fit(X_dev, y_dev)
    fit_seconds = round(time.time() - started, 2)

    proba = model.predict_proba(X_hold)[:, 1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": holdout["id"], "Churn": proba}).to_csv(BASELINE_PRED, index=False)
    metrics = score_predictions(BASELINE_PRED)
    metrics.update({
        "model": "XGBoost baseline",
        "fit_seconds": fit_seconds,
        "n_features_after_encoding": int(X_dev.shape[1]),
        "scale_pos_weight": round(neg / max(pos, 1), 4),
    })
    _write_json(OUT_DIR / "baseline_result.json", metrics)
    return metrics


def cmd_baseline(args: argparse.Namespace) -> None:
    metrics = train_baseline(seed=args.seed)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# 智能体运行
# ---------------------------------------------------------------------------

def _build_task_prompt(
    dev_path: Path,
    holdout_features: Path,
    pred_path: Path,
    report_path: Path,
) -> str:
    n_holdout = len(pd.read_csv(holdout_features, usecols=["id"]))
    return f"""\
你正在参加 Kaggle Playground S6E3 电信客户流失预测的离线评测。请自主完成完整的数据分析流程。

- 训练集（含标签）：{dev_path}
- 待预测集（无标签）：{holdout_features}
- 目标变量：Churn（0=未流失，1=流失）
- 业务约束：漏报流失客户的代价远大于误报，请优先保证 Recall，同时兼顾 F1。
- 数据规模约 47.5 万行。请使用 class_weight='balanced' 或 XGBoost 的 scale_pos_weight 处理类别不平衡，不要使用 SMOTE（避免内存和耗时问题）。
- 在训练集内部做分层验证，至少比较 2 种模型，报告 ROC-AUC、Recall、F1，并说明业务阈值的选择依据。
- 最终必须：对待预测集的每一行输出流失概率，写入 {pred_path}（CSV，包含 id 和 Churn 两列，共 {n_holdout} 行，顺序与待预测集一致）。写完后用代码打印文件路径、行数和前 3 行，确认文件已生成。
- 同时将最终报告写入 {report_path}（Markdown）。
- 重要：评测脚本会在你结束后检查这两个文件。如果文件不存在，本次运行直接判为失败。因此**在输出最终报告之前，必须先用代码实际生成这些文件并确认**，不要只描述方案。
- 最后再输出最终报告，包含模型对比表和业务建议；报告第一行必须写明预测文件和报告文件已生成及其路径。
"""


class AnthropicLLMClient:
    """Anthropic Messages API 适配器（用于 DeepSeek 的 /anthropic 端点）。

    与 LLMClient 保持相同的 chat() 接口，返回 (文本, token_usage)。
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 8192,
    ):
        self.model = model or os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro[1m]")
        self.api_key = api_key or os.getenv("ANTHROPIC_AUTH_TOKEN", "")
        self.base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL", "")).rstrip("/")
        self.max_tokens = max_tokens
        self.max_retries = 2

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.base_url)

    def chat(self, messages: list[dict], temperature: float = 0.1) -> tuple[str, dict]:
        system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
        convo: list[dict] = []
        for message in messages:
            role = message.get("role")
            if role not in ("user", "assistant"):
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(item.get("text", item)) if isinstance(item, dict) else str(item)
                    for item in content
                )
            if convo and convo[-1]["role"] == role:
                convo[-1]["content"] += "\n\n" + content
            else:
                convo.append({"role": role, "content": content})
        if not convo:
            convo = [{"role": "user", "content": ""}]

        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": convo,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(
                    self.base_url + "/v1/messages",
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    body = json.loads(response.read().decode("utf-8"))
                text = "".join(
                    block.get("text", "")
                    for block in body.get("content", [])
                    if block.get("type") == "text"
                )
                usage_raw = body.get("usage", {})
                prompt_tokens = int(usage_raw.get("input_tokens", 0) or 0)
                completion_tokens = int(usage_raw.get("output_tokens", 0) or 0)
                if not text.strip():
                    raise RuntimeError(
                        "Anthropic response contained no text "
                        f"(stop_reason={body.get('stop_reason')})"
                    )
                return text, {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            except urllib.error.HTTPError as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        raise RuntimeError(
            f"Anthropic API request failed after {self.max_retries + 1} attempts: {last_error}"
        )


def _build_llm(provider: str):
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from agent.llm_client import LLMClient

    if provider == "auto":
        provider = "anthropic" if os.getenv("ANTHROPIC_AUTH_TOKEN") else "openai"
    if provider == "anthropic":
        return AnthropicLLMClient()
    return LLMClient()


def _run_agent(
    max_iter: int,
    task: str,
    dev_path: Path,
    provider: str = "auto",
    artifact_contract: dict | None = None,
) -> dict:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from agent import nodes
    from agent.graph import build_agent, make_thread_config
    from agent.state import init_state
    from sandbox.compressor import OutputCompressor
    from sandbox.kernel_manager import KernelManager

    class LongTimeoutKernelManager(KernelManager):
        def execute(self, code: str, timeout: int = 600):
            return super().execute(code, timeout)

    nodes.MAX_ITERATIONS = max_iter

    llm = _build_llm(provider)
    if not llm.is_configured:
        raise SystemExit("LLM 未配置，无法运行智能体")

    km = LongTimeoutKernelManager()
    km.start()
    compressor = OutputCompressor()
    app = build_agent(llm, km, compressor)
    config = make_thread_config()
    state = init_state(
        task,
        str(dev_path),
        domain="telco_churn",
        artifact_contract=artifact_contract,
    )

    transcript: list[dict] = []
    reasoning_calls = 0
    iteration = 0
    hard_cap = max_iter + 4  # 验证重试会额外消耗动作预算
    started = time.time()
    finished_normally = False
    error = ""

    try:
        inputs = state
        hit_cap = False
        while True:
            for event in app.stream(inputs, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        continue
                    if node_name == "reasoning":
                        reasoning_calls += 1
                    if isinstance(node_output, dict):
                        iteration = max(iteration, int(node_output.get("iteration", 0) or 0))
                    transcript.append({
                        "node": node_name,
                        "output": _truncate_output(node_output),
                        "t": round(time.time() - started, 1),
                    })
                    print(f"[{time.time() - started:7.1f}s] node={node_name} "
                          f"iter={iteration} reason_calls={reasoning_calls}", flush=True)
                    if reasoning_calls >= hard_cap:
                        hit_cap = True
                        break
                if hit_cap:
                    print(f"  [STOP] reasoning call cap reached: {reasoning_calls}", flush=True)
                    break

            if hit_cap:
                break

            snapshot = app.get_state(config)
            values = snapshot.values if hasattr(snapshot, "values") else snapshot
            next_nodes = tuple(getattr(snapshot, "next", ()) or ())

            if next_nodes and any(str(n).startswith("checkpoint") for n in next_nodes):
                print(f"  [CHECKPOINT] {next_nodes} — auto-confirming")
                inputs = None
                if reasoning_calls >= hard_cap:
                    print(f"  [STOP] reasoning call cap reached: {reasoning_calls}")
                    break
                continue

            finished_normally = True
            break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        duration = round(time.time() - started, 2)
        try:
            final_values = app.get_state(config).values
        except Exception:
            final_values = state
        try:
            km.shutdown()
        except Exception:
            pass

    token_usage = final_values.get("token_usage", {}) if isinstance(final_values, dict) else {}
    result = {
        "duration_sec": duration,
        "reasoning_calls": reasoning_calls,
        "iterations": iteration,
        "finished_normally": finished_normally,
        "error": error,
        "token_usage": token_usage,
        "started_at": final_values.get("started_at", "") if isinstance(final_values, dict) else "",
        "phase": final_values.get("phase", "") if isinstance(final_values, dict) else "",
        "data_profile": final_values.get("data_profile", {}) if isinstance(final_values, dict) else {},
        "final_report": (final_values.get("final_report", "") if isinstance(final_values, dict) else "")[:5000],
        "code_history": (final_values.get("code_history", []) if isinstance(final_values, dict) else []),
        "verification_result": final_values.get("verification_result", {}) if isinstance(final_values, dict) else {},
        "verification_passed": final_values.get("verification_passed", None) if isinstance(final_values, dict) else None,
        "verification_retries": final_values.get("verification_retries", 0) if isinstance(final_values, dict) else 0,
        "transcript": transcript,
    }
    return result


def _truncate_output(output, limit: int = 1200):
    if not isinstance(output, dict):
        return str(output)[:limit]
    return {
        k: (str(v)[:limit] if not isinstance(v, (int, float, bool, type(None))) else v)
        for k, v in output.items()
    }


def salvage_agent_pipeline() -> dict:
    """用智能体保存的 processed_data.pkl 复现其最终方案，生成留出集预测。

    智能体经常在完成训练后直接输出报告而忘记写预测文件。此函数加载它保存的
    特征工程结果，按它在代码中选择的 XGBoost 配置重新拟合，并对留出集评分。
    """
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    processed_path = PROJECT_ROOT / "processed_data.pkl"
    if not processed_path.exists():
        raise FileNotFoundError(f"找不到 {processed_path}")

    processed = joblib.load(processed_path)
    X = processed["X"]
    y = processed["y"]
    numeric_features = list(processed["numeric_features"])
    categorical_features = list(processed["categorical_features"])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
        ]
    )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(1 - y.mean()) / y.mean(),
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss",
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    started = time.time()
    pipeline.fit(X, y)
    fit_seconds = round(time.time() - started, 2)

    holdout = pd.read_csv(HOLDOUT_FEATURES)
    for col in [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]:
        holdout[col] = holdout[col].replace("No internet service", "No")
    holdout["MultipleLines"] = holdout["MultipleLines"].replace("No phone service", "No")
    holdout["SeniorCitizen"] = holdout["SeniorCitizen"].astype(str)
    holdout["AvgMonthlyCharges"] = np.where(
        holdout["tenure"] == 0,
        holdout["MonthlyCharges"],
        holdout["TotalCharges"] / holdout["tenure"],
    )

    proba = pipeline.predict_proba(holdout[numeric_features + categorical_features])[:, 1]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": holdout["id"], "Churn": proba}).to_csv(AGENT_PRED, index=False)
    return {
        "source": str(processed_path),
        "fit_seconds": fit_seconds,
        "n_rows": int(len(holdout)),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }


def cmd_salvage(args: argparse.Namespace) -> None:
    info = salvage_agent_pipeline()
    metrics = score_predictions(AGENT_PRED)
    payload = {"salvage": info, "holdout_metrics": metrics}
    _write_json(OUT_DIR / "salvage_result.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_agent(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (DEV_TRAIN, HOLDOUT_FEATURES, HOLDOUT_LABELS):
        if not path.exists():
            raise SystemExit(f"缺少 {path}，请先运行 prepare")

    for path in (AGENT_PRED, AGENT_REPORT):
        if path.exists():
            path.unlink()

    n_holdout = len(pd.read_csv(HOLDOUT_FEATURES, usecols=["id"]))
    from agent.verifier import make_prediction_contract

    contract = make_prediction_contract(
        output_dir=OUT_DIR,
        prediction_path=AGENT_PRED,
        report_path=AGENT_REPORT,
        id_column="id",
        target_column="Churn",
        expected_rows=n_holdout,
        probability=True,
        labels_path=HOLDOUT_LABELS,
        metric="roc_auc",
        min_metric=0.80,
        max_verification_retries=2,
    )

    task = _build_task_prompt(DEV_TRAIN, HOLDOUT_FEATURES, AGENT_PRED, AGENT_REPORT)
    print("=" * 70)
    print(task)
    print("=" * 70)

    try:
        result = _run_agent(
            args.max_iter, task, DEV_TRAIN, args.provider,
            artifact_contract=contract,
        )
    except SystemExit:
        raise
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}

    if not AGENT_PRED.exists():
        # 智能体未写预测文件时，用它保存的特征工程结果和模型配置补写
        try:
            result["salvage"] = salvage_agent_pipeline()
            result["note"] = "智能体未写预测文件，已用其保存的 processed_data.pkl 补写"
        except Exception as exc:
            result["salvage_error"] = f"{type(exc).__name__}: {exc}"

    if not AGENT_REPORT.exists():
        fallback_report = result.get("final_report") or "智能体未生成报告文件。"
        AGENT_REPORT.write_text(fallback_report, encoding="utf-8")
        result["report_fallback_written"] = str(AGENT_REPORT)

    if AGENT_PRED.exists():
        try:
            result["holdout_metrics"] = score_predictions(AGENT_PRED)
        except Exception as exc:
            result["holdout_metrics_error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["holdout_metrics"] = None
        result.setdefault("note", f"智能体未生成 {AGENT_PRED}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"agent_result_{timestamp}.json"
    _write_json(out_path, result)
    summary = {k: result.get(k) for k in (
        "duration_sec", "reasoning_calls", "iterations", "finished_normally",
        "error", "token_usage", "phase", "verification_passed",
        "verification_retries", "holdout_metrics",
    )}
    print("\n" + "=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n完整结果: {out_path}")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="S6E3 competition evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="生成固定训练/留出集")
    p_prepare.add_argument("--holdout-size", type=float, default=0.2)
    p_prepare.add_argument("--seed", type=int, default=42)
    p_prepare.set_defaults(func=cmd_prepare)

    p_baseline = sub.add_parser("baseline", help="运行确定性 XGBoost 基线")
    p_baseline.add_argument("--seed", type=int, default=42)
    p_baseline.set_defaults(func=cmd_baseline)

    p_agent = sub.add_parser("agent", help="运行自主数据科学智能体")
    p_agent.add_argument("--max-iter", type=int, default=20)
    p_agent.add_argument(
        "--provider",
        choices=["auto", "openai", "anthropic"],
        default="auto",
        help="auto: 优先使用 ANTHROPIC_AUTH_TOKEN；也可强制 openai/anthropic",
    )
    p_agent.set_defaults(func=cmd_agent)

    p_salvage = sub.add_parser("salvage", help="用智能体保存的 pipeline 补写预测并评分")
    p_salvage.set_defaults(func=cmd_salvage)

    p_score = sub.add_parser("score", help="对预测文件评分")
    p_score.add_argument("--pred", type=str, required=True)
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
