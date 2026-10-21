"""第二阶段实验场：多数据集、多轮次、skill library 开/关对比。

用法:
    python src/experiment_runner.py --datasets adult,bank --rounds 2 --skill-mode on
    python src/experiment_runner.py --datasets adult,bank --rounds 2 --skill-mode off
    python src/experiment_runner.py --datasets diabetes --rounds 1 --skill-mode on

每个 episode 都会：
1. 准备固定训练/留出切分（缓存在 data/benchmarks/gym/<name>/）
2. 检索 skill library 并注入任务提示
3. 运行 verifier-gated 智能体
4. 用留出标签评分（AUC / F1 / Recall）
5. 从验证失败中沉淀新 skill
6. 保存单次结果和汇总表
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from agent.episode import run_episode
from agent.llm_client import build_llm_client
from agent.skills import SkillLibrary
from agent.verifier import make_prediction_contract
from competition_eval import score_predictions

GYM_DIR = PROJECT_ROOT / "data" / "benchmarks" / "gym"
OUT_DIR = PROJECT_ROOT / "outputs" / "experiments"
DEFAULT_SKILL_PATH = PROJECT_ROOT / "outputs" / "skills" / "skills.json"


# ---------------------------------------------------------------------------
# 数据集注册表
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "telco": {
        "source": PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "sep": ",",
        "target": "Churn",
        "id_column": "customerID",
        "positive": "Yes",
        "task": "电信客户流失预测",
        "business": "漏报流失客户的代价远大于误报，优先保证 Recall，同时兼顾 F1 和 AUC。",
        "domain": "telco_churn",
        "tags": ["binary_classification", "churn", "imbalanced", "tabular"],
        "min_auc": 0.80,
        "held_out": False,
    },
    "bank": {
        "source": PROJECT_ROOT / "data" / "uci_datasets" / "bank-additional" / "bank-additional-full.csv",
        "sep": ";",
        "target": "y",
        "id_column": None,
        "positive": "yes",
        "task": "银行营销响应预测",
        "business": "正类比例低，漏报潜在响应客户的代价高，优先 Recall。注意 duration 是通话结束后才知道的，不能作为特征（会造成泄漏）。",
        "domain": "generic",
        "tags": ["binary_classification", "marketing", "imbalanced", "tabular"],
        "min_auc": 0.75,
        "held_out": False,
        # duration 是通话结束后才知道的，作为特征会造成泄漏
        "drop_columns": ["duration"],
    },
    "adult": {
        "source": PROJECT_ROOT / "data" / "uci_datasets" / "adult.csv",
        "sep": ",",
        "target": "income",
        "id_column": None,
        "positive": ">50K",
        "task": "成人收入预测",
        "business": "预测个人年收入是否超过 50K；这是类别不平衡的二分类任务，关注 Recall 与 F1。",
        "domain": "generic",
        "tags": ["binary_classification", "income", "imbalanced", "tabular"],
        "min_auc": 0.85,
        "held_out": False,
    },
    "diabetes": {
        "source": PROJECT_ROOT / "data" / "uci_datasets" / "diabetes.csv",
        "sep": ",",
        "target": "Outcome",
        "id_column": None,
        "positive": 1,
        "task": "糖尿病预测（held-out）",
        "business": "漏诊代价高于误诊，优先 Recall。",
        "domain": "generic",
        "tags": ["binary_classification", "health", "imbalanced", "tabular"],
        "min_auc": 0.75,
        "held_out": True,
    },
}


def _map_target(series: pd.Series, positive) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return (series == positive).astype(int)
    text = series.astype(str).str.strip()
    return (text.str.lower() == str(positive).lower()).astype(int)


def prepare_dataset(
    name: str,
    output_root: Path = GYM_DIR,
    holdout_size: float = 0.2,
    seed: int = 42,
    force: bool = False,
) -> dict:
    """准备固定切分并返回路径/元数据。已存在时直接复用。"""
    if name not in DATASETS:
        raise KeyError(f"未知数据集: {name}")

    spec = DATASETS[name]
    out_dir = output_root / name
    train_path = out_dir / "train.csv"
    test_features = out_dir / "test_features.csv"
    test_labels = out_dir / "test_labels.csv"
    meta_path = out_dir / "meta.json"

    if (
        not force
        and train_path.exists()
        and test_features.exists()
        and test_labels.exists()
        and meta_path.exists()
    ):
        return json.loads(meta_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(spec["source"], sep=spec["sep"])

    dropped_columns: list[str] = []
    for column in spec.get("drop_columns", []):
        if column in df.columns:
            df = df.drop(columns=[column])
            dropped_columns.append(column)

    if spec.get("id_column") and spec["id_column"] in df.columns:
        df = df.rename(columns={spec["id_column"]: "id"})
    else:
        df.insert(0, "id", range(len(df)))

    target = spec["target"]
    if target not in df.columns:
        raise ValueError(f"{name}: 找不到目标列 {target}")

    # 目标映射（Yes/No、>50K、0/1）
    original_target = df[target].copy()
    df[target] = _map_target(original_target, spec["positive"])

    # 常见缺失标记处理（adult 的 '?'）
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].replace("?", np.nan)

    df = df.dropna(subset=[target]).reset_index(drop=True)

    train_df, test_df = train_test_split(
        df,
        test_size=holdout_size,
        random_state=seed,
        stratify=df[target],
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(train_path, index=False)
    test_df.drop(columns=[target]).to_csv(test_features, index=False)
    test_df[["id", target]].to_csv(test_labels, index=False)

    meta = {
        "name": name,
        "task": spec["task"],
        "business": spec["business"],
        "domain": spec["domain"],
        "tags": spec["tags"],
        "target": target,
        "positive": str(spec["positive"]),
        "min_auc": spec["min_auc"],
        "held_out": spec["held_out"],
        "source": str(spec["source"]),
        "train_path": str(train_path),
        "test_features_path": str(test_features),
        "test_labels_path": str(test_labels),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "positive_rate": round(float(train_df[target].mean()), 4),
        "dropped_columns": dropped_columns,
        "seed": seed,
        "holdout_size": holdout_size,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------------------
# 任务 / 契约
# ---------------------------------------------------------------------------

def build_paths(name: str, result_dir: Path) -> dict:
    run_dir = result_dir / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "predictions": run_dir / "predictions.csv",
        "report": run_dir / "report.md",
    }


SELF_CHECK_TEMPLATE = '''\
```python
import os
import pandas as pd

pred_path = r"__PRED__"
report_path = r"__REPORT__"
target = "__TARGET__"
expected_rows = __ROWS__

assert os.path.exists(pred_path), "预测文件不存在"
pred = pd.read_csv(pred_path)
assert len(pred) == expected_rows, f"行数错误: {len(pred)}"
assert list(pred.columns) == ["id", target], f"列名错误: {list(pred.columns)}"
assert pred["id"].is_unique, "id 有重复"
assert pred[target].notna().all(), "预测列有缺失"
assert pred[target].between(0, 1).all(), "预测值不是概率"
assert os.path.getsize(report_path) > 200, "报告文件不存在或太短"
print("SELF-CHECK PASSED")
```'''


def build_self_check_snippet(meta: dict, paths: dict) -> str:
    return (
        SELF_CHECK_TEMPLATE
        .replace("__PRED__", str(paths["predictions"]))
        .replace("__REPORT__", str(paths["report"]))
        .replace("__TARGET__", meta["target"])
        .replace("__ROWS__", str(meta["n_test"]))
    )


def build_task(meta: dict, paths: dict, skill_keys: set[str] | None = None) -> str:
    target = meta["target"]
    task = f"""\
这是一个离线数据科学评测任务：{meta['task']}。

- 训练集（含标签）：{meta['train_path']}
- 待预测集（无标签）：{meta['test_features_path']}
- 目标列：{target}（0/1；1 表示正类）
- 评估指标：ROC-AUC（首要），同时报告 F1、Recall 和混淆矩阵。
- 交付物：把待预测集每一行的概率写入 {paths['predictions']}（CSV，列 id 和 {target}），并把最终报告写入 {paths['report']}（Markdown）。
- 数据规模：训练 {meta['n_train']} 行，待预测 {meta['n_test']} 行，正类比例约 {meta['positive_rate']:.2%}。
- 业务约束：{meta['business']}
- id 列只用于对齐，不要作为特征。
- 在输出最终报告之前，必须用代码实际生成上述两个文件并打印确认。
"""
    # 只有检索到相关经验时才注入可执行自检；off 组保持原任务不变
    if skill_keys and skill_keys & {"write_deliverables", "probability_not_hard_labels"}:
        task += f"""\

## 提交前自检（必须执行）
在输出最终报告之前，先运行下面这段自检代码；只有打印 `SELF-CHECK PASSED` 后才能结束。
{build_self_check_snippet(meta, paths)}
"""
    return task


def build_contract(meta: dict, paths: dict) -> dict:
    return make_prediction_contract(
        output_dir=paths["run_dir"],
        prediction_path=paths["predictions"],
        report_path=paths["report"],
        id_column="id",
        target_column=meta["target"],
        expected_rows=meta["n_test"],
        probability=True,
        labels_path=meta["test_labels_path"],
        metric="roc_auc",
        min_metric=meta["min_auc"],
        max_verification_retries=2,
    )


def build_skill_query(meta: dict) -> dict:
    return {
        "task": f"{meta['task']} {meta['business']}",
        "domain": meta["domain"],
        "dataset": meta["name"],
        "target_type": "binary_classification",
        "tags": meta["tags"],
    }


# ---------------------------------------------------------------------------
# 实验主循环
# ---------------------------------------------------------------------------

def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_root = (Path(args.results_dir) if args.results_dir else OUT_DIR) / f"{timestamp}_{args.skill_mode}"
    result_root.mkdir(parents=True, exist_ok=True)

    skill_path = Path(args.skill_db) if args.skill_db else DEFAULT_SKILL_PATH
    library = SkillLibrary(skill_path)
    if args.reset_skills:
        library.clear()
        print(f"[SKILLS] cleared {skill_path}")

    llm = build_llm_client(args.provider)
    if not llm.is_configured:
        raise SystemExit("LLM 未配置：检查 LLM_API_KEY 或 ANTHROPIC_AUTH_TOKEN")

    dataset_names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    unknown = [n for n in dataset_names if n not in DATASETS]
    if unknown:
        raise SystemExit(f"未知数据集: {unknown}")

    print(f"[RUN] mode={args.skill_mode} rounds={args.rounds} datasets={dataset_names}")
    print(f"[RUN] results -> {result_root}")
    print(f"[RUN] skills  -> {skill_path} ({len(library)} existing)")

    rows: list[dict] = []
    stopped = False

    for round_index in range(1, args.rounds + 1):
        if stopped:
            break
        # 每轮开始时冻结技能库：本轮新学到的经验只从下一轮开始生效，
        # 避免“同一轮里后跑的数据集”污染轮次对比。
        round_library = SkillLibrary(skill_path) if args.skill_mode == "on" else None
        for name in dataset_names:
            print("\n" + "=" * 78)
            print(f"[EPISODE] round={round_index} dataset={name} mode={args.skill_mode}")
            print("=" * 78, flush=True)

            meta = prepare_dataset(name, force=args.force_data)
            paths = build_paths(name, result_root)
            for key in ("predictions", "report"):
                if paths[key].exists():
                    paths[key].unlink()

            injected: list[dict] = []
            skill_context = "暂无"
            if args.skill_mode == "on":
                injected = round_library.retrieve(build_skill_query(meta), k=args.skill_top_k)
                skill_context = round_library.format_for_prompt(injected)
                print(f"[SKILLS] injected {len(injected)} skill(s)", flush=True)
                for skill in injected:
                    print(f"  - {skill['key']}: {skill['lesson'][:80]}", flush=True)

            skill_keys = {skill["key"] for skill in injected}
            task = build_task(meta, paths, skill_keys=skill_keys)
            contract = build_contract(meta, paths)
            started = time.time()

            try:
                result = run_episode(
                    llm=llm,
                    task=task,
                    data_path=meta["train_path"],
                    domain=meta["domain"],
                    artifact_contract=contract,
                    max_iter=args.max_iter,
                    skill_context=skill_context,
                    kernel_factory=None,
                )
            except Exception as exc:
                result = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                    "verification_passed": False,
                    "verification_history": [],
                }

            result.update({
                "dataset": name,
                "round": round_index,
                "skill_mode": args.skill_mode,
                "skill_context": skill_context,
                "skills_injected": [s["key"] for s in injected],
                "wall_clock_sec": round(time.time() - started, 2),
            })

            if paths["predictions"].exists():
                try:
                    result["holdout_metrics"] = score_predictions(
                        paths["predictions"],
                        meta["test_labels_path"],
                        target_column=meta["target"],
                    )
                except Exception as exc:
                    result["holdout_metrics_error"] = f"{type(exc).__name__}: {exc}"
            else:
                result["holdout_metrics"] = None

            added: list[dict] = []
            if args.skill_mode == "on":
                added = library.add_from_verification(
                    result.get("verification_history", []),
                    source=name,
                )
            result["skills_added"] = [s["key"] for s in added]
            result["skill_library_size"] = len(library)

            run_path = paths["run_dir"] / f"result_round{round_index}.json"
            _save_json(run_path, result)

            metrics = result.get("holdout_metrics") or {}
            best_f1 = metrics.get("best_f1_on_holdout") or {}
            row = {
                "dataset": name,
                "round": round_index,
                "skill_mode": args.skill_mode,
                "verified": bool(result.get("verification_passed")),
                "verification_retries": result.get("verification_retries", 0),
                "pre_submit_failures": result.get("pre_submit_failures", 0),
                "phase": result.get("phase", ""),
                "auc": metrics.get("auc"),
                "best_f1": best_f1.get("f1"),
                "recall_at_best_f1": best_f1.get("recall"),
                "tokens": (result.get("token_usage") or {}).get("total_tokens"),
                "duration_sec": result.get("duration_sec"),
                "skills_injected": len(result.get("skills_injected", [])),
                "skills_added": len(result.get("skills_added", [])),
                "skill_library_size": result.get("skill_library_size"),
                "error": result.get("error", ""),
                "result_path": str(run_path),
            }
            rows.append(row)
            print(
                f"[RESULT] verified={row['verified']} retries={row['verification_retries']} "
                f"AUC={row['auc']} F1={row['best_f1']} tokens={row['tokens']} "
                f"duration={row['duration_sec']}s",
                flush=True,
            )

            # 连续两次完全失败时提前停止，避免浪费预算
            if not row["verified"] and not metrics:
                recent_failures = [r for r in rows[-2:] if not r["verified"]]
                if len(recent_failures) == 2:
                    print("[STOP] 连续两次任务失败且无预测结果，提前终止", flush=True)
                    stopped = True
                    break

    summary = pd.DataFrame(rows)
    summary_path = result_root / "summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    _save_json(result_root / "summary.json", rows)

    print("\n" + "=" * 78)
    print(f"[DONE] summary -> {summary_path}")
    if not summary.empty:
        print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill-library experiment runner")
    parser.add_argument("--datasets", type=str, default="telco,bank,adult")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--skill-mode", choices=["on", "off"], default="on")
    parser.add_argument("--skill-db", type=str, default="")
    parser.add_argument("--skill-top-k", type=int, default=5)
    parser.add_argument("--reset-skills", action="store_true")
    parser.add_argument("--results-dir", type=str, default="")
    parser.add_argument("--force-data", action="store_true", help="强制重建数据集切分")
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--provider", choices=["auto", "openai", "anthropic"], default="auto")
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
