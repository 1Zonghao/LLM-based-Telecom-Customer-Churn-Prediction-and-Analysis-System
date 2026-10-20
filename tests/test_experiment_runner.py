"""实验场数据准备与任务构造测试（不运行真实 agent）。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import experiment_runner as er


def test_map_target_variants():
    assert er._map_target(pd.Series(["Yes", "No", "Yes"]), "Yes").tolist() == [1, 0, 1]
    assert er._map_target(pd.Series([">50K", "<=50K"]), ">50K").tolist() == [1, 0]
    assert er._map_target(pd.Series([1, 0, 1]), 1).tolist() == [1, 0, 1]


def _register_toy_dataset(monkeypatch, source: Path) -> None:
    monkeypatch.setitem(er.DATASETS, "toy", {
        "source": source,
        "sep": ",",
        "target": "Churn",
        "id_column": "customerID",
        "positive": "Yes",
        "task": "玩具流失预测",
        "business": "测试用",
        "domain": "generic",
        "tags": ["binary_classification", "tabular"],
        "min_auc": 0.5,
        "held_out": False,
    })


def test_prepare_dataset_and_contract(tmp_path, monkeypatch):
    source = tmp_path / "toy.csv"
    rows = 40
    pd.DataFrame({
        "customerID": [f"id{i}" for i in range(rows)],
        "feature": range(rows),
        "Churn": ["Yes" if i % 3 == 0 else "No" for i in range(rows)],
    }).to_csv(source, index=False)
    _register_toy_dataset(monkeypatch, source)

    meta = er.prepare_dataset(
        "toy",
        output_root=tmp_path / "gym",
        holdout_size=0.25,
        seed=42,
    )

    assert meta["n_train"] == 30
    assert meta["n_test"] == 10
    assert meta["target"] == "Churn"
    assert meta["positive_rate"] > 0

    train = pd.read_csv(meta["train_path"])
    test_features = pd.read_csv(meta["test_features_path"])
    test_labels = pd.read_csv(meta["test_labels_path"])

    assert "id" in train.columns and "Churn" in train.columns
    assert "Churn" not in test_features.columns
    assert list(test_labels.columns) == ["id", "Churn"]
    assert set(test_labels["Churn"].unique()) <= {0, 1}
    assert len(test_labels) == 10

    # 第二次调用应直接复用缓存
    cached = er.prepare_dataset("toy", output_root=tmp_path / "gym")
    assert cached["n_train"] == meta["n_train"]
    assert cached["n_test"] == meta["n_test"]

    paths = er.build_paths("toy", tmp_path / "results")
    task = er.build_task(meta, paths)
    assert "Churn" in task
    assert str(paths["predictions"]) in task
    assert "SELF-CHECK PASSED" not in task

    task_with_skill = er.build_task(
        meta, paths, skill_keys={"write_deliverables"}
    )
    assert "SELF-CHECK PASSED" in task_with_skill
    assert str(paths["predictions"]) in task_with_skill

    contract = er.build_contract(meta, paths)
    assert contract["target_column"] == "Churn"
    assert contract["required_files"][0]["expected_rows"] == 10
    assert contract["required_files"][0]["probability"] is True
    assert contract["labels_path"] == meta["test_labels_path"]
    assert contract["metric"] == "roc_auc"
