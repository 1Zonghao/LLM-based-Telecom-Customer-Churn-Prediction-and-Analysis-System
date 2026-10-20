"""Skill library 单元测试。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.skills import SkillLibrary


def test_add_dedupe_and_persist(tmp_path):
    path = tmp_path / "skills.json"
    library = SkillLibrary(path)

    first = library.add(
        "write_deliverables",
        "在报告前写出交付物。",
        ["deliverables", "file"],
        source="telco",
    )
    second = library.add(
        "write_deliverables",
        "在报告前写出交付物。",
        ["deliverables", "completion"],
        source="bank",
    )

    assert first is second
    assert len(library) == 1
    assert library.skills[0]["seen_count"] == 2
    assert set(library.skills[0]["tags"]) == {"deliverables", "file", "completion"}
    assert set(library.skills[0]["sources"]) == {"telco", "bank"}

    library.save()
    reloaded = SkillLibrary(path)
    assert len(reloaded) == 1
    assert reloaded.skills[0]["key"] == "write_deliverables"


def test_add_from_verification_history(tmp_path):
    history = [
        {
            "passed": False,
            "checks": [
                {
                    "name": "file_exists:agent_predictions.csv",
                    "passed": False,
                    "detail": "文件不存在",
                },
                {
                    "name": "metric",
                    "passed": False,
                    "detail": "AUC 需要概率值，但预测列看起来是 0/1 硬标签",
                },
                {
                    "name": "columns:agent_predictions.csv",
                    "passed": True,
                    "detail": "列齐全",
                },
            ],
        }
    ]
    library = SkillLibrary(tmp_path / "skills.json")
    added = library.add_from_verification(history, source="telco")
    keys = {skill["key"] for skill in added}

    assert "write_deliverables" in keys
    assert "probability_not_hard_labels" in keys
    assert len(library) == 2

    # 同一失败再次出现时只增加 seen_count，不重复新增
    library.add_from_verification(history, source="bank")
    assert len(library) == 2
    by_key = {skill["key"]: skill for skill in library.skills}
    assert by_key["write_deliverables"]["seen_count"] == 2
    assert by_key["probability_not_hard_labels"]["seen_count"] == 2


def test_retrieve_ranks_relevant_skills(tmp_path):
    library = SkillLibrary(tmp_path / "skills.json")
    library.add(
        "probability_not_hard_labels",
        "AUC 任务必须输出概率。",
        ["prediction", "probability", "metric", "auc"],
    )
    library.add(
        "write_deliverables",
        "必须实际写出交付物。",
        ["deliverables", "file"],
    )
    library.add(
        "group_cv",
        "分组数据必须用 GroupKFold。",
        ["validation", "grouping", "leakage"],
    )

    skills = library.retrieve(
        {
            "task": "二分类流失预测，输出 ROC-AUC 概率",
            "tags": ["binary_classification", "imbalanced"],
        },
        k=2,
    )
    keys = [skill["key"] for skill in skills]

    assert "probability_not_hard_labels" in keys
    assert "write_deliverables" in keys
    text = library.format_for_prompt(skills)
    assert "1." in text
    assert "概率" in text or "交付物" in text


def test_empty_library_returns_none(tmp_path):
    library = SkillLibrary(tmp_path / "skills.json")
    assert library.retrieve("anything") == []
    assert library.format_for_prompt([]) == "暂无"
