"""Skill library：把已验证任务中的失败教训沉淀成可复用经验。

v0 采用确定性规则（不额外调用 LLM）：
- 从 verifier 的失败项映射到一条通用教训；
- 按 key 去重，记录出现次数；
- 下一次任务开始前按标签/关键词检索 top-k，注入 TASK_PROMPT。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_SKILL_PATH = Path("outputs/skills/skills.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 失败检查 -> 可复用教训。顺序很重要：更具体的规则放前面。
_LESSON_RULES: list[dict[str, Any]] = [
    {
        "key": "probability_not_hard_labels",
        "match": lambda name, detail: (
            name == "metric" and ("AUC" in detail and ("硬标签" in detail or "概率" in detail))
        ),
        "lesson": "AUC/概率类任务必须输出连续概率（0~1），不能输出 0/1 硬标签。",
        "tags": ["prediction", "probability", "metric", "auc"],
    },
    {
        "key": "write_deliverables",
        "match": lambda name, detail: name.startswith("file_exists:"),
        "lesson": "在输出最终报告之前，必须用代码实际写出任务要求的全部交付物文件，并打印确认。",
        "tags": ["deliverables", "file", "completion"],
    },
    {
        "key": "required_columns",
        "match": lambda name, detail: name.startswith("columns:"),
        "lesson": "交付的 CSV 必须使用契约要求的精确列名，不要改列名或额外加索引列。",
        "tags": ["deliverables", "schema", "columns"],
    },
    {
        "key": "match_row_count",
        "match": lambda name, detail: name.startswith("row_count:"),
        "lesson": "预测文件的行数必须与待预测集完全一致，并保持相同顺序。",
        "tags": ["deliverables", "row_count", "prediction"],
    },
    {
        "key": "unique_ids",
        "match": lambda name, detail: name.startswith("unique_id:"),
        "lesson": "预测文件的 id 必须唯一，合并前要检查重复 id。",
        "tags": ["deliverables", "id", "data_quality"],
    },
    {
        "key": "no_missing_predictions",
        "match": lambda name, detail: name.startswith("no_missing_predictions:"),
        "lesson": "预测列不能有缺失值；写出文件前先用 isnull().sum() 检查。",
        "tags": ["deliverables", "missing_values", "data_quality"],
    },
    {
        "key": "probability_range",
        "match": lambda name, detail: name.startswith("probability_range:"),
        "lesson": "概率预测必须落在 [0,1] 区间；越界通常意味着口径错误或未做 sigmoid。",
        "tags": ["prediction", "probability", "data_quality"],
    },
    {
        "key": "report_file",
        "match": lambda name, detail: name.startswith("report_content:"),
        "lesson": "最终报告不仅要放在对话里，还要写入契约要求的报告文件，并达到最小长度。",
        "tags": ["deliverables", "report"],
    },
    {
        "key": "final_report",
        "match": lambda name, detail: name == "final_report",
        "lesson": "流程结束前必须输出一段完整的最终报告，不能只留下代码执行记录。",
        "tags": ["deliverables", "report", "completion"],
    },
    {
        "key": "group_cv",
        "match": lambda name, detail: name == "group_validation",
        "lesson": "数据存在分组列时，必须使用 GroupKFold/StratifiedGroupKFold/GroupShuffleSplit；随机 K-fold 会把同组样本拆到训练和验证两边，造成泄漏。",
        "tags": ["validation", "grouping", "leakage"],
    },
    {
        "key": "forbidden_features",
        "match": lambda name, detail: name == "forbidden_features",
        "lesson": "不要使用业务上不可用或会造成泄漏的特征（例如通话结束后的 duration）；即使指标更好也必须丢弃。",
        "tags": ["leakage", "features", "validation"],
    },
    {
        "key": "valid_csv",
        "match": lambda name, detail: name.startswith("csv_parse:"),
        "lesson": "交付 CSV 前先自己 read_csv 一遍，确认文件可解析、编码正确。",
        "tags": ["deliverables", "csv", "data_quality"],
    },
    {
        "key": "metric_reproducible",
        "match": lambda name, detail: (
            name == "metric"
            and "FileNotFoundError" not in detail
            and "No such file" not in detail
        ),
        "lesson": "报告里的指标必须能用交付文件复现；预测格式要和指标要求匹配。",
        "tags": ["metric", "validation", "reproducibility"],
    },
]


def _match_rule(check: dict) -> dict[str, Any] | None:
    name = str(check.get("name", ""))
    detail = str(check.get("detail", ""))
    for rule in _LESSON_RULES:
        try:
            if rule["match"](name, detail):
                return rule
        except Exception:
            continue
    return None


class SkillLibrary:
    """JSON 文件支持的轻量技能库。"""

    def __init__(self, path: str | Path = DEFAULT_SKILL_PATH):
        self.path = Path(path)
        self.skills: list[dict] = []
        self.load()

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            self.skills = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.skills = data if isinstance(data, list) else []
        except Exception:
            self.skills = []

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.skills, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add(
        self,
        key: str,
        lesson: str,
        tags: Iterable[str],
        source: str = "",
    ) -> dict:
        tags = sorted(set(str(t) for t in tags if t))
        existing = next((s for s in self.skills if s.get("key") == key), None)
        if existing is not None:
            existing["seen_count"] = int(existing.get("seen_count", 1)) + 1
            existing["last_seen"] = _now()
            existing["tags"] = sorted(set(existing.get("tags", [])) | set(tags))
            if source:
                existing.setdefault("sources", [])
                if source not in existing["sources"]:
                    existing["sources"].append(source)
            return existing

        skill = {
            "key": key,
            "lesson": lesson,
            "tags": tags,
            "seen_count": 1,
            "created_at": _now(),
            "last_seen": _now(),
            "sources": [source] if source else [],
        }
        self.skills.append(skill)
        return skill

    def add_from_verification(
        self,
        verification_history: Iterable[dict],
        source: str = "",
    ) -> list[dict]:
        """从验证历史中的失败项提炼技能，返回本次新增/更新的技能。"""
        touched: list[dict] = []
        seen_keys: set[str] = set()

        for result in verification_history or []:
            for check in result.get("checks", []):
                if check.get("passed"):
                    continue
                rule = _match_rule(check)
                if rule is None:
                    continue
                # 同一次验证中，同一类问题只沉淀一条经验
                if rule["key"] in seen_keys:
                    continue
                seen_keys.add(rule["key"])
                touched.append(
                    self.add(
                        key=rule["key"],
                        lesson=rule["lesson"],
                        tags=rule["tags"],
                        source=source,
                    )
                )

        if touched:
            self.save()
        return touched

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def retrieve(self, query: str | dict, k: int = 5) -> list[dict]:
        if not self.skills:
            return []

        if isinstance(query, dict):
            text = " ".join(
                str(query.get(key, ""))
                for key in ("task", "domain", "dataset", "target_type")
            )
            query_tags = set(query.get("tags", []) or [])
        else:
            text = str(query)
            query_tags = set()

        text_lower = text.lower()
        scored: list[tuple[float, dict]] = []
        for skill in self.skills:
            tags = set(skill.get("tags", []))
            tag_score = len(query_tags & tags) * 2.0
            keyword_score = sum(
                1.0 for tag in tags if tag.lower() in text_lower
            )
            # 通用交付类经验始终保留一点权重
            universal = 1.0 if "deliverables" in tags else 0.0
            seen_bonus = min(int(skill.get("seen_count", 1)), 5) * 0.2
            score = tag_score + keyword_score + universal + seen_bonus
            scored.append((score, skill))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [skill for _, skill in scored[:k]]

    def format_for_prompt(self, skills: list[dict]) -> str:
        if not skills:
            return "暂无"
        lines = []
        for index, skill in enumerate(skills, 1):
            tags = ", ".join(skill.get("tags", []))
            lines.append(f"{index}. [{tags}] {skill.get('lesson', '')}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.skills = []
        self.save()

    def __len__(self) -> int:
        return len(self.skills)
