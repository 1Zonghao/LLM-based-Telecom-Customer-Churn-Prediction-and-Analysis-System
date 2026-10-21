"""图级测试：验证节点、修复回路、迭代上限和断点恢复。"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import nodes
from agent.graph import build_agent, make_thread_config
from agent.llm_client import MockLLMClient
from agent.state import init_state
from agent.verifier import make_prediction_contract
from sandbox.kernel_manager import ExecutionResult


class FakeKernel:
    """看到 WRITE_ARTIFACTS 标记时写出预测文件和报告。"""

    def __init__(self, pred_path: Path, report_path: Path | None = None):
        self.pred_path = pred_path
        self.report_path = report_path

    def execute(self, code: str, timeout: int = 120) -> ExecutionResult:
        if "SELF-CHECK PASSED" in code:
            report_ok = self.report_path is None or self.report_path.exists()
            if self.pred_path.exists() and report_ok:
                return ExecutionResult(text="SELF-CHECK PASSED")
            return ExecutionResult(text="", error="AssertionError: 预测文件不存在")
        if "WRITE_ARTIFACTS" in code:
            pd.DataFrame({
                "id": [1, 2, 3],
                "Churn": [0.1, 0.8, 0.3],
            }).to_csv(self.pred_path, index=False)
            if self.report_path is not None:
                self.report_path.write_text("分析报告" * 100, encoding="utf-8")
        return ExecutionResult(text="执行成功")

    def shutdown(self) -> None:
        pass


def _drain(app, config, inputs) -> None:
    for _ in app.stream(inputs, config, stream_mode="updates"):
        pass


def test_verifier_forces_repair_then_passes(tmp_path):
    pred = tmp_path / "pred.csv"
    report = tmp_path / "report.md"
    contract = make_prediction_contract(
        tmp_path,
        pred,
        report,
        target_column="Churn",
        expected_rows=3,
        probability=True,
    )

    responses = [
        "```python\nprint('eda')\n```",
        "分析完成，交付物已经写好。",   # 没有文件 -> 验证失败
        "```python\nWRITE_ARTIFACTS\n```",
        "分析完成，交付物已经写好。",   # 文件存在 -> 验证通过
    ]
    llm = MockLLMClient(responses)
    app = build_agent(llm, FakeKernel(pred, report))
    config = make_thread_config()
    state = init_state(
        "测试任务",
        "dummy.csv",
        domain="generic",
        artifact_contract=contract,
    )

    _drain(app, config, state)
    final = app.get_state(config).values

    assert final["verification_passed"] is True, final.get("verification_result")
    assert final["phase"] == "completed"
    assert final["pre_submit_failures"] == 1
    assert final["verification_retries"] == 0
    assert len(final["verification_history"]) == 1
    assert len(final["pre_submit_history"]) == 2
    assert pred.exists() and report.exists()


def test_iteration_cap_ends_with_verification_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(nodes, "MAX_ITERATIONS", 1)

    pred = tmp_path / "never_written.csv"
    contract = make_prediction_contract(
        tmp_path,
        pred,
        target_column="Churn",
        max_verification_retries=0,
    )
    llm = MockLLMClient(["```python\nprint('x')\n```"] * 5)
    app = build_agent(llm, FakeKernel(pred))
    config = make_thread_config()
    state = init_state(
        "测试迭代上限",
        "dummy.csv",
        domain="generic",
        artifact_contract=contract,
    )

    _drain(app, config, state)
    final = app.get_state(config).values

    assert final["verification_passed"] is False
    assert final["phase"] == "verification_failed"
    assert not final.get("next_action") == "repair"
    assert not pred.exists()


def test_checkpoint_resume_uses_none(tmp_path):
    responses = [
        "[CHECKPOINT:FE]\n特征工程方案",
        "```python\nprint('feature engineering')\n```",
        "分析完成。",
    ]
    llm = MockLLMClient(responses)
    app = build_agent(llm, FakeKernel(tmp_path / "unused.csv"))
    config = make_thread_config()
    state = init_state("测试断点恢复", "dummy.csv", domain="generic")

    _drain(app, config, state)
    interrupted = app.get_state(config)
    assert "checkpoint_fe" in tuple(interrupted.next), interrupted.next

    # 静态 interrupt_before 必须用 None 恢复
    _drain(app, config, None)
    final = app.get_state(config)

    assert not tuple(final.next)
    assert final.values["verification_passed"] is True
    assert final.values["phase"] == "completed"
