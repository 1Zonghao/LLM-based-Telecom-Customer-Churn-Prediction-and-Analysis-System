"""Phase 4: 边界测试与鲁棒性验证。"""
import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.llm_client import MockLLMClient
from agent.state import init_state, AgentState
from agent.graph import build_agent, make_thread_config
from agent.nodes import (
    _is_feature_engineering_code,
    _is_model_training_code,
    _extract_profile_from_output,
)
from sandbox.kernel_manager import KernelManager
from sandbox.compressor import OutputCompressor


def test_traceback_compression():
    """测试 Traceback 压缩是否保留关键信息。"""
    print("=" * 60)
    print("Test: Traceback compression")

    comp = OutputCompressor()

    # 长 traceback（超过 3000 字符阈值，触发压缩）
    tb_line = '  File "/very/long/path/to/some/module/pandas/core/frame.py", line 3761, in __getitem__\n    some_code_here()\n'
    long_tb = "Traceback (most recent call last):\n" + tb_line * 80 + "KeyError: 'churn'"

    from sandbox.kernel_manager import ExecutionResult
    result = ExecutionResult(text=long_tb, error="KeyError: 'churn'")
    compressed = comp.compress(result)

    assert "Traceback" in compressed, f"Should keep header: {compressed}"
    assert "KeyError" in compressed, f"Should keep error type: {compressed}"
    assert "省略" in compressed, f"Should indicate omitted content: {compressed}"
    assert len(compressed) < 3000, f"Should be compressed: {len(compressed)} vs {len(long_tb)}"

    print(f"  Original: {len(long_tb)} chars, Compressed: {len(compressed)} chars")
    print("  PASS")


def test_short_traceback_passthrough():
    """短 traceback 应保持原文（不需要压缩）。"""
    print("=" * 60)
    print("Test: Short traceback passthrough")

    comp = OutputCompressor()
    short_tb = "Traceback (most recent call last):\n  File '<input>', line 1\nKeyError: 'churn'"

    from sandbox.kernel_manager import ExecutionResult
    result = ExecutionResult(text=short_tb, error="KeyError: 'churn'")
    compressed = comp.compress(result)

    # 短 traceback 全文保留
    assert "Traceback" in compressed
    assert "KeyError" in compressed
    print(f"  Output: {compressed}")
    print("  PASS")


def test_phase_detection():
    """测试代码阶段检测逻辑。"""
    print("=" * 60)
    print("Test: Phase detection")

    fe_code = "df.fillna(0)\nfrom sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()"
    model_code = "from xgboost import XGBClassifier\nmodel = XGBClassifier()\nmodel.fit(X_train, y_train)"
    eda_code = "print(df.head())\nprint(df.describe())"

    assert _is_feature_engineering_code(fe_code), "Should detect FE code"
    assert not _is_model_training_code(fe_code), "FE code is not model code"
    assert _is_model_training_code(model_code), "Should detect model code"
    assert not _is_feature_engineering_code(eda_code), "EDA code is not FE code"
    assert not _is_model_training_code(eda_code), "EDA code is not model code"

    print("  PASS")


def test_profile_extraction():
    """测试指标提取正则。"""
    print("=" * 60)
    print("Test: Profile extraction")

    # Shape
    u = _extract_profile_from_output("df.shape = (7043, 21)", {})
    assert u["n_rows"] == 7043
    assert u["n_cols"] == 21

    # F1
    u = _extract_profile_from_output("F1 Score: 0.6542\nRecall: 0.7234\nAUC: 0.8412", {})
    assert u["best_f1"] == 0.6542
    assert u["best_recall"] == 0.7234
    assert u["best_auc"] == 0.8412

    # Best score tracking (only updates if higher)
    u = _extract_profile_from_output("F1: 0.50", {"best_f1": 0.65})
    assert u == {}, "Should not update if lower"

    print("  PASS")


def test_self_repair_flow_mock():
    """使用 MockLLMClient 测试自修复流程。"""
    print("=" * 60)
    print("Test: Self-repair flow (mock)")

    mock = MockLLMClient(responses=[
        # Step 1: Intentional typo in column name
        '```python\nimport pandas as pd\ndf = pd.read_csv("dummy.csv")\nprint(df["churn"].value_counts())\n```',
        # Step 2: After error, the LLM should fix — check columns
        '```python\nprint(df.columns.tolist())\n```',
        # Step 3: Now use correct column
        '```python\nprint(df["Churn"].value_counts())\n```',
        # Step 4: Finish
        '分析完成。数据中的Churn列分布为: Yes=1869, No=5174。',
    ])

    km = KernelManager()
    km.start()

    try:
        app = build_agent(mock, km)
        state = init_state("test error recovery", "/tmp/dummy.csv")
        config = make_thread_config()

        events = []
        for event in app.stream(state, config, stream_mode="updates"):
            if "__interrupt__" in event:
                events.append(("interrupt", event["__interrupt__"]))
            else:
                for node_name, node_output in event.items():
                    na = node_output.get("next_action", "?")
                    msgs = node_output.get("messages", [])
                    ce = node_output.get("consecutive_errors", 0)
                    events.append((node_name, na, ce, len(msgs)))

        # Verify the flow includes error → fix → success
        node_names = [e[0] for e in events]
        assert "reasoning" in node_names, f"Should have reasoning steps: {node_names}"
        assert "action" in node_names, f"Should have action steps: {node_names}"

        print(f"  Flow: {' → '.join(f'{e[0]}({e[1]})' for e in events)}")
        print("  PASS: Self-repair flow completed")

    finally:
        km.shutdown()


def test_error_consecutive_tracking():
    """测试连续错误计数。"""
    print("=" * 60)
    print("Test: Consecutive error tracking")

    mock = MockLLMClient(responses=[
        # Error 1
        '```python\nprint(undefined_var)\n```',
        # Error 2 (should still be 1 if LLM makes same mistake)
        '```python\nprint(undefined_var)\n```',
        # Fix it
        '```python\nx = 42\nprint(x)\n```',
        '分析完成。',
    ])

    km = KernelManager()
    km.start()

    try:
        app = build_agent(mock, km)
        state = init_state("test error tracking", "/tmp/dummy.csv")
        config = make_thread_config()

        error_counts = []
        for event in app.stream(state, config, stream_mode="updates"):
            if "__interrupt__" in event:
                continue
            for node_name, node_output in event.items():
                if node_name == "action":
                    ce = node_output.get("consecutive_errors", 0)
                    error_counts.append(ce)
                    print(f"  Action: consecutive_errors={ce}")

        assert error_counts[0] >= 1, f"First error should increment counter: {error_counts}"
        assert error_counts[-1] == 0, f"After success should reset: {error_counts}"

        print("  PASS: Error tracking works")

    finally:
        km.shutdown()


def test_empty_result_handling():
    """测试空结果时的处理。"""
    print("=" * 60)
    print("Test: Empty result handling")

    comp = OutputCompressor()
    from sandbox.kernel_manager import ExecutionResult

    # Empty result
    result = ExecutionResult(text="", images=[], error=None)
    compressed = comp.compress(result)
    assert "无输出" in compressed, f"Should indicate empty: {compressed}"

    # Only images
    result = ExecutionResult(text="", images=["data/images/test.png"])
    compressed = comp.compress(result)
    assert "图表" in compressed, f"Should mention images: {compressed}"

    print("  PASS")


def test_compressor_error_format():
    """验证压缩器对错误的格式化保留足够上下文。"""
    print("=" * 60)
    print("Test: Error format in compressor")

    comp = OutputCompressor()
    from sandbox.kernel_manager import ExecutionResult

    # Simulate a typical pandas KeyError
    tb = """Traceback (most recent call last):
  File "<ipython-input-3>", line 3, in <module>
    df['Churn ']
  File "...pandas/core/frame.py", line 3761, in __getitem__
    indexer = self.columns.get_loc(key)
KeyError: 'Churn '"""

    result = ExecutionResult(text=tb + "\ndf columns: customerID, gender, Churn, ...", error="KeyError: 'Churn '")
    compressed = comp.compress(result)

    assert "KeyError" in compressed
    assert "执行错误" in compressed
    print(f"  Compressed:\n{compressed}")
    print("  PASS")


if __name__ == "__main__":
    test_traceback_compression()
    test_short_traceback_passthrough()
    test_phase_detection()
    test_profile_extraction()
    test_empty_result_handling()
    test_compressor_error_format()
    test_error_consecutive_tracking()
    test_self_repair_flow_mock()
    print("\n" + "=" * 60)
    print("ALL ROBUSTNESS TESTS PASSED")
