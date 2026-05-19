"""Phase 2 集成测试：验证 LangGraph 智能体的图结构与执行流程。"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sandbox.kernel_manager import KernelManager
from sandbox.compressor import OutputCompressor
from agent.llm_client import LLMClient, MockLLMClient
from agent.graph import build_agent, run_agent, init_state

DATASET = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def test_graph_structure_with_mock() -> None:
    """使用 MockLLMClient 测试图结构（无需 API key）。"""
    print("=" * 60)
    print("Test: Graph structure with MockLLMClient")

    # Use forward slashes to avoid escape issues in Python strings
    data_path_posix = str(DATASET).replace("\\", "/")

    # Step 1: code → Step 2: finish
    mock = MockLLMClient(responses=[
        "我来加载数据并查看基本信息。\n\n```python\nimport pandas as pd\ndf = pd.read_csv('{}')\nprint(df.shape)\nprint(df.dtypes)\n```".format(data_path_posix),
        "数据加载完成。数据集包含 7043 行，21 列。分析完毕。",
    ])

    km = KernelManager()
    km.start()
    compressor = OutputCompressor()

    try:
        from agent.graph import make_thread_config
        app = build_agent(mock, km, compressor)
        state = init_state("加载数据集并展示基本信息", str(DATASET))
        final = app.invoke(state, make_thread_config())

        assert final["iteration"] == 1, f"Expected 1 iteration, got {final['iteration']}"
        assert final["next_action"] == "finish"
        assert len(final["code_history"]) == 1, "Should have 1 code block"
        assert "import pandas" in final["code_history"][0]
        assert "数据加载完成" in final["final_report"]
        assert "n_rows" in final["data_profile"], f"Should have n_rows in profile: {final['data_profile']}"
        assert final["data_profile"]["n_rows"] == 7043

        print("  PASS: Graph structure works correctly")
        print(f"    - Iterations: {final['iteration']}")
        print(f"    - Code blocks: {len(final['code_history'])}")
        print(f"    - Profile: {final['data_profile']}")

    finally:
        km.shutdown()


def test_router_directly() -> None:
    """直接测试路由逻辑。"""
    print("=" * 60)
    print("Test: Router logic")

    from agent.nodes import router

    assert router({"next_action": "execute"}) == "execute"
    assert router({"next_action": "finish"}) == "finish"
    assert router({}) == "finish"  # default

    print("  PASS: Router works correctly")


def test_code_block_parsing() -> None:
    """测试代码块解析。"""
    print("=" * 60)
    print("Test: Code block parsing")

    import re
    from agent.nodes import CODE_BLOCK_RE

    # Standard code block
    text1 = "Let me check the data.\n\n```python\nimport pandas as pd\nprint(df.head())\n```"
    m1 = CODE_BLOCK_RE.search(text1)
    assert m1 is not None
    assert "import pandas as pd" in m1.group(1)

    # No code block (final report)
    text2 = "数据分析完成。主要发现：1) 数据集有7043行 2) 流失率约26.5%"
    m2 = CODE_BLOCK_RE.search(text2)
    assert m2 is None

    # Multiple code blocks — should only match first
    text3 = "```python\nprint('first')\n```\nSome text\n```python\nprint('second')\n```"
    m3 = CODE_BLOCK_RE.search(text3)
    assert m3 is not None
    assert "first" in m3.group(1)

    print("  PASS: Code block parsing works correctly")


def test_e2e_with_api() -> None:
    """端到端测试（需要 LLM_API_KEY 环境变量）。"""
    api_key = os.getenv("LLM_API_KEY", "")
    if not api_key:
        print("=" * 60)
        print("Test: End-to-end (SKIPPED — LLM_API_KEY not set)")
        print("  Set LLM_API_KEY env var and re-run for live test.")
        return

    print("=" * 60)
    print("Test: End-to-end with real LLM")

    llm = LLMClient()
    km = KernelManager()
    km.start()

    try:
        final = run_agent(
            task="加载 Telco Customer Churn 数据集，展示基本信息（shape、列类型、缺失值、目标变量分布），然后输出简短的数据概览报告。",
            data_path=str(DATASET),
            llm=llm,
            km=km,
        )

        print(f"  Iterations: {final['iteration']}")
        print(f"  Code blocks: {len(final['code_history'])}")
        print(f"  Profile: {final['data_profile']}")
        print(f"  Report:\n{final['final_report'][:500]}...")
        print("  PASS")

    finally:
        km.shutdown()


def main() -> None:
    print("\n" + "=" * 60)
    print("Phase 2 — LangGraph Agent Integration Tests")
    print("=" * 60)

    test_router_directly()
    test_code_block_parsing()
    test_graph_structure_with_mock()
    test_e2e_with_api()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
