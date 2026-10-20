"""Phase 1 冒烟测试：验证有状态代码执行沙箱的核心能力。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from sandbox.kernel_manager import KernelManager
from sandbox.compressor import OutputCompressor

DATASET = Path(__file__).resolve().parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


@pytest.fixture(scope="module")
def km():
    manager = KernelManager()
    manager.start()
    yield manager
    manager.shutdown()


@pytest.fixture(scope="module")
def comp():
    return OutputCompressor()


def test_basic_execution(km: KernelManager) -> None:
    print("=" * 60)
    print("Test 1: Basic code execution")
    result = km.execute("x = 42; print(f'x = {x}')")
    assert "x = 42" in result.text, f"Expected 'x = 42', got: {result.text}"
    assert not result.has_error
    print("  PASS")


def test_variable_persistence(km: KernelManager) -> None:
    print("=" * 60)
    print("Test 2: Variable persistence across cells")
    result = km.execute("print(f'x is still {x}')")
    assert "x is still 42" in result.text, f"Variable not persisted: {result.text}"
    assert not result.has_error
    print("  PASS")


def test_dataframe_loading(km: KernelManager) -> None:
    print("=" * 60)
    print("Test 3: DataFrame loading")
    code = f"""
import pandas as pd
df = pd.read_csv(r'{DATASET}')
print(df.shape)
print(df.dtypes.value_counts().to_string())
"""
    result = km.execute(code)
    assert "(7043, 21)" in result.text, f"Expected shape (7043, 21), got: {result.text}"
    assert not result.has_error
    print("  PASS")


def test_dataframe_reuse(km: KernelManager) -> None:
    print("=" * 60)
    print("Test 4: DataFrame reuse (no reload)")
    result = km.execute("print(f'Churn rate: {df.Churn.value_counts(normalize=True).to_dict()}')")
    assert "Churn rate" in result.text
    assert not result.has_error
    print("  PASS")


def test_matplotlib_chart(km: KernelManager) -> None:
    print("=" * 60)
    print("Test 5: Matplotlib chart capture")
    code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from IPython.display import display, Image as IPImage

fig, ax = plt.subplots()
df['Churn'].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'coral'])
ax.set_title('Churn Distribution')
plt.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format='png', bbox_inches='tight')
buf.seek(0)
display(IPImage(buf.read()))
plt.close(fig)
"""
    result = km.execute(code)
    assert len(result.images) > 0, "Expected at least 1 captured image"
    assert Path(result.images[0]).exists(), f"Image not saved: {result.images[0]}"
    print(f"  PASS (captured {len(result.images)} image(s))")


def test_error_handling(km: KernelManager) -> None:
    print("=" * 60)
    print("Test 6: Error handling")
    result = km.execute("print(undefined_var)  # NameError expected")
    assert result.has_error
    assert "NameError" in result.error
    print(f"  PASS (error captured: {result.error})")


def test_compressor(km: KernelManager, comp: OutputCompressor) -> None:
    print("=" * 60)
    print("Test 7: Output compression")
    result = km.execute("print(df.head(50).to_string())")
    compressed = comp.compress(result)
    assert len(compressed) < len(result.text), f"Compression should reduce size: {len(compressed)} >= {len(result.text)}"
    assert "省略" in compressed or "rows ×" in compressed or len(compressed) < 3000
    print(f"  PASS (original: {len(result.text)} chars, compressed: {len(compressed)} chars)")


def main() -> None:
    km = KernelManager()
    comp = OutputCompressor()

    try:
        print("\n>>> 启动 Jupyter Kernel ...")
        km.start()
        print(f"  Kernel alive: {km.is_alive}\n")

        test_basic_execution(km)
        test_variable_persistence(km)
        test_dataframe_loading(km)
        test_dataframe_reuse(km)
        test_matplotlib_chart(km)
        test_error_handling(km)
        test_compressor(km, comp)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60)

    finally:
        km.shutdown()
        print("\nKernel shut down.")


if __name__ == "__main__":
    main()
