"""批量测试脚本 — 在多个 UCI 数据集上运行自主数据科学智能体。

用法:
    python src/batch_test.py --data-dir data/uci_datasets --output-dir outputs/batch_results

说明:
    - 自动遍历目录下所有 CSV 文件
    - 以最后一列作为目标变量，自动生成任务指令
    - 自动通过所有 checkpoint（不注入人工反馈）
    - 记录执行时间、迭代轮次、最佳指标、最终报告等
"""
import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.llm_client import LLMClient
from agent.state import init_state
from agent.graph import build_agent, make_thread_config
from sandbox.kernel_manager import KernelManager
from sandbox.compressor import OutputCompressor


def _build_task_prompt(csv_path: Path, target_col: str) -> str:
    """根据数据集路径和目标列生成任务指令。"""
    # 简单启发式：根据目标列名或文件名推断任务类型
    name = csv_path.stem.lower()
    if "income" in name or "adult" in name:
        return f"分析 {csv_path.name} 数据集，以 {target_col} 为目标变量，构建收入预测模型。由于高收入人群样本较少，关注召回率优化，确保尽可能多的识别出高收入人群。"
    elif "bank" in name or "marketing" in name:
        return f"分析 {csv_path.name} 数据集，以 {target_col} 为目标变量，构建营销响应预测模型。由于响应用户比例很低，关注召回率优化，尽可能找出所有潜在响应客户。"
    elif "diabetes" in name:
        return f"分析 {csv_path.name} 数据集，以 {target_col} 为目标变量，构建糖尿病预测模型。由于漏诊代价远高于误诊，关注召回率优化，尽可能识别出所有糖尿病患者。"
    else:
        return f"分析 {csv_path.name} 数据集，以 {target_col} 为目标变量，构建预测模型，关注召回率优化。"


def run_single(
    csv_path: Path,
    llm: LLMClient,
    target_col: str | None = None,
    max_total_iterations: int = 25,
) -> dict:
    """在单个数据集上运行智能体，自动处理断点。

    Returns:
        result dict with keys: success, duration_sec, iterations, phase,
        data_profile, final_report, error, code_history_length
    """
    result = {
        "dataset": csv_path.name,
        "success": False,
        "duration_sec": 0.0,
        "iterations": 0,
        "phase": "unknown",
        "data_profile": {},
        "final_report": "",
        "error": "",
        "code_history_length": 0,
    }

    # Infer target column if not provided
    if target_col is None:
        import pandas as pd
        df = pd.read_csv(csv_path)
        target_col = df.columns[-1]

    task = _build_task_prompt(csv_path, target_col)
    print(f"\n{'='*60}")
    print(f"[START] {csv_path.name} -> target={target_col}")
    print(f"Task: {task}")
    print(f"{'='*60}")

    km = KernelManager()
    try:
        km.start()
    except Exception as e:
        result["error"] = f"Kernel start failed: {e}"
        print(f"[ERROR] {csv_path.name}: {result['error']}")
        return result

    compressor = OutputCompressor()
    app = build_agent(llm, km, compressor)
    config = make_thread_config()
    state = init_state(task, str(csv_path), domain="generic")

    start_time = time.time()
    total_iterations = 0
    total_stream_count = 0

    def _accumulate(node_output: dict) -> None:
        nonlocal total_iterations, total_stream_count
        for key, value in node_output.items():
            if key in ("messages", "code_history"):
                state.setdefault(key, [])
                if isinstance(value, list):
                    state[key].extend(value)
                else:
                    state[key].append(value)
            else:
                state[key] = value
            if key == "iteration":
                total_iterations = max(total_iterations, int(value))
        total_stream_count += 1

    try:
        # Main loop: first run with the initial state, then resume with None.
        inputs = state
        while total_iterations < max_total_iterations:
            for event in app.stream(inputs, config, stream_mode="updates"):
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        continue
                    _accumulate(node_output)

            # After the stream ends, check whether we stopped at a static checkpoint.
            try:
                snapshot = app.get_state(config)
            except Exception:
                break

            next_nodes = tuple(getattr(snapshot, "next", ()) or ())
            if next_nodes and any(str(n).startswith("checkpoint") for n in next_nodes):
                print(f"  [CHECKPOINT] {next_nodes} — auto-confirming...")
                try:
                    app.update_state(config, {"user_feedback": ""})
                except Exception:
                    pass
                # Static interrupts resume by passing None (Command(resume=...) is a no-op here)
                inputs = None
                continue
            break

        duration = time.time() - start_time
        result["duration_sec"] = round(duration, 2)
        result["iterations"] = total_iterations
        result["phase"] = state.get("phase", "unknown")
        result["data_profile"] = state.get("data_profile", {})
        result["final_report"] = state.get("final_report", "")[:2000]  # truncate
        result["code_history_length"] = len(state.get("code_history", []))
        result["success"] = True

        print(f"  [DONE] {csv_path.name} | iter={total_iterations} | time={duration:.1f}s | profile={result['data_profile']}")

    except Exception as e:
        duration = time.time() - start_time
        result["duration_sec"] = round(duration, 2)
        result["error"] = f"{type(e).__name__}: {str(e)}"
        print(f"  [ERROR] {csv_path.name}: {result['error']}")
        traceback.print_exc()

    finally:
        try:
            km.shutdown()
        except Exception:
            pass

    return result


def main():
    parser = argparse.ArgumentParser(description="Batch test autonomous data science agent")
    parser.add_argument("--data-dir", type=str, default="data/uci_datasets",
                        help="Directory containing CSV datasets")
    parser.add_argument("--output-dir", type=str, default="outputs/batch_results",
                        help="Directory to save results")
    parser.add_argument("--max-iter", type=int, default=25,
                        help="Max total iterations per dataset")
    parser.add_argument("--datasets", type=str, default="",
                        help="Comma-separated dataset names to test (default: all CSVs)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover CSVs
    if args.datasets:
        names = [n.strip() for n in args.datasets.split(",")]
        csv_files = [data_dir / f"{n}.csv" for n in names]
        csv_files = [f for f in csv_files if f.exists()]
    else:
        csv_files = sorted(data_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"Found {len(csv_files)} datasets: {[f.name for f in csv_files]}")

    llm = LLMClient()
    if not llm.is_configured:
        print("[FATAL] LLM_API_KEY not set. Export it as environment variable before running.")
        return

    results = []
    for csv_path in csv_files:
        res = run_single(csv_path, llm, max_total_iterations=args.max_iter)
        results.append(res)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"batch_result_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Summary CSV
    summary_rows = []
    for r in results:
        summary_rows.append({
            "dataset": r["dataset"],
            "success": r["success"],
            "duration_sec": r["duration_sec"],
            "iterations": r["iterations"],
            "phase": r["phase"],
            "best_recall": r["data_profile"].get("best_recall", None),
            "best_f1": r["data_profile"].get("best_f1", None),
            "best_auc": r["data_profile"].get("best_auc", None),
            "code_history_length": r["code_history_length"],
            "error": r["error"],
        })

    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / f"batch_summary_{timestamp}.csv"
    summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print("BATCH TEST COMPLETE")
    print(f"Results JSON: {json_path}")
    print(f"Summary CSV:  {csv_path}")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
