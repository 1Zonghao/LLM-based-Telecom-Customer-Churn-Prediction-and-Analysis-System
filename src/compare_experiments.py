"""对比 skill-mode on/off 的实验结果。

用法:
    python src/compare_experiments.py --on-dir outputs/experiments/xxx_on --off-dir outputs/experiments/xxx_off
    python src/compare_experiments.py            # 自动找最近一组 on/off
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = PROJECT_ROOT / "outputs" / "experiments"


def _latest_dirs() -> tuple[Path, Path]:
    on_dirs = sorted(EXP_DIR.glob("*_on"), key=lambda p: p.stat().st_mtime)
    off_dirs = sorted(EXP_DIR.glob("*_off"), key=lambda p: p.stat().st_mtime)
    if not on_dirs or not off_dirs:
        raise SystemExit("找不到 on/off 结果目录，请先运行实验")
    return on_dirs[-1], off_dirs[-1]


def _load(directory: Path, label: str) -> pd.DataFrame:
    path = directory / "summary.csv"
    if not path.exists():
        raise SystemExit(f"缺少 {path}")
    df = pd.read_csv(path)
    df["mode"] = label
    return df


def _aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = df.groupby(keys, dropna=False)
    return grouped.agg(
        n=("dataset", "size"),
        verified_rate=("verified", "mean"),
        mean_retries=("verification_retries", "mean"),
        mean_pre_submit_failures=("pre_submit_failures", "mean"),
        mean_tokens=("tokens", "mean"),
        mean_duration_sec=("duration_sec", "mean"),
        mean_auc=("auc", "mean"),
        mean_f1=("best_f1", "mean"),
        skill_reuse_rate=("skills_injected", lambda s: float((s > 0).mean())),
    ).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare skill on/off experiments")
    parser.add_argument("--on-dir", type=str, default="")
    parser.add_argument("--off-dir", type=str, default="")
    args = parser.parse_args()

    if args.on_dir and args.off_dir:
        on_dir, off_dir = Path(args.on_dir), Path(args.off_dir)
    else:
        on_dir, off_dir = _latest_dirs()

    df = pd.concat([_load(on_dir, "on"), _load(off_dir, "off")], ignore_index=True)
    df["verified"] = df["verified"].astype(str).str.lower().isin(["true", "1"])

    by_round = _aggregate(df, ["mode", "round"])
    by_dataset = _aggregate(df, ["mode", "dataset"])
    by_round_dataset = _aggregate(df, ["mode", "round", "dataset"])

    print("=" * 90)
    print(f"ON : {on_dir}")
    print(f"OFF: {off_dir}")
    print("=" * 90)
    print("\n--- 按轮次 ---")
    print(by_round.round(4).to_string(index=False))
    print("\n--- 按数据集 ---")
    print(by_dataset.round(4).to_string(index=False))
    print("\n--- 按轮次 × 数据集 ---")
    print(by_round_dataset.round(4).to_string(index=False))

    payload = {
        "on_dir": str(on_dir),
        "off_dir": str(off_dir),
        "by_round": by_round.to_dict(orient="records"),
        "by_dataset": by_dataset.to_dict(orient="records"),
        "by_round_dataset": by_round_dataset.to_dict(orient="records"),
        "raw": df.to_dict(orient="records"),
    }
    out_path = on_dir.parent / "comparison.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\ncomparison -> {out_path}")


if __name__ == "__main__":
    main()
