#JoyTestStart
# JoyChange 20260405 原因：合并 batch pkl 并统一 candle_begin_time 为 UTC 无时区，避免 tz-aware 与 tz-naive 时间比较报错
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

import pandas as pd


def _to_utc_naive_datetime(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, utc=True, errors='coerce')
    return dt.dt.tz_localize(None)


def _normalize_time_cols(df: pd.DataFrame) -> None:
    for col in ("candle_begin_time", "first_candle_time", "last_candle_time"):
        if col in df.columns:
            df[col] = _to_utc_naive_datetime(df[col])


def _iter_files(root: Path, pattern: str) -> list[Path]:
    return sorted([p for p in root.glob(pattern) if p.is_file()])


def _load_dict(fp: Path) -> Dict[str, Any]:
    obj = pd.read_pickle(fp)
    if not isinstance(obj, dict):
        raise TypeError(f"{fp} 不是 dict，实际类型: {type(obj)}")

    for _, v in obj.items():
        if isinstance(v, pd.DataFrame) and not v.empty:
            _normalize_time_cols(v)

    return obj


def merge_batches(
    src_root: Path,
    pattern: str,
    out_path: Path,
    *,
    allow_key_overlap: bool = False,
) -> Tuple[Path, int]:
    files = _iter_files(src_root, pattern)
    if not files:
        raise FileNotFoundError(f"未匹配到文件: root={src_root} pattern={pattern}")

    merged: Dict[str, Any] = {}
    for fp in files:
        d = _load_dict(fp)
        if not allow_key_overlap:
            overlap = set(merged).intersection(d)
            if overlap:
                sample = list(sorted(overlap))[:10]
                raise RuntimeError(f"发现重复 key（示例前10个）: {sample}；出现在文件: {fp}")
        merged.update(d)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(merged, out_path)
    return out_path, len(merged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src-root",
        default="/home/ubuntu/datacenter/data/preprocess_1h_resample/0m",
        help="batch pkl 所在目录",
    )
    ap.add_argument(
        "--out-root",
        default="/home/ubuntu/datacenter/data/preprocess_1h_resample/0m/joymerged",
        help="输出目录（生成 swap_dict.pkl / spot_dict.pkl）",
    )
    ap.add_argument(
        "--swap-pattern",
        default="swap_dict_batch*.pkl",
        help="合约 batch 文件匹配模式",
    )
    ap.add_argument(
        "--spot-pattern",
        default="spot_dict_batch*.pkl",
        help="现货 batch 文件匹配模式",
    )
    ap.add_argument(
        "--allow-key-overlap",
        action="store_true",
        help="允许不同 batch 之间出现相同 key（后者覆盖前者）；默认不允许",
    )
    args = ap.parse_args()

    src_root = Path(args.src_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"src-root 不存在: {src_root}")

    swap_out, swap_keys = merge_batches(
        src_root, args.swap_pattern, out_root / "swap_dict.pkl", allow_key_overlap=args.allow_key_overlap
    )
    spot_out, spot_keys = merge_batches(
        src_root, args.spot_pattern, out_root / "spot_dict.pkl", allow_key_overlap=args.allow_key_overlap
    )

    print(f"OK swap: {swap_out} keys={swap_keys}")
    print(f"OK spot: {spot_out} keys={spot_keys}")


if __name__ == "__main__":
    main()
#JoyTestEnd