#JoyAddded CTA币池因子趋势强度评估_20260329直播
import os
from config import SWAP_1M_PARQUET_PATH

CONFIG = {
    "picks_csv": "",
    "picks_path": os.path.join(os.path.dirname(__file__), ""),
    "picks_pattern": "final_select_results.csv",
    "kline_period": "1h",
    "bars": 72,
    "out_dir": os.path.join(os.path.dirname(__file__), "evl_output"),
    "kline_root": SWAP_1M_PARQUET_PATH,
    "segment_start_mode": "next-day",
    "amp_mode": "pct",
    "amp_beta": 1.0,
    "penalty_dd": 0.5,
    "penalty_zz": 0.5,
    "do_visual": True,
}
