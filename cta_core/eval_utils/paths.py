#JoyAddded CTA币池因子趋势强度评估_20260329直播
import os
import re

def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""

def read_project_paths(md_path: str):
    text = _read_text(md_path)
    lines = text.splitlines() if text else []
    patterns = {
        "open_interest": r"持仓量数据目录[^:：]*[:：]\s*(.+)",
        "market_cap": r"市值数据目录[^:：]*[:：]\s*(.+)",
        "kline_parquet": r"行情K线数据目录[^:：]*[:：]\s*(.+)",
    }
    result = {}
    for key, pat in patterns.items():
        for line in lines:
            m = re.search(pat, line)
            if m:
                result[key] = m.group(1).strip()
                break
    return result

