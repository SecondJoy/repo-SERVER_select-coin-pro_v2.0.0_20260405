#JoyAddded CTA币池因子趋势强度评估_20260329直播
import numpy as np
import pandas as pd

def atr_mean(ohlc: pd.DataFrame) -> float:
    c_prev = ohlc['close'].shift(1)
    tr1 = (ohlc['high'] - ohlc['low']).abs()
    tr2 = (ohlc['high'] - c_prev).abs()
    tr3 = (ohlc['low'] - c_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return float(tr.mean())

def compute_er_for_window(ohlc: pd.DataFrame) -> dict:
    if ohlc is None or ohlc.empty:
        return {'er': np.nan, 'move_scale': np.nan, 'noise_scale': np.nan, 'er_amp': np.nan, 'r2': np.nan, 'eff_score': np.nan, 'drawdown_atr': np.nan, 'zigzag_ratio': np.nan, 'max_amp_pct': np.nan, 'max_amp_atr': np.nan}
    atrm = atr_mean(ohlc)
    c = pd.Series(ohlc['close']).astype(float)
    if len(c) < 2:
        return {'er': np.nan, 'move_scale': np.nan, 'noise_scale': np.nan, 'er_amp': np.nan, 'r2': np.nan, 'eff_score': np.nan, 'drawdown_atr': np.nan, 'zigzag_ratio': np.nan, 'max_amp_pct': np.nan, 'max_amp_atr': np.nan}
    net = float(abs(c.iloc[-1] - c.iloc[0]))
    path = float(c.diff().abs().sum())
    er = net / path if path > 0 else np.nan
    move_scale = net / atrm if atrm and atrm > 0 else np.nan
    noise_scale = path / atrm if atrm and atrm > 0 else np.nan
    er_amp = er * move_scale if np.isfinite(er) and np.isfinite(move_scale) else np.nan
    
    # Adaptive Adverse Excursion (Drawdown for Long, Drawup for Short)
    direction = c.iloc[-1] - c.iloc[0]
    if direction >= 0:
        # Long trend: Penalize drop from peak
        run_max = c.cummax()
        adverse_excursion = float((run_max - c).max())
    else:
        # Short trend: Penalize rally from trough
        run_min = c.cummin()
        adverse_excursion = float((c - run_min).max())
        
    drawdown_atr = adverse_excursion / atrm if atrm and atrm > 0 else np.nan
    zigzag_ratio = (noise_scale / move_scale) if np.isfinite(noise_scale) and np.isfinite(move_scale) and move_scale != 0 else np.nan
    try:
        hmax = float(ohlc['high'].max())
        lmin = float(ohlc['low'].min())
        amp_abs = hmax - lmin
    except Exception:
        amp_abs = np.nan
    max_amp_pct = (amp_abs / c.iloc[0]) if np.isfinite(amp_abs) and len(c) and c.iloc[0] > 0 else np.nan
    max_amp_atr = (amp_abs / atrm) if np.isfinite(amp_abs) and atrm and atrm > 0 else np.nan
    x = np.arange(len(c), dtype=float)
    try:
        coef = np.polyfit(x, c.values, 1)
        yhat = coef[0] * x + coef[1]
        ss_res = float(np.sum((c.values - yhat) ** 2))
        ss_tot = float(np.sum((c.values - np.mean(c.values)) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    except Exception:
        r2 = np.nan
    eff_score = er_amp * r2 if np.isfinite(er_amp) and np.isfinite(r2) else np.nan
    return {'er': er, 'move_scale': move_scale, 'noise_scale': noise_scale, 'er_amp': er_amp, 'r2': r2, 'eff_score': eff_score, 'drawdown_atr': drawdown_atr, 'zigzag_ratio': zigzag_ratio, 'max_amp_pct': max_amp_pct, 'max_amp_atr': max_amp_atr}

