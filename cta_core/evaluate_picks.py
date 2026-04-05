import os
import sys
import argparse
import datetime as dt
import numpy as np
import pandas as pd

TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from eval_utils import read_project_paths
from eval_utils import compute_er_for_window


def _parse_date(s: str) -> dt.date:
    return pd.to_datetime(s).date()


def _parse_period(period: str) -> tuple:
    p = str(period).strip().lower()
    if p.endswith('min'):
        n = int(p[:-3])
        return ('m', n)
    if p.endswith('m'):
        n = int(p[:-1])
        return ('m', n)
    if p.endswith('h'):
        n = int(p[:-1])
        return ('h', n)
    if p.endswith('d'):
        n = int(p[:-1])
        return ('d', n)
    return ('h', 1)

def _symbol_variants(symbol: str) -> tuple:
    s = str(symbol).strip().upper()
    s = s.replace('-', '').replace('_', '')
    base = s[:-4] if s.endswith('USDT') else s
    with_dash = f"{base}-USDT"
    no_dash = f"{base}USDT"
    return with_dash, no_dash


#JoyTestStart
# JoyChange 20260405 原因：支持swap_dict.pkl（dict: symbol->DataFrame）数据源，统一DataFrame到OHLC并重采样
def _normalize_ohlc_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw is None or (not isinstance(df_raw, pd.DataFrame)) or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    df.columns = [str(c).lower() for c in df.columns]

    dt_col = None
    for cand in ['datetime', 'candle_begin_time', 'timestamp', 'ts']:
        if cand in df.columns:
            dt_col = cand
            break
    if not dt_col:
        return pd.DataFrame()

    if not all(c in df.columns for c in ['open', 'high', 'low', 'close']):
        return pd.DataFrame()

    out = pd.DataFrame({
        'datetime': pd.to_datetime(df[dt_col], errors='coerce'),
        'open': pd.to_numeric(df['open'], errors='coerce'),
        'high': pd.to_numeric(df['high'], errors='coerce'),
        'low': pd.to_numeric(df['low'], errors='coerce'),
        'close': pd.to_numeric(df['close'], errors='coerce'),
    })

    v_col = 'quote_volume' if 'quote_volume' in df.columns else ('volume' if 'volume' in df.columns else None)
    if v_col:
        out['quote_volume'] = pd.to_numeric(df[v_col], errors='coerce')

    out = out.dropna(subset=['datetime'])
    return out


def load_kline_period_from_df(df_raw: pd.DataFrame, start_dt: dt.datetime, end_dt: dt.datetime, period: str = '1h') -> pd.DataFrame:
    df = _normalize_ohlc_df(df_raw)
    if df.empty:
        return df

    df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()
    if df.empty:
        return pd.DataFrame()

    df.set_index('datetime', inplace=True)
    freq = str(period).lower()
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'quote_volume' in df.columns:
        agg['quote_volume'] = 'sum'
    d = df.resample(freq).agg(agg)
    d = d.dropna(subset=['open', 'high', 'low', 'close'])
    return d
#JoyTestEnd


def load_kline_period(kline_path: str, start_dt: dt.datetime, end_dt: dt.datetime, period: str = '1h') -> pd.DataFrame:
    ext = os.path.splitext(kline_path)[1].lower()
    # ========== 支持 PKL 格式 ==========
    if ext == '.pkl':
        df = pd.read_pickle(kline_path)
        # 标准化列名
        df.columns = [c.lower() for c in df.columns]
        # 查找时间列
        dt_col = None
        for cand in ['datetime', 'candle_begin_time', 'timestamp', 'ts']:
            if cand in df.columns:
                dt_col = cand
                break
        if not dt_col:
            return pd.DataFrame()
        # 查找价格列
        o_col = 'open' if 'open' in df.columns else None
        h_col = 'high' if 'high' in df.columns else None
        l_col = 'low' if 'low' in df.columns else None
        c_col = 'close' if 'close' in df.columns else None
        if not o_col or not h_col or not l_col or not c_col:
            return pd.DataFrame()
        v_col = 'quote_volume' if 'quote_volume' in df.columns else ('volume' if 'volume' in df.columns else None)
        df = pd.DataFrame({
            'datetime': pd.to_datetime(df[dt_col]),
            'open': pd.to_numeric(df[o_col], errors='coerce'),
            'high': pd.to_numeric(df[h_col], errors='coerce'),
            'low': pd.to_numeric(df[l_col], errors='coerce'),
            'close': pd.to_numeric(df[c_col], errors='coerce'),
        })
        if v_col:
            df['quote_volume'] = pd.to_numeric(df[v_col], errors='coerce')
    # ========== PKL 支持结束 ==========
    
    elif ext == '.parquet':
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(kline_path, columns=['datetime', 'open', 'high', 'low', 'close', 'quote_volume'])
            df = t.to_pandas()
        except Exception:
            df = pd.read_parquet(kline_path)
        if 'datetime' not in df.columns:
            return pd.DataFrame()
        df['datetime'] = pd.to_datetime(df['datetime'])
        cols = ['open', 'high', 'low', 'close']
        vol_col = 'quote_volume' if 'quote_volume' in df.columns else None
        df = df[['datetime'] + cols + ([vol_col] if vol_col else [])].copy()
    else:
        try:
            df_raw = pd.read_csv(kline_path, encoding='utf-8',  skiprows=1)
        except UnicodeDecodeError:
            df_raw = pd.read_csv(kline_path, encoding='gbk',  skiprows=1)

        
        cmap = {c.lower(): c for c in df_raw.columns}
        dt_col = None
        for cand in ['datetime', 'candle_begin_time', 'timestamp', 'ts']:
            if cand in cmap:
                dt_col = cmap[cand]
                break
        if not dt_col:
            return pd.DataFrame()
        o_col = cmap.get('open')
        h_col = cmap.get('high')
        l_col = cmap.get('low')
        c_col = cmap.get('close')
        if not o_col or not h_col or not l_col or not c_col:
            return pd.DataFrame()
        v_col = cmap.get('quote_volume') or cmap.get('volume') or None
        df = pd.DataFrame({
            'datetime': pd.to_datetime(df_raw[dt_col]),
            'open': pd.to_numeric(df_raw[o_col], errors='coerce'),
            'high': pd.to_numeric(df_raw[h_col], errors='coerce'),
            'low': pd.to_numeric(df_raw[l_col], errors='coerce'),
            'close': pd.to_numeric(df_raw[c_col], errors='coerce'),
        })
        if v_col:
            df['quote_volume'] = pd.to_numeric(df_raw[v_col], errors='coerce')
    df = df[(df['datetime'] >= start_dt) & (df['datetime'] <= end_dt)].copy()
    df.set_index('datetime', inplace=True)
    freq = str(period).lower()
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'quote_volume' in df.columns:
        agg['quote_volume'] = 'sum'
    d = df.resample(freq).agg(agg)
    d = d.dropna(subset=['open', 'high', 'low', 'close'])
    return d


def compute_forward_eff_amp(df: pd.DataFrame, paths: dict, kline_period: str = '1h', bars: int = 72, segment_start_mode: str = 'next-day', penalty_coef_dd: float = 0.5, penalty_coef_zz: float = 0.5, amp_mode: str = 'pct', amp_beta: float = 1.0) -> pd.DataFrame:
    syms = df['symbol'].astype(str).unique().tolist()
    dates = sorted(pd.to_datetime(df['date']).dt.date.unique())
    unit, qty = _parse_period(kline_period)
    if segment_start_mode.lower() == 'same-day':
        ext_start_dt = dt.datetime.combine(min(dates), dt.time(0, 0))
    else:
        ext_start_dt = dt.datetime.combine(min(dates), dt.time(0, 0)) + dt.timedelta(days=1)
    if unit == 'h':
        ext_end_dt = dt.datetime.combine(max(dates), dt.time(0, 0)) + dt.timedelta(hours=qty * bars)
    elif unit == 'm':
        ext_end_dt = dt.datetime.combine(max(dates), dt.time(0, 0)) + dt.timedelta(minutes=qty * bars)
    elif unit == 'd':
        ext_end_dt = dt.datetime.combine(max(dates), dt.time(0, 0)) + dt.timedelta(days=qty * bars)
    else:
        ext_end_dt = dt.datetime.combine(max(dates), dt.time(0, 0)) + dt.timedelta(hours=bars)
    # JoyChange 20260405 原因：支持kline_root为swap_dict.pkl（dict: symbol->DataFrame），并保留原目录模式
    sym_1h_map = {}
    kline_dict = paths.get('kline_dict')
    kroot = paths.get('kline_parquet')

    kline_dict_index = None
    if isinstance(kline_dict, dict):
        kline_dict_index = paths.get('_kline_dict_index')
        if kline_dict_index is None:
            kline_dict_index = {}
            for k in kline_dict.keys():
                nk = str(k).strip().upper().replace('-', '').replace('_', '')
                if nk and nk not in kline_dict_index:
                    kline_dict_index[nk] = k
            paths['_kline_dict_index'] = kline_dict_index

    for s in syms:
        if isinstance(kline_dict, dict):
            sym_with_dash, sym_no_dash = _symbol_variants(s)
            df_sym = None
            for cand_name in [s, sym_with_dash, sym_no_dash]:
                if cand_name in kline_dict:
                    df_sym = kline_dict.get(cand_name)
                    break
            if df_sym is None and isinstance(kline_dict_index, dict):
                for nk in [
                    str(s).strip().upper().replace('-', '').replace('_', ''),
                    sym_with_dash.replace('-', '').replace('_', ''),
                    sym_no_dash.replace('-', '').replace('_', ''),
                ]:
                    key = kline_dict_index.get(nk)
                    if key is not None:
                        df_sym = kline_dict.get(key)
                        if df_sym is not None:
                            break
            if not isinstance(df_sym, pd.DataFrame):
                continue
            ohlc_1h = load_kline_period_from_df(df_sym, ext_start_dt, ext_end_dt, kline_period)
            sym_1h_map[s] = ohlc_1h
            continue

        p = None
        sym_with_dash, sym_no_dash = _symbol_variants(s)
        for ext in ['.parquet', '.pkl', '.csv']:
            for cand_name in [s, sym_with_dash, sym_no_dash]:
                cand = os.path.join(kroot, f"{cand_name}{ext}")
                if os.path.isfile(cand):
                    p = cand
                    break
            if p:
                break
        if not p:
            names = sorted(os.listdir(kroot))
            cands = []
            for name in names:
                en = name.lower()
                if not (en.endswith('.parquet') or en.endswith('.pkl') or en.endswith('.csv')):
                    continue
                if (sym_with_dash.lower() in en) or (sym_no_dash.lower() in en):
                    cands.append(os.path.join(kroot, name))
            p = cands[0] if cands else None
        if not p or not os.path.isfile(p):
            continue
        ohlc_1h = load_kline_period(p, ext_start_dt, ext_end_dt, kline_period)
        sym_1h_map[s] = ohlc_1h
    rows = []
    for _, r in df.iterrows():
        d = pd.to_datetime(r['date']).date()
        s = str(r['symbol'])
        layer = r.get('layer', None)
        bucket = r.get('bucket', None)
        ohlc_1h = sym_1h_map.get(s)
        if ohlc_1h is None or ohlc_1h.empty:
            continue
        start_win = dt.datetime.combine(d, dt.time(0, 0)) + dt.timedelta(days=1)
        if segment_start_mode.lower() == 'same-day':
            start_win = dt.datetime.combine(d, dt.time(0, 0))
        if unit == 'h':
            end_win = start_win + dt.timedelta(hours=qty * bars)
        elif unit == 'm':
            end_win = start_win + dt.timedelta(minutes=qty * bars)
        elif unit == 'd':
            end_win = start_win + dt.timedelta(days=qty * bars)
        else:
            end_win = start_win + dt.timedelta(hours=bars)
        sub = ohlc_1h[(ohlc_1h.index >= start_win) & (ohlc_1h.index < end_win)]
        m2 = compute_er_for_window(sub)
        amp_comp = m2['max_amp_pct'] if amp_mode.lower() == 'pct' else m2['max_amp_atr']
        log_ms = np.log1p(m2['move_scale']) if np.isfinite(m2['move_scale']) and m2['move_scale'] > -1 else np.nan
        er = m2['er'] if np.isfinite(m2['er']) else np.nan
        x = min(1.0, max(0.0, float(er))) if np.isfinite(er) else np.nan
        er_smooth = (3.0 * x * x - 2.0 * x * x * x) if np.isfinite(x) else np.nan
        eff_amp_logms_er_smooth = (er_smooth * m2['r2'] * (log_ms + amp_beta * amp_comp)) if np.isfinite(er_smooth) and np.isfinite(m2['r2']) and np.isfinite(log_ms) and np.isfinite(amp_comp) else np.nan
        denom_pen = (1.0 + (penalty_coef_dd * m2['drawdown_atr'] if np.isfinite(m2['drawdown_atr']) else 0.0) + (penalty_coef_zz * m2['zigzag_ratio'] if np.isfinite(m2['zigzag_ratio']) else 0.0))
        eff_penal = (eff_amp_logms_er_smooth / denom_pen) if np.isfinite(eff_amp_logms_er_smooth) and denom_pen > 0 else np.nan
        rows.append({'date': d, 'symbol': s, 'layer': layer, 'bucket': bucket, 'period': kline_period, 'bars': bars, 'eff_amp_logms_er_smooth_penal': eff_penal})
    out = pd.DataFrame(rows)
    return out


def summarize_by_strategy(picks: pd.DataFrame, fr: pd.DataFrame, year_for_default: int, hours: int) -> pd.DataFrame:
    px = picks.copy()
    px['date'] = pd.to_datetime(px['date']).dt.date
    px['layer'] = px['layer'] if 'layer' in px.columns else None
    px['bucket'] = px['bucket'] if 'bucket' in px.columns else None
    frp = fr.copy()
    pxm2 = px.merge(frp, on=['date', 'symbol'], how='left')
    rows = []
    if 'strategy' in pxm2.columns:
        for strat, gs in pxm2.groupby('strategy'):
            s = pd.Series({
                'strategy': strat,
                'year': year_for_default,
                'hours': hours,
                'count': pd.to_numeric(gs['eff_amp_logms_er_smooth_penal'], errors='coerce').count(),
                'mean': pd.to_numeric(gs['eff_amp_logms_er_smooth_penal'], errors='coerce').mean(),
                'median': pd.to_numeric(gs['eff_amp_logms_er_smooth_penal'], errors='coerce').median(),
                'p75': pd.to_numeric(gs['eff_amp_logms_er_smooth_penal'], errors='coerce').quantile(0.75),
                'p90': pd.to_numeric(gs['eff_amp_logms_er_smooth_penal'], errors='coerce').quantile(0.90)
            })
            rows.append(s)
    out = pd.DataFrame(rows)
    return out.sort_values(['strategy']) if not out.empty else out


def make_monthly_mean_line(fr: pd.DataFrame, out_html: str):
    from pyecharts import options as opts
    from pyecharts.charts import Line, Grid
    df = fr.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['ym'] = df['date'].dt.strftime('%Y-%m')
    g = df.groupby('ym')['eff_amp_logms_er_smooth_penal'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean()).reset_index()
    xs = g['ym'].tolist()
    ys = [None if pd.isna(v) else float(v) for v in g['eff_amp_logms_er_smooth_penal']]
    ln = Line().add_xaxis(xs).add_yaxis('monthly_mean', [None if v is None else round(v, 6) for v in ys], is_symbol_show=False, label_opts=opts.LabelOpts(is_show=False))
    ln.set_global_opts(title_opts=opts.TitleOpts(title='Monthly mean of evaluation'), tooltip_opts=opts.TooltipOpts(trigger='axis'), xaxis_opts=opts.AxisOpts(type_='category'), yaxis_opts=opts.AxisOpts(min_='dataMin', max_='dataMax'))
    grid = Grid()
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    grid.add(ln, grid_opts=opts.GridOpts(pos_left='5%', pos_right='2%', pos_top='8%', pos_bottom='8%'))
    grid.render(out_html)
    print(f"OUTPUT_HTML: {out_html}")


def visualize_day_symbols(paths: dict, day: dt.date, period: str, bars: int, out_html: str, symbols: list):
    from pyecharts import options as opts
    from pyecharts.charts import Kline, Line, Grid, Page
    # JoyChange 20260405 原因：绘图阶段兼容swap_dict.pkl数据源
    kline_dict = paths.get('kline_dict')
    kroot = paths.get('kline_parquet')
    unit, qty = _parse_period(period)
    start_dt = dt.datetime.combine(day, dt.time(0, 0)) + dt.timedelta(days=1)
    if unit == 'h':
        end_dt = start_dt + dt.timedelta(hours=qty * bars)
    elif unit == 'm':
        end_dt = start_dt + dt.timedelta(minutes=qty * bars)
    elif unit == 'd':
        end_dt = start_dt + dt.timedelta(days=qty * bars)
    else:
        end_dt = start_dt + dt.timedelta(hours=bars)
    page = Page()
    for sym in symbols:
        if isinstance(kline_dict, dict):
            sym_with_dash, sym_no_dash = _symbol_variants(sym)
            df_sym = None
            for cand_name in [sym, sym_with_dash, sym_no_dash]:
                if cand_name in kline_dict:
                    df_sym = kline_dict.get(cand_name)
                    break
            if not isinstance(df_sym, pd.DataFrame):
                continue
            ohlc = load_kline_period_from_df(df_sym, start_dt, end_dt, period)
        else:
            p = None
            sym_with_dash, sym_no_dash = _symbol_variants(sym)
            for ext in ['.parquet', '.pkl', '.csv']:
                for cand_name in [sym, sym_with_dash, sym_no_dash]:
                    cand = os.path.join(kroot, f"{cand_name}{ext}")
                    if os.path.isfile(cand):
                        p = cand
                        break
                if p:
                    break
            if not p:
                names = sorted(os.listdir(kroot))
                cands = []
                for name in names:
                    en = name.lower()
                    if not (en.endswith('.parquet') or en.endswith('.pkl') or en.endswith('.csv')):
                        continue
                    if (sym_with_dash.lower() in en) or (sym_no_dash.lower() in en):
                        cands.append(os.path.join(kroot, name))
                p = cands[0] if cands else None
            if not p or not os.path.isfile(p):
                continue
            ohlc = load_kline_period(p, start_dt, end_dt, period)
        if ohlc is None or ohlc.empty:
            continue
        em = compute_er_for_window(ohlc)
        xs = [dt.datetime.strftime(ts, '%Y-%m-%d %H:%M') for ts in ohlc.index]
        data = [[float(o), float(c), float(l), float(h)] for o, c, l, h in zip(ohlc['open'], ohlc['close'], ohlc['low'], ohlc['high'])]
        k = Kline().add_xaxis(xs).add_yaxis(sym, data)
        ln = Line().add_xaxis(xs).add_yaxis('close', [float(v) for v in ohlc['close']], is_symbol_show=False)
        k.overlap(ln)
        title = f"{sym} ER:{round(em['er'], 4) if np.isfinite(em['er']) else 'nan'} R2:{round(em['r2'], 4) if np.isfinite(em['r2']) else 'nan'} MoveATR:{round(em['move_scale'], 4) if np.isfinite(em['move_scale']) else 'nan'} MaxAmp%:{round(em['max_amp_pct']*100, 2) if np.isfinite(em['max_amp_pct']) else 'nan'} DDATR:{round(em['drawdown_atr'], 4) if np.isfinite(em['drawdown_atr']) else 'nan'} ZZ:{round(em['zigzag_ratio'], 4) if np.isfinite(em['zigzag_ratio']) else 'nan'}"
        k.set_global_opts(title_opts=opts.TitleOpts(title=title), datazoom_opts=[opts.DataZoomOpts(xaxis_index=[0], filter_mode='filter', range_start=0, range_end=100), opts.DataZoomOpts(type_='inside', xaxis_index=[0], filter_mode='filter', range_start=0, range_end=100)])
        g = Grid()
        g.add(k, grid_opts=opts.GridOpts(pos_left='5%', pos_right='2%', pos_top='8%', pos_bottom='8%'))
        page.add(g)
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    page.render(out_html)
    print(f"OUTPUT_HTML: {out_html}")


def make_monthly_mean_embed(fr: pd.DataFrame) -> str:
    from pyecharts import options as opts
    from pyecharts.charts import Line, Grid
    df = fr.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['ym'] = df['date'].dt.strftime('%Y-%m')
    g = df.groupby('ym')['eff_amp_logms_er_smooth_penal'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean()).reset_index()
    xs = g['ym'].tolist()
    ys = [None if pd.isna(v) else float(v) for v in g['eff_amp_logms_er_smooth_penal']]
    ln = Line().add_xaxis(xs).add_yaxis('monthly_mean', [None if v is None else round(v, 6) for v in ys], is_symbol_show=False, label_opts=opts.LabelOpts(is_show=False))
    ln.set_global_opts(title_opts=opts.TitleOpts(title='Monthly mean of evaluation'), tooltip_opts=opts.TooltipOpts(trigger='axis'), xaxis_opts=opts.AxisOpts(type_='category'), yaxis_opts=opts.AxisOpts(min_='dataMin', max_='dataMax'))
    grid = Grid()
    grid.add(ln, grid_opts=opts.GridOpts(pos_left='5%', pos_right='2%', pos_top='8%', pos_bottom='8%'))
    return grid.render_embed()


def generate_summary_index(out_dir: str, base_name: str, period: str, bars: int, fr: pd.DataFrame, summary_rows: pd.DataFrame, monthly_embed: str, top_info: list, bot_info: list):
    def _fmt(v):
        try:
            x = float(v)
            return f"{x:.6f}"
        except Exception:
            return str(v)
    html = []
    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'><title>综合报告</title>")
    html.append("<style>body{font-family:Arial,Helvetica,sans-serif;margin:20px;} h2{margin-top:24px;} table{border-collapse:collapse;width:100%;margin-top:12px;} th,td{border:1px solid #ddd;padding:8px;text-align:left;} tr:nth-child(even){background:#f9f9f9;} .section{margin-bottom:28px;} .grid{margin-top:12px;} .list a{color:#1a73e8;text-decoration:none;} .list a:hover{text-decoration:underline;}</style>")
    html.append("</head><body>")
    html.append(f"<h1>综合报告：{base_name}（周期 {period}, 观察数 {bars}）</h1>")
    html.append("<div class='section'><h2>总体与年度统计</h2>")
    html.append("<table><thead><tr><th>组别</th><th>count</th><th>mean</th><th>median</th><th>p75</th><th>p90</th></tr></thead><tbody>")
    for _, r in summary_rows.iterrows():
        grp = r.get('group', '')
        cnt = r.get('count', '')
        mean = _fmt(r.get('mean', ''))
        median = _fmt(r.get('median', ''))
        p75 = _fmt(r.get('p75', ''))
        p90 = _fmt(r.get('p90', ''))
        html.append(f"<tr><td>{grp}</td><td>{cnt}</td><td>{mean}</td><td>{median}</td><td>{p75}</td><td>{p90}</td></tr>")
    html.append("</tbody></table></div>")
    html.append("<div class='section'><h2>月度均值</h2>")
    html.append("<div class='grid'>")
    html.append(monthly_embed)
    html.append("</div></div>")
    html.append("<div class='section'><h2>Top10 单日均值</h2><div class='list'><ul>")
    for day, mean, link in top_info:
        html.append(f"<li>{day} — 均值 {mean:.6f} — <a href='{link}' target='_blank'>查看K线</a></li>")
    html.append("</ul></div></div>")
    html.append("<div class='section'><h2>Bottom10 单日均值</h2><div class='list'><ul>")
    for day, mean, link in bot_info:
        html.append(f"<li>{day} — 均值 {mean:.6f} — <a href='{link}' target='_blank'>查看K线</a></li>")
    html.append("</ul></div></div>")
    html.append("</body></html>")
    out_path = os.path.join(out_dir, f"report_index_{period}_{bars}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"OUTPUT_HTML: {out_path}")


def main(cfg:dict ={}):
    from types import SimpleNamespace
    import fnmatch
    args = SimpleNamespace(
        picks_csv=cfg.get("picks_csv", ""),
        picks_path=cfg.get("picks_path", ""),
        picks_pattern=cfg.get("picks_pattern", ""),
        picks_root_dir=cfg.get("picks_root_dir", os.path.dirname(os.path.dirname(__file__))),
        picks_file_name=cfg.get("picks_file_name", "final_select_results.*"),
        kline_period=cfg.get("kline_period", "1h"),
        bars=int(cfg.get("bars", 72)),
        out_dir=cfg.get("out_dir", os.path.join(os.path.dirname(__file__), "evl_output")),
        kline_root=cfg.get("kline_root",None ),
        segment_start_mode=cfg.get("segment_start_mode", "next-day"),
        amp_mode=cfg.get("amp_mode", "pct"),
        amp_beta=float(cfg.get("amp_beta", 1.0)),
        penalty_dd=float(cfg.get("penalty_dd", 0.5)),
        penalty_zz=float(cfg.get("penalty_zz", 0.5)),
        do_visual=bool(cfg.get("do_visual", True)),
    )
    picks_path = None
    if args.picks_csv and str(args.picks_csv).strip() and os.path.isfile(args.picks_csv):
        picks_path = args.picks_csv
    if not picks_path and args.picks_path and str(args.picks_path).strip():
        cand = args.picks_path
        if os.path.isfile(cand):
            picks_path = cand
        elif os.path.isdir(cand):
            names = sorted(os.listdir(cand))
            selected = None
            pat = args.picks_pattern.strip()
            if pat:
                for name in names:
                    if fnmatch.fnmatch(name, pat):
                        selected = os.path.join(cand, name)
                        break
            else:
                if not selected:
                    files = [os.path.join(cand, name) for name in names if name.lower().endswith('.csv') or name.lower().endswith('.pkl')]
                    if files:
                        selected = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)[0]
            if selected and os.path.isfile(selected):
                picks_path = selected
            else:
                raise RuntimeError('no csv found in picks-path')
        else:
            raise RuntimeError('invalid picks-path')
    if not picks_path:
        roots = []
        if args.picks_root_dir and os.path.isdir(args.picks_root_dir):
            roots.append(args.picks_root_dir)
        proj_root = os.path.dirname(os.path.dirname(__file__))
        if os.path.isdir(proj_root):
            roots.append(proj_root)
        script_dir = os.path.dirname(__file__)
        if os.path.isdir(script_dir):
            roots.append(script_dir)
        pat = args.picks_file_name.strip() or args.picks_pattern.strip() or "final_select_results.*"
        selected = None
        for rd in roots:
            names = sorted(os.listdir(rd))
            if pat:
                for name in names:
                    if fnmatch.fnmatch(name, pat):
                        selected = os.path.join(rd, name)
                        break
            if not selected:
                files = [os.path.join(rd, name) for name in names if name.lower().endswith('.csv') or name.lower().endswith('.pkl')]
                if files:
                    selected = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)[0]
            if selected and os.path.isfile(selected):
                picks_path = selected
                break
    if not picks_path or not os.path.isfile(picks_path):
        raise RuntimeError('picks csv not found')
    print(f"SELECTED_CSV: {picks_path}")
    ext_sel = os.path.splitext(picks_path)[1].lower()
    if ext_sel == '.pkl':
        df = pd.read_pickle(picks_path)
    else:
        df = pd.read_csv(picks_path)
    if 'candle_begin_time' in df.columns:
        df['date'] = pd.to_datetime(df['candle_begin_time']).dt.date
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date']).dt.date
    else:
        raise RuntimeError('csv missing required date column: candle_begin_time or date')
    if 'symbol' not in df.columns:
        raise RuntimeError('csv missing required symbol column')
    df['symbol'] = df['symbol'].astype(str)
    df = df[['date', 'symbol']].drop_duplicates()
    base_years = sorted(pd.Series(df['date']).dropna().apply(lambda d: d.year).unique().tolist())
    year_dir = 'multi_year' if len(base_years) > 1 else str(base_years[0])
    base_name = os.path.splitext(os.path.basename(picks_path))[0]
    # Modify redundant path design: if base_name is final_select_results, skip creating a subfolder for it
    if base_name == 'final_select_results':
        out_dir = os.path.join(args.out_dir, year_dir)
    else:
        out_dir = os.path.join(args.out_dir, year_dir, base_name)
    os.makedirs(out_dir, exist_ok=True)

    # JoyChange 20260405 原因：允许kline_root直接配置为swap_dict.pkl（dict: symbol->DataFrame）
    paths = {}
    if args.kline_root and isinstance(args.kline_root, str) and args.kline_root.lower().endswith('.pkl') and os.path.isfile(args.kline_root):
        _kdata = pd.read_pickle(args.kline_root)
        if isinstance(_kdata, dict):
            paths['kline_dict'] = _kdata
            paths['kline_parquet'] = None
        else:
            paths['kline_parquet'] = args.kline_root
    else:
        paths['kline_parquet'] = args.kline_root

    picks = df[['date', 'symbol']].copy()
    fr = compute_forward_eff_amp(picks, paths, kline_period=args.kline_period, bars=args.bars, segment_start_mode=args.segment_start_mode, penalty_coef_dd=args.penalty_dd, penalty_coef_zz=args.penalty_zz, amp_mode=args.amp_mode, amp_beta=args.amp_beta)
    if not fr.empty:
        fr['year'] = pd.to_datetime(fr['date']).dt.year
    out_forward = os.path.join(out_dir, f'forward_eval_{args.kline_period}_{args.bars}.csv')
    fr.to_csv(out_forward, index=False)
    print(f"OUTPUT_CSV: {out_forward}")
    if not fr.empty and 'bars' in fr.columns and 'period' in fr.columns:
        rows = []
        overall = pd.to_numeric(fr['eff_amp_logms_er_smooth_penal'], errors='coerce')
        rows.append(pd.Series({'group': 'overall', 'period': args.kline_period, 'bars': args.bars, 'count': overall.count(), 'mean': overall.mean(), 'median': overall.median(), 'p75': overall.quantile(0.75), 'p90': overall.quantile(0.90)}))
        for y, gy in fr.groupby('year'):
            s = pd.Series({'group': f'year_{y}', 'period': args.kline_period, 'bars': args.bars, 'count': pd.to_numeric(gy['eff_amp_logms_er_smooth_penal'], errors='coerce').count(), 'mean': pd.to_numeric(gy['eff_amp_logms_er_smooth_penal'], errors='coerce').mean(), 'median': pd.to_numeric(gy['eff_amp_logms_er_smooth_penal'], errors='coerce').median(), 'p75': pd.to_numeric(gy['eff_amp_logms_er_smooth_penal'], errors='coerce').quantile(0.75), 'p90': pd.to_numeric(gy['eff_amp_logms_er_smooth_penal'], errors='coerce').quantile(0.90)})
            rows.append(s)
        out_summ_overall = os.path.join(out_dir, f'summary_overall_{args.kline_period}_{args.bars}.csv')
        pd.DataFrame(rows).to_csv(out_summ_overall, index=False)
        print(f"OUTPUT_CSV: {out_summ_overall}")
        monthly_embed = None
        try:
            make_monthly_mean_line(fr, os.path.join(out_dir, f'monthly_mean_{args.kline_period}_{args.bars}.html'))
            monthly_embed = make_monthly_mean_embed(fr)
        except Exception:
            monthly_embed = "<p>Monthly chart render failed.</p>"
        top_info = []
        bot_info = []
        try:
            dmean = fr.groupby('date')['eff_amp_logms_er_smooth_penal'].apply(lambda x: pd.to_numeric(x, errors='coerce').mean()).reset_index()
            dmean_sorted = dmean.sort_values('eff_amp_logms_er_smooth_penal', ascending=False)
            top_days = dmean_sorted.head(10)['date'].tolist()
            bot_days = dmean_sorted.tail(10)['date'].tolist()
            for day in top_days:
                syms = fr[fr['date'] == day]['symbol'].astype(str).unique().tolist()
                out_html = os.path.join(out_dir, f'top_day_{day}_{args.kline_period}_{args.bars}.html')
                visualize_day_symbols(paths, day, args.kline_period, args.bars, out_html, syms)
                mv = float(dmean[dmean['date'] == day]['eff_amp_logms_er_smooth_penal'].iloc[0])
                top_info.append((str(day), mv, os.path.relpath(out_html, out_dir)))
            for day in bot_days:
                syms = fr[fr['date'] == day]['symbol'].astype(str).unique().tolist()
                out_html = os.path.join(out_dir, f'bottom_day_{day}_{args.kline_period}_{args.bars}.html')
                visualize_day_symbols(paths, day, args.kline_period, args.bars, out_html, syms)
                mv = float(dmean[dmean['date'] == day]['eff_amp_logms_er_smooth_penal'].iloc[0])
                bot_info.append((str(day), mv, os.path.relpath(out_html, out_dir)))
        except Exception as e:
            print(f"VISUALIZE_TOP_BOTTOM_FAIL: {e}")
        try:
            base_name = os.path.splitext(os.path.basename(picks_path))[0]
            
            # Determine report title name
            report_title_name = base_name
            if base_name == 'final_select_results':
                # If base_name is generic, try to use the parent folder name (factor name)
                # args.out_dir is typically .../FactorName/evl_output
                parent_dir = os.path.dirname(args.out_dir.rstrip(os.sep))
                report_title_name = os.path.basename(parent_dir)

            summary_rows_df = pd.DataFrame(rows)
            generate_summary_index(out_dir, report_title_name, args.kline_period, args.bars, fr, summary_rows_df, monthly_embed, top_info, bot_info)
        except Exception:
            pass

def run_evaluation_task(config:dict):
    if config is None:
        try:
            import eval_config
            config = getattr(eval_config, "CONFIG", {})
        except Exception:
            config = {}

    main(cfg=config)

if __name__ == '__main__':
    main()