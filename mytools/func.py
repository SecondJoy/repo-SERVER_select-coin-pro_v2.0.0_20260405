import dataclasses
import importlib
import importlib.util
import os
import pkgutil
import re
import datetime
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Any, List

import numpy as np
import pandas as pd
import zipfile

from numpy import ndarray
from pandas import Categorical, CategoricalDtype, Series
from pandas.core.arrays import ExtensionArray

from core.utils.log_kit import get_logger
from core.utils.path_kit import get_folder_path, get_file_path

logger = get_logger()


def pct_str_to_float(s: str) -> float | None:
    match = re.search(r'([-+]?\d*\.?\d+)', s)

    if match:
        # 提取到的数字字符串
        number_str = match.group(1)
        # 转换为浮点数
        number = float(number_str)
        return number
    else:
        print("没有找到数字")


def categorize_group(group, group_by_col, group_num: int = 10):
    group[f'{group_by_col}_group'] = pd.qcut(group[group_by_col], group_num, labels=range(group_num))
    return group


def cache_do_nothing():
    pass


def cache_only_df(df: pd.DataFrame) -> pd.DataFrame:
    return df


def cache_data(load_func: Callable[..., Any] | None, name: str, force_refresh=False,
               refresh_hour=168, *args, **kwargs) -> Any:
    if load_func is None:
        load_func = cache_do_nothing
    cache_folder = Path(get_folder_path('data', 'analysis_cache'))
    if not cache_folder.exists():
        cache_folder.mkdir(parents=True)
    matching_files = [file.name for file in cache_folder.iterdir() if file.is_file() and file.name.startswith(name)]

    if len(matching_files) == 0 or force_refresh:
        df = load_func(*args, **kwargs)
        now = datetime.datetime.now()
        date_time_string = now.strftime("%Y-%m-%d-%H-%M-%S")
        cache_name = f"{name}.{date_time_string}.pickle"
        pd.to_pickle(df, cache_folder / cache_name)
        return df
    else:
        cache_filename = matching_files[0]
        cache = cache_folder / cache_filename
        date_parts = cache_filename.split('.')
        last_refresh_time = datetime.datetime.strptime(date_parts[1], "%Y-%m-%d-%H-%M-%S")
        now = datetime.datetime.now()
        diff_hour = (now - last_refresh_time).seconds / 3600
        if diff_hour > refresh_hour:
            cache.unlink()
            df = load_func(*args, **kwargs)
            now = datetime.datetime.now()
            date_time_string = now.strftime("%Y-%m-%d-%H-%M-%S")
            cache_name = f"{name}.{date_time_string}.pickle"
            pd.to_pickle(df, cache_folder / cache_name)
            logger.info(f"上次缓存时间为{last_refresh_time},超过更新时间{refresh_hour},更新缓存")
            return df

        logger.info(f"load {name} from cache at {last_refresh_time}")
        return pd.read_pickle(cache)


def reload_module_py(module_name: str):
    spec = importlib.util.find_spec(module_name)
    factors_pkg = importlib.import_module(module_name)
    # 仅当确实是包（可以有子模块路径）时再遍历子模块并重载
    if getattr(spec, 'submodule_search_locations', None):
        for finder, modname, ispkg in pkgutil.iter_modules(factors_pkg.__path__):
            full_name = f"{factors_pkg.__name__}.{modname}"
            try:
                m = importlib.import_module(full_name)
                importlib.reload(m)
            except Exception:
                continue


class Snowflake:
    def __init__(self, datacenter_id, machine_id):
        self.epoch = int(time.time() * 1000)
        self.datacenter_id = datacenter_id
        self.machine_id = machine_id
        self.sequence = 0
        self.last_timestamp = self.epoch
        self.lock = threading.Lock()

    def generate(self):
        with self.lock:
            timestamp = int(time.time() * 1000)
            if timestamp < self.last_timestamp:
                raise Exception("Clock moved backwards")

            if timestamp == self.last_timestamp:
                self.sequence += 1
                if self.sequence > 4095:
                    raise Exception("Sequence exceeded")
            else:
                self.sequence = 0
                self.last_timestamp = timestamp

            snowflake_id = (
                                   timestamp - self.epoch) << 22 | self.datacenter_id << 17 | self.machine_id << 12 | self.sequence
            return snowflake_id


def top_drawdowns(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """
    输入:
        returns: 收益率序列（DatetimeIndex），例如 equity.pct_change().fillna(0)
        top: 需要的前 N 个回撤区间
    输出:
        DataFrame: 列包含 [start, trough, end, drawdown, duration]
        - drawdown 为最深处的回撤幅度（负数）
        - duration 为区间长度（以原索引频率计数，可按需换算为天/小时）
    """
    returns = returns.astype(float).ffill().fillna(0.0)
    # 转净值曲线
    prices = (1.0 + returns).cumprod()
    # 回撤序列（0 到负数）
    cummax = prices.cummax()
    dd = prices / cummax - 1.0

    is_dd = dd < 0
    # 回撤开始：由非回撤->回撤
    start_idx = np.flatnonzero((~is_dd.shift(fill_value=False)) & is_dd)
    # 回撤结束：由回撤->非回撤（恢复到新高）
    end_idx = np.flatnonzero(is_dd & (~is_dd.shift(-1, fill_value=False)))

    rows = []
    si = 0
    for e in end_idx:
        # 匹配相应的开始点
        while si < len(start_idx) and start_idx[si] > e:
            si += 1
        if si >= len(start_idx):
            break
        s = start_idx[si]
        if s > e:
            continue
        window = dd.iloc[s:e + 1]
        trough_loc = window.idxmin()
        trough_dd = window.min()
        rows.append({
            "start": dd.index[s],
            "valley": trough_loc,
            "end": dd.index[e],
            "drawdown": float(trough_dd),
            "hours": e - s + 1
        })
        si += 1

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 从深到浅排序（drawdown 为负数，越小越深）
    out = out.sort_values("drawdown").head(top).reset_index(drop=True)
    return out


def add_n_hour_return(
        df: pd.DataFrame,
        n: int,
        symbol_col: str = "symbol",
        time_col: str = "candle_begin_time",
        close_col: str = "close",
        out_col: str | None = None,
        inplace: bool = False
) -> pd.DataFrame:
    """为行情数据计算 n 小时后的收益率列（未来收益）。

    计算逻辑：
      1. 先尝试按精确时间匹配 (symbol, time + n小时) 找到对应的未来 close 值。
      2. 如果精确匹配找不到，则退回到按 symbol 分组使用 shift(-n) 获取未来第 n 条记录的 close 作为备用。
      3. 计算未来收益 = future_close / current_close - 1。遇到 current_close == 0 时返回 NaN，避免除零。

    参数校验和行为与原函数保持一致，但方向由过去改为未来。
    """
    # 参数校验
    if n <= 0:
        raise ValueError("n must be a positive integer")

    required_cols = {symbol_col, time_col, close_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Required column(s) {missing} not found")

    out_col_name = out_col if out_col else f"return_{n}h"

    # working copy
    working = df if inplace else df.copy()

    # preserve original order index for alignment
    working['_orig_index_for_return'] = range(len(working))

    # ensure datetime
    if not np.issubdtype(working[time_col].dtype, np.datetime64):
        try:
            working[time_col] = pd.to_datetime(working[time_col])
        except Exception as e:
            raise TypeError("time_col must be datetime64 or convertible") from e

    # sort for stable operations
    working.sort_values([symbol_col, time_col], inplace=True)

    # compute future time and a group-shift fallback
    working['__future_time'] = working[time_col] + pd.Timedelta(hours=n)
    # shift(-n) gives the close n rows ahead within each symbol (fallback)
    working['__shift_future_close'] = working.groupby(symbol_col)[close_col].shift(-n)

    # base table to merge: (symbol, time) -> future close
    base = working[[symbol_col, time_col, close_col]].copy().rename(columns={close_col: '__future_close'})

    merged = working.merge(
        base,
        left_on=[symbol_col, '__future_time'],
        right_on=[symbol_col, time_col],
        how='left',
        suffixes=('', '_future')
    )

    # if exact match not found, use shifted future close
    merged['__future_close'] = merged['__future_close'].fillna(merged['__shift_future_close'])

    # compute future return, guard divide-by-zero
    merged[out_col_name] = np.where(
        merged[close_col] == 0,
        np.nan,
        (merged['__future_close'] / merged[close_col] - 1)
    )

    # cleanup helper cols
    drop_cols = ['__future_time', '__shift_future_close', '__future_close']
    # remove any duplicated time columns created by merge
    future_time_col = f"{time_col}_future"
    if future_time_col in merged.columns:
        drop_cols.append(future_time_col)
    merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)

    # restore original order
    merged.sort_values('_orig_index_for_return', inplace=True)

    # assign back if inplace, preserving original df index/shape
    if inplace:
        df[out_col_name] = merged[out_col_name].values
        # remove helper column from df if present
        if '_orig_index_for_return' in df.columns:
            df.drop(columns=['_orig_index_for_return'], inplace=True)
        return df

    # drop helper orig index before returning
    merged.drop(columns=['_orig_index_for_return'], inplace=True)
    return merged


def add_n_hour_extreme_points(
        df: pd.DataFrame,
        n: int,
        symbol_col: str = "symbol",
        time_col: str = "candle_begin_time",
        close_col: str = "close",
        max_hour_col: str | None = None,
        min_hour_col: str | None = None,
        inplace: bool = False
) -> pd.DataFrame:
    """为行情数据计算 n 小时内达到最高点和最低点的小时数。

    计算逻辑：
      1. 对每个 symbol，按时间排序
      2. 对于每一行，查看未来 n 小时内的 close 价格
      3. 找到最高价和最低价出现的位置（相对当前时间的小时数）
      4. 返回两列：达到最高点的小时数和达到最低点的小时数

    参数：
        df: 输入的行情数据 DataFrame
        n: 向前看的小时数
        symbol_col: 交易对列名
        time_col: 时间列名
        close_col: 收盘价列名
        max_hour_col: 输出的最高点小时数列名（默认为 "max_hour_{n}h"）
        min_hour_col: 输出的最低点小时数列名（默认为 "min_hour_{n}h"）
        inplace: 是否在原 DataFrame 上修改

    返回：
        处理后的 DataFrame，包含最高点和最低点的小时数列
    """
    # 参数校验
    if n <= 0:
        raise ValueError("n must be a positive integer")

    required_cols = {symbol_col, time_col, close_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Required column(s) {missing} not found")

    max_hour_col_name = max_hour_col if max_hour_col else f"max_hour_{n}h"
    min_hour_col_name = min_hour_col if min_hour_col else f"min_hour_{n}h"

    # working copy
    working = df if inplace else df.copy()

    # preserve original order index for alignment
    working['_orig_index_for_extreme'] = range(len(working))

    # ensure datetime
    if not np.issubdtype(working[time_col].dtype, np.datetime64):
        try:
            working[time_col] = pd.to_datetime(working[time_col])
        except Exception as e:
            raise TypeError("time_col must be datetime64 or convertible") from e

    # sort for stable operations
    working.sort_values([symbol_col, time_col], inplace=True)

    # 计算每个symbol的极值点小时数
    def calc_extreme_hours(group):
        result_max = []
        result_min = []

        for idx in range(len(group)):
            # 获取未来n个时间点（包括当前点）
            future_slice = group.iloc[idx:min(idx + n + 1, len(group))]

            if len(future_slice) <= 1:
                # 如果没有未来数据，返回 NaN
                result_max.append(np.nan)
                result_min.append(np.nan)
            else:
                # 获取未来价格
                future_prices = future_slice[close_col].values

                # 找到最高价和最低价的位置（相对于当前位置的小时数）
                max_idx = np.argmax(future_prices)
                min_idx = np.argmin(future_prices)

                result_max.append(max_idx)
                result_min.append(min_idx)

        group = group.copy()
        group.loc[:, max_hour_col_name] = result_max
        group.loc[:, min_hour_col_name] = result_min
        return group

    # 按 symbol 分组计算，保留所有列
    # 使用 lambda 包装来避免 pandas 2.1+ 的 DeprecationWarning
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        working = working.groupby(symbol_col, group_keys=False).apply(calc_extreme_hours).reset_index(drop=True)

    # restore original order
    if '_orig_index_for_extreme' in working.columns:
        working = working.sort_values('_orig_index_for_extreme').reset_index(drop=True)

    # assign back if inplace
    if inplace:
        df[max_hour_col_name] = working[max_hour_col_name].values
        df[min_hour_col_name] = working[min_hour_col_name].values
        if '_orig_index_for_extreme' in df.columns:
            df.drop(columns=['_orig_index_for_extreme'], inplace=True)
        return df

    # drop helper orig index before returning
    working.drop(columns=['_orig_index_for_extreme'], inplace=True)
    return working


def cal_next(group, next_hour: int = 10):
    group = group.sort_values(by="candle_begin_time", ascending=True)
    group['next_close'] = group['close'].shift(-next_hour)
    group['ret_next'] = group['next_close'] / group['close'] - 1
    return group


def factor_group_return(df: pd.DataFrame, factor_name: str, bins: int = 10) -> List[float]:
    df['total_coins'] = df.groupby('candle_begin_time', observed=True)['symbol'].transform('size')
    # 过滤有效数据
    valid_df = df[df['total_coins'] >= bins].copy()
    valid_df['rank'] = valid_df.groupby('candle_begin_time', observed=True)[factor_name].rank(method='first')
    valid_df = valid_df.dropna(subset=[factor_name])

    # 计算 ret_next（如果不存在）
    if 'ret_next' not in valid_df.columns:
        if 'next_close' in valid_df.columns and 'close' in valid_df.columns:
            valid_df['ret_next'] = valid_df['next_close'] / valid_df['close'] - 1

    # 分组处理 - 使用安全的分组函数
    def safe_qcut(x, bins, labels):
        try:
            return pd.qcut(x, q=bins, labels=labels, duplicates='drop')
        except (ValueError, IndexError):
            # 当数据点太少无法分组时,返回 NaN
            return pd.Series([np.nan] * len(x), index=x.index)

    labels = [i for i in range(1, bins + 1)]
    valid_df['groups'] = valid_df.groupby('candle_begin_time', observed=True)['rank'].transform(
        lambda x: safe_qcut(x, bins, labels)
    )

    # 过滤掉 groups 为 NaN 的行
    valid_df = valid_df.dropna(subset=['groups'])

    group_returns = valid_df.groupby(['candle_begin_time', 'groups'], observed=True)['ret_next'].mean().to_frame()
    group_returns.reset_index('groups', inplace=True)
    group_returns = group_returns.reset_index()
    group_returns = pd.pivot(group_returns,
                             index='candle_begin_time',
                             columns='groups',
                             values='ret_next')
    group_curve = (group_returns + 1).cumprod()
    group_curve = group_curve[labels]
    group_curve = group_curve.ffill()
    bar_df = group_curve.iloc[-1].reset_index()
    bar_df.columns = ['groups', 'asset']
    return bar_df['asset'].tolist()

@dataclasses.dataclass
class BinInfo:
    cut_type: str  # 'qcut' 或 'cut'
    group_period: str  # 'all' 或 'hour'

def daily_bin_counts(source, date_col='date', value_col='perv_group', bins_count: int = 10, new_col: str = 'Zf_bin',
                     decimal_places=5, bin_info: BinInfo | None = None) -> pd.DataFrame:
    if bin_info is None:
        bin_info = BinInfo(cut_type='cut', group_period='all')

    if bins_count <= 0:
        raise ValueError("bins_count must be a positive integer")
    if not isinstance(decimal_places, int) or decimal_places < 0:
        raise ValueError("decimal_places must be a non-negative integer")

    df = source.copy()

    # 确保日期列为日期（按日统计）
    if date_col not in df.columns:
        raise KeyError(f"column '{date_col}' not found")
    try:
        df[date_col] = pd.to_datetime(df[date_col]).dt.normalize()
    except Exception:
        # 如果无法转换则保留原样（分组时仍按原值分组）
        pass

    vals = df[value_col].dropna()
    # helper：格式化数字区间标签
    fmt = lambda x: format(x, f'.{decimal_places}f')
    clean_interval = lambda s: re.sub(r'[\(\)\[\]]', '', str(s))

    # 如果没有有效数值，返回空的 pivot（索引为所有日期去重）
    if vals.empty:
        idx = pd.Index(sorted(df[date_col].unique())) if len(df) else pd.Index([], name=date_col)
        return pd.DataFrame(index=idx)

    vmin = float(vals.min())
    vmax = float(vals.max())

    # 常数列处理：构造单一区间标签
    if vmin == vmax:
        label = f"{fmt(vmin)}-{fmt(vmax)}"
        df['bin'] = pd.NA
        mask = df[value_col].notna()
        df.loc[mask, 'bin'] = label
        # pivot 统计并返回
        result = df.groupby([date_col, 'bin']).size().unstack(fill_value=0).sort_index()
        return result

    # 非常数列：根据 cut_type 选择分箱方式
    if bin_info.cut_type == 'qcut':
        try:
            q = min(bins_count, len(vals))
            # 对整个列分箱（保留 NaN 不参与分箱）
            binned = pd.qcut(df[value_col], q=q, duplicates='drop')
            # categories 可能比 q 少（duplicates drop），生成 columns 顺序
            if isinstance(binned.dtype, CategoricalDtype):
                cats = [clean_interval(c) for c in binned.cat.categories]
            else:
                # fallback：从非空值中推断
                cats = []
            # 将分箱转换为清洗后的字符串（保持 NaN）
            df['bin'] = binned.astype(str).where(binned.notna()).apply(lambda x: clean_interval(x) if pd.notna(x) else x)
        except Exception:
            # qcut 失败，返回空 pivot
            idx = pd.Index(sorted(df[date_col].unique())) if len(df) else pd.Index([], name=date_col)
            return pd.DataFrame(index=idx)
    else:
        # 使用等宽 cut，与 bin_group 一致的标签格式
        edges = np.linspace(vmin, vmax, bins_count + 1)
        labels = [f"{fmt(edges[i])}-{fmt(edges[i + 1])}" for i in range(len(edges) - 1)]
        df['bin'] = pd.cut(df[value_col], bins=edges, labels=labels, include_lowest=True, right=True)
        # 如果是 Categorical，保留其字符串表示（pd.cut 已返回 category）
        if isinstance(df['bin'].dtype, CategoricalDtype):
            df['bin'] = df['bin'].astype(str)

    # 最后按日期和区间统计并 pivot（缺失区间将不在结果列中，使用 fill_value=0）
    result = df.groupby([date_col, 'bin']).size().unstack(fill_value=0).sort_index()
    return result





# python
def bin_group(source: pd.DataFrame, value_col: str, bins_count: int = 10, new_col: str = 'Zf_bin',
              decimal_places: int = 5, bin_info: BinInfo | None = None) -> pd.DataFrame:
    if bin_info is None:
        bin_info = BinInfo(cut_type='cut', group_period='all')

    if bins_count <= 0:
        raise ValueError("bins_count must be a positive integer")
    if not isinstance(decimal_places, int) or decimal_places < 0:
        raise ValueError("decimal_places must be a non-negative integer")

    df = source.copy()

    def _bin_subframe(sub_df: pd.DataFrame, labels_override: list | None = None):
        vals = sub_df[value_col].dropna()
        if vals.empty:
            dtype = pd.Int64Dtype() if labels_override else "string"
            return pd.Series([pd.NA] * len(sub_df), index=sub_df.index, dtype=dtype)

        v_min = float(vals.min())
        v_max = float(vals.max())
        fmt = lambda x: format(x, f'.{decimal_places}f')

        if v_min == v_max:
            if labels_override:
                label = labels_override[0]
                series = pd.Series([pd.NA] * len(sub_df), index=sub_df.index, dtype=pd.Int64Dtype())
                mask = sub_df[value_col].notna()
                series.loc[mask] = label
                return pd.Categorical(series, categories=labels_override, ordered=True)
            label = f"{fmt(v_min)}-{fmt(v_max)}"
            series = pd.Series([pd.NA] * len(sub_df), index=sub_df.index, dtype="string")
            mask = sub_df[value_col].notna()
            series.loc[mask] = label
            return pd.Categorical(series, categories=[label], ordered=True)

        if bin_info.cut_type == 'qcut':
            try:
                q = min(bins_count, len(vals))
                if labels_override:
                    binned = pd.qcut(sub_df[value_col], q=q, labels=labels_override[:q], duplicates='drop')
                    return binned.astype(pd.CategoricalDtype(categories=labels_override[:q], ordered=True))
                binned = pd.qcut(sub_df[value_col], q=q, duplicates='drop')
                return binned.astype(str).apply(lambda x: re.sub(r'[(\]\[]', '', x) if pd.notna(x) else x)
            except ValueError:
                dtype = pd.Int64Dtype() if labels_override else "string"
                return pd.Series([pd.NA] * len(sub_df), index=sub_df.index, dtype=dtype)

        edges = np.linspace(v_min, v_max, bins_count + 1)
        if labels_override:
            labels = labels_override
            binned = pd.cut(sub_df[value_col], bins=edges, labels=labels[:len(edges)-1], include_lowest=True, right=True)
            return binned.astype(pd.CategoricalDtype(categories=labels[:len(edges)-1], ordered=True))

        labels = [f"{fmt(edges[i])}-{fmt(edges[i + 1])}" for i in range(len(edges) - 1)]
        binned = pd.cut(sub_df[value_col], bins=edges, labels=labels, include_lowest=True, right=True)
        return binned.astype(pd.CategoricalDtype(categories=labels, ordered=True))

    # 分组模式：按小时（candle_begin_time）独立分箱
    if bin_info.group_period == 'hour':
        if 'candle_begin_time' not in df.columns:
            raise KeyError("column 'candle_begin_time' not found")

        hour_labels = list(range(bins_count))

        def _process_hour(idx: int, sub_df: pd.DataFrame):
            working = sub_df.copy()
            if len(working) < bins_count:
                working[new_col] = pd.Series([pd.NA] * len(working), index=working.index, dtype=pd.Int64Dtype())
            else:
                working[new_col] = _bin_subframe(working, labels_override=hour_labels)
            return idx, working

        grouped_frames = []
        max_workers = min(6, os.cpu_count())
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for idx, (_, sub) in enumerate(df.groupby('candle_begin_time', observed=True, sort=False)):
                futures.append(executor.submit(_process_hour, idx, sub))
            for fut in futures:
                grouped_frames.append(fut.result())

        grouped_frames.sort(key=lambda x: x[0])
        valid_frames = [frame for _, frame in grouped_frames if not frame[new_col].isna().all()]
        if not valid_frames:
            return df.iloc[0:0].copy()
        df = pd.concat(valid_frames).sort_index()
        df = df.dropna(subset=[new_col])
        return df

    # 默认模式：全局分箱
    df[new_col] = _bin_subframe(df)
    return df


def cleanup_analysis_cache(threshold_gb=20):
    analysis_cache_folder = get_file_path('data', 'analysis_cache', as_path_type=True)
    # 如果目录不存在或不是目录，提前返回
    if not (analysis_cache_folder.exists() and analysis_cache_folder.is_dir()):
        return

    # 计算目录大小（递归），单位：字节
    total_size = 0
    for root, dirs, files in os.walk(analysis_cache_folder):
        for fname in files:
            try:
                fp = os.path.join(root, fname)
                total_size += os.path.getsize(fp)
            except Exception:
                # 忽略无法访问的文件
                pass
    size_gb = total_size / (1024 ** 3)
    if total_size < threshold_gb * 1024 ** 3:
        print(f"analysis_cache 文件夹大小 {size_gb:.2f} GB，小于阈值 {threshold_gb} GB，跳过清理。")
        return

    print(f"analysis_cache 文件夹大小 {size_gb:.2f} GB，超过阈值 {threshold_gb} GB，开始清理...")

    # 进行清理：删除文件、符号链接和子目录（递归）
    for child in analysis_cache_folder.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                cache_filename = child.name
                date_parts = cache_filename.split('.')
                last_refresh_time = datetime.datetime.strptime(date_parts[1], "%Y-%m-%d-%H-%M-%S")
                now = datetime.datetime.now()
                diff_hour = (now - last_refresh_time).seconds / 3600
                if diff_hour > 12:
                    child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except Exception as e:
            print(f"无法删除文件 {child}: {e}")

def diff_transaction(right_df:pd.DataFrame, left_df:pd.DataFrame,diff_type:str='diff')->pd.DataFrame:
    """对比两份交易记录，找出不同的交易。
    两个df的header是['symbol', 'start_time', 'end_time', 'start_close', 'end_close','direction', 'offset', 'is_spot', 'profit', 'strategy']
    df每一行代表一个transaction。其中symbol,start_time,end_time,direction相同。认为是同一笔交易。
    参数:
        right_df (pd.DataFrame): 新的交易记录数据框。
        left_df (pd.DataFrame): 旧的交易记录数据框。
        diff_type (str): 对比类型，
         - 'same_buy' 一笔交易的只有end_time不同的交易
         - 'diff' 完全不同的交易
    返回:
        pd.DataFrame: 包含差异记录的 DataFrame。
    """
    required_cols = ['symbol', 'start_time', 'end_time', 'start_close', 'end_close', 'direction', 'offset', 'is_spot', 'profit', 'strategy']
    missing_right = set(required_cols) - set(right_df.columns)
    missing_left = set(required_cols) - set(left_df.columns)
    if missing_right:
        raise KeyError(f"right_df 缺少列: {missing_right}")
    if missing_left:
        raise KeyError(f"left_df 缺少列: {missing_left}")

    if diff_type not in {'diff', 'same_buy'}:
        raise ValueError("diff_type must be 'diff' or 'same_buy'")

    if diff_type == 'diff':
        key_cols = ['symbol', 'start_time', 'end_time', 'direction']

        # 标记原始索引，便于过滤后还原
        right_work = right_df.reset_index(drop=False).rename(columns={'index': '__idx_right'})
        left_work = left_df.reset_index(drop=False).rename(columns={'index': '__idx_left'})

        # 找到同 symbol+direction 且时间区间有交集的记录，视为相同，先剔除
        cross = right_work.merge(left_work, on=['symbol', 'direction'], suffixes=('_r', '_l'), how='inner')
        overlap_mask = (cross['start_time_r'] <= cross['end_time_l']) & (cross['end_time_r'] >= cross['start_time_l'])
        matched_right_idx = set(cross.loc[overlap_mask, '__idx_right'])
        matched_left_idx = set(cross.loc[overlap_mask, '__idx_left'])

        if matched_right_idx:
            right_work = right_work[~right_work['__idx_right'].isin(matched_right_idx)]
        if matched_left_idx:
            left_work = left_work[~left_work['__idx_left'].isin(matched_left_idx)]

        # 剩余部分按完整键比较差异
        right_keys = right_work[key_cols].apply(tuple, axis=1)
        left_keys = left_work[key_cols].apply(tuple, axis=1)
        added_idx = right_keys[~right_keys.isin(left_keys)].index
        removed_idx = left_keys[~left_keys.isin(right_keys)].index

        added = right_work.loc[added_idx].copy()
        removed = left_work.loc[removed_idx].copy()
        out = pd.concat([added, removed], ignore_index=True)
        return out[right_df.columns]
    # diff_type == 'same_buy'
    base_key = ['symbol', 'start_time', 'direction']
    merged = right_df.merge(left_df, on=base_key, suffixes=('_right', '_left'), how='inner')
    diff_mask = merged['end_time_right'] != merged['end_time_left']
    if diff_mask.empty or not diff_mask.any():
        return right_df.iloc[0:0].copy()

    changed_keys = merged.loc[diff_mask, base_key]
    key_tuples = set(changed_keys.apply(tuple, axis=1))

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        idx_mask = df[base_key].apply(tuple, axis=1).isin(key_tuples)
        return df.loc[idx_mask].copy()

    added_rows = _filter(right_df)
    removed_rows = _filter(left_df)
    result = pd.concat([added_rows, removed_rows], ignore_index=True)
    return result

def assign_equal_groups(df, source_col, target_col, n_groups=10, per_group_cols=None, ascending=True):
    """
    在 DataFrame 上将 source_col 做等额分组（等频），结果写入 target_col。
    per_group_cols: None 表示全局分组；若为列名列表，则在每个子组内分别分组。
    ascending=True: 小值分到组号小的组；ascending=False 则将标签反转（大值组号小 -> 可通过改动labels实现）。
    """
    labels = list(range(1, n_groups + 1)) if ascending else list(range(n_groups, 0, -1))

    def _bin_series(s: pd.Series):
        valid = s.dropna()
        if valid.nunique() < n_groups:
            # fallback: use rank -> pct -> ceil
            ranks = s.rank(method='first', pct=True)
            bins = (ranks * n_groups).apply(lambda x: int(ceil(x)) if pd.notna(x) and x > 0 else (np.nan if pd.isna(x) else 1))
            bins = bins.clip(1, n_groups)
            if not ascending:
                bins = bins.apply(lambda b: (n_groups - b + 1) if pd.notna(b) else b)
            return bins
        else:
            try:
                binned = pd.qcut(s, q=n_groups, labels=labels, duplicates='drop')
                # qcut 返回 Categorical，保持 NaN，转换为整型
                # 当 duplicates='drop' 使实际箱数少于 n_groups 时，转换需要照顾
                if isinstance(binned.dtype, pd.CategoricalDtype):
                    # 先把类别映射到整数；类别可能不是连续 1..n_groups
                    cat = binned.cat
                    mapping = {cat.categories[i]: labels[i] for i in range(len(cat.categories))}
                    return binned.map(mapping).astype('Int64')
                else:
                    return binned.astype('Int64')
            except Exception:
                # 最后兜底
                ranks = s.rank(method='first', pct=True)
                bins = (ranks * n_groups).apply(lambda x: int(ceil(x)) if pd.notna(x) and x > 0 else (np.nan if pd.isna(x) else 1))
                bins = bins.clip(1, n_groups)
                if not ascending:
                    bins = bins.apply(lambda b: (n_groups - b + 1) if pd.notna(b) else b)
                return bins

    if per_group_cols is None:
        df[target_col] = _bin_series(df[source_col])
    else:
        # 按每个子组分别分配
        def apply_group(g):
            res = _bin_series(g[source_col])
            return res
        grouped = df.groupby(per_group_cols, group_keys=False)
        df[target_col] = grouped.apply(lambda g: apply_group(g)).astype('Int64')
    return df