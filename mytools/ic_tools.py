import concurrent.futures
import numpy as np
import pandas as pd
from numpy.typing import NDArray
import numba as nb
from tqdm import tqdm
import numba.experimental as nb_exp
import config
from mytools.models import OHLCData, AdditionInfo


def cal_ic(factor_val: pd.Series, next_close_col: pd.Series, hold_hours: int, ) -> float:
    factor_scores = factor_val[:-hold_hours]
    returns = next_close_col[:-hold_hours]
    covariance = np.cov(factor_scores, returns)[0][1]
    return covariance / (factor_scores.std() * returns.std())


def cal_factor_ic(factor_name: str, n: int, hold_hours: int) -> float:
    ohlc = OHLCData()
    factors = [(factor_name, n)]
    symbols = ohlc.symbols
    factor_col_name = f"{factor_name}_{n}"
    return_col = f"return_col"
    hold_info = AdditionInfo(
        hold_hours,

    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.job_num) as executor:
        futures = [executor.submit(ohlc.cal_factor_one_symbol, symbol, factors, hold_info)
                   for
                   symbol in symbols]
    res = []
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        df = future.result()
        df[return_col] = df["close"].shift(-hold_hours) / df['close'] - 1
        df = df.dropna(subset=[factor_col_name])
        df = df.dropna(subset=[return_col])
        res.append(df)
    red_df = pd.concat(res, ignore_index=True)
    grouped = red_df.groupby("candle_begin_time")

    group_res = []
    for category, group in tqdm(grouped, desc="计算分组"):
        if group.shape[0] < 10:
            continue
        try:
            # 将 B 列等分成 10 组
            group['B_group'] = pd.qcut(group[factor_col_name], q=10, labels=False)
            # 计算每组中 C 的平均值
            mean_val = group.groupby('B_group')[return_col].mean().reset_index()
            group_res.append(mean_val)
        except ValueError:
            continue

    # 对每个组应用处理函数

    group_df = pd.concat(group_res, ignore_index=True)
    # 将结果转换为 NumPy 数组
    result_array = group_df.groupby('B_group')[return_col].mean().to_numpy()
    ic = cal_ic(red_df[factor_col_name], red_df[return_col], hold_hours)
    return [ic] + result_array[::-1]


IC_DATA_SPEC = [
    ('symbol_id', nb.int64[:]),
    ('close', nb.float64[:]),
    ('next_close', nb.float64[:]),
    ('next_return', nb.float64[:]),
    ('factor', nb.float64[:]),
    ('n_symbol', nb.int64),
    ('n_group', nb.int64),
    ('group_start', nb.int64[:]),
]


@nb_exp.jitclass(IC_DATA_SPEC)
class ICDataSets:
    def __init__(
            self,
            symbol_id: NDArray[np.int64],
            close: NDArray[np.float64],
            next_close: NDArray[np.float64],
            next_return: NDArray[np.float64],
            factor: NDArray[np.float64],
            n_symbol: np.int64 | int,
            n_group: np.int64,
            group_start: NDArray[np.int64],
    ):
        self.symbol_id = symbol_id
        self.close = close
        self.n_symbol = n_symbol
        self.next_close = next_close
        self.factor = factor
        self.next_return = next_return
        self.n_group = n_group
        self.group_start = group_start


def convert_2_ic_dataset(data: pd.DataFrame, factor_name: str) -> ICDataSets:
    data.sort_values(['candle_begin_time', 'symbol'], inplace=True)
    all_symbol_list = sorted(list(set(data['symbol'].unique())))
    n_symbol = len(all_symbol_list)
    symbol_to_int = {v: k for k, v in enumerate(all_symbol_list)}
    data['symbol_id'] = data['symbol'].map(symbol_to_int)
    arr_cbt = data['candle_begin_time'].values
    unique_times, group_start = np.unique(arr_cbt, return_index=True)
    group_start = np.append(group_start, arr_cbt.size)
    return ICDataSets(
        symbol_id=data['symbol_id'].values,
        close=data['close'].values,
        next_close=data['next_close'].values,
        next_return=data['next_return'].values,
        factor=data[factor_name].values,
        n_symbol=n_symbol,
        n_group=unique_times.size,
        group_start=group_start,
    )


@nb.njit
def cal_ic_dataset(dataset: ICDataSets) -> int:
    for period_idx in range(dataset.n_group):
        _ = process_one_ic_period(dataset, period_idx)
    return 1


@nb.njit()
def process_one_ic_period(dataset: ICDataSets, period_idx: int, num_bins=10) -> tuple[
    np.float64, np.ndarray, np.ndarray]:
    start_idx = dataset.group_start[period_idx]
    end_idx = dataset.group_start[period_idx + 1] if period_idx < dataset.n_group - 1 else len(dataset.close)
    index_rank = sort_and_get_indices(dataset.factor, start_idx, end_idx)[start_idx:end_idx]

    num = end_idx - start_idx
    bin_size = num // num_bins
    remainder = num % num_bins

    groups_num = np.full(num_bins, bin_size)
    for idx in range(remainder):
        groups_num[-idx] = groups_num[-idx] + 1

    curr_group = 0
    curr_group_idx = 0
    groups_sum = np.zeros(num_bins, dtype=np.float64)
    for idx in index_rank:
        if curr_group_idx >= groups_num[curr_group]:
            curr_group_idx = 0
            curr_group += 1

        groups_sum[curr_group] = groups_sum[curr_group] + dataset.next_return[idx]
        curr_group_idx = curr_group_idx + 1
    ic = calculate_ic(dataset.factor, dataset.next_return)
    return ic, groups_num, groups_sum


@nb.njit
def calculate_ic(factor, future_return):
    n = len(factor)

    # 计算因子和未来收益的排名
    rank_factor = np.argsort(np.argsort(factor))
    rank_return = np.argsort(np.argsort(future_return))

    # 计算 Spearman 相关系数
    d = rank_factor - rank_return
    ic = 1 - (6 * np.sum(d ** 2)) / (n * (n ** 2 - 1))

    return ic


@nb.njit
def quicksort(arr, indices, start, end):
    if start >= end:
        return indices

    pivot_index = partition(arr, indices, start, end)
    quicksort(arr, indices, start, pivot_index - 1)
    quicksort(arr, indices, pivot_index + 1, end)

    return indices


@nb.njit
def partition(arr, indices, start, end):
    pivot = arr[indices[end]]
    i = start - 1

    for j in range(start, end):
        if arr[indices[j]] < pivot:
            i += 1
            indices[i], indices[j] = indices[j], indices[i]

    indices[i + 1], indices[end] = indices[end], indices[i + 1]
    return i + 1


@nb.njit
def sort_and_get_indices(arr, start, end):
    indices = np.arange(len(arr))
    sorted_indices = quicksort(arr, indices, start, end - 1)
    return sorted_indices
