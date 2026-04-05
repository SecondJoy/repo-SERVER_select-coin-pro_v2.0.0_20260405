
import numpy as np
import pandas as pd

def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    # 转换时间为pandas Series
    candle_series = pd.to_datetime(pd.Series(df['candle_begin_time']))
    # 满足有合约有现货的条件
    cond = (df['symbol_swap'] != '') & (df['symbol_spot'] != '')
    # 找到首次满足条件的索引
    if cond.any():
        first_idx = cond[cond].index[0]
        first_time = candle_series.iloc[first_idx]
        # 计算每一行到首次满足条件的时间差（小时）
        hours_since_first = (candle_series - first_time).dt.total_seconds() / 3600
        # 只有满足条件的行才赋值，其他为0
        df[factor_name] = np.where(cond, hours_since_first, np.nan)
    else:
        df[factor_name] = np.nan

    return df

