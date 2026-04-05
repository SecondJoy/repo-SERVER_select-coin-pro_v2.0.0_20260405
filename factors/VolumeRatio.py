import numpy as np
import pandas as pd


def signal(*args):
    df = args[0]
    n = args[1]
    factor_name = args[2]

    upvolume = np.where(df['close'] > df['close'].shift(1), df['volume'], 0)
    downvolume = np.where(df['close'] < df['close'].shift(1), df['volume'], 0)

    upvolumes = pd.Series(upvolume).rolling(n, min_periods=1).sum()
    downvolumes = pd.Series(downvolume).rolling(n, min_periods=1).sum()

    ratio = (upvolumes ) / (1e-9 + downvolumes )

    df[factor_name] = ratio

    return df
