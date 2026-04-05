#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy  as np
import pandas as pd
import talib as ta
eps = 1e-8


def signal(*args):
    # Bolling_width 指标
    df = args[0]
    n  = args[1]
    factor_name = args[2]

    df['median'] = df['close'].rolling(window=n).mean()
    df['std'] = df['close'].rolling(n, min_periods=1).std(ddof=0)
    df['z_score'] = abs(df['close'] - df['median']) / df['std']
    df['m'] = df['z_score'].rolling(window=n).mean()
    df['upper'] = df['median'] + df['std'] * df['m']
    df['lower'] = df['median'] - df['std'] * df['m']
    df[factor_name] = df['std'] * df['m'] * 2 / (df['median'] + eps)

    # 删除多余列
    del df['median'], df['std'], df['z_score'], df['m']
    del df['upper'], df['lower']

    return df
