"""
邢不行™️ 策略分享会
仓位管理框架

版权所有 ©️ 邢不行
微信: xbx6660

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
extra_data_dict = {
    'coin-btc': ['btc_close']
}


def signal(*args):
    """计算与BTC指数的相关性"""
    df = args[0]
    n = args[1]
    factor_name = args[2]

    # 使用 extra_data_dict 获取的 btc_close 计算指数涨跌幅
    if 'btc_close' in df.columns:
        df['指数涨跌幅'] = df['btc_close'].pct_change()
    else:
        df['指数涨跌幅'] = 0
    df['指数涨跌幅'].fillna(0, inplace=True)

    indicator1 = df['close'].pct_change()
    indicator2 = df['指数涨跌幅']
    corr = indicator1.rolling(n, min_periods=1).corr(indicator2)

    df[factor_name] = corr

    return df
