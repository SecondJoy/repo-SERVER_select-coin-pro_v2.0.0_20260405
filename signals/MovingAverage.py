"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
import pandas as pd


def dynamic_leverage(equity: pd.Series, *args) -> pd.Series:
    """
    根据资金曲线，动态调整杠杆
    :param equity: 资金曲线
    :param args: 其他参数
    :return: 返回包含 leverage 的数据
    """

    # ===== 获取策略参数
    n = int(args[0])

    # ===== 计算指标
    # 计算均线
    ma = equity.rolling(n, min_periods=1).mean()

    # 默认空仓
    leverage = pd.Series(0., index=equity.index)

    # equity 在均线之上，才持有
    above = equity > ma
    leverage.loc[above] = 1.

    return leverage
