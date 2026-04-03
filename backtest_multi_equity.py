"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
import warnings

import pandas as pd

from config import swap_path, backtest_name
from core.backtest import find_best_params
from core.evaluate import strategy_evaluate
from core.figure import draw_equity_curve_plotly
from core.model.backtest_config import BacktestConfigFactory
from core.utils.log_kit import logger
from core.version import version_prompt

# ====================================================================================================
# ** 脚本运行前配置 **
# 主要是解决各种各样奇怪的问题们
# ====================================================================================================
# region 脚本运行前准备
warnings.filterwarnings('ignore')  # 过滤一下warnings，不要吓到老实人

# pandas相关的显示设置，基础课程都有介绍
pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 设置命令行输出时的列对齐功能
pd.set_option('display.unicode.east_asian_width', True)


def run():
    # ====================================================================================================
    # 1. 配置需要遍历的参数
    # ====================================================================================================
    pass

    # ====================================================================================================
    # 2. 生成策略配置
    # ====================================================================================================
    logger.info(f'生成策略配置...')
    backtest_factory = BacktestConfigFactory(backtest_name=f'多空分离-{backtest_name}')
    backtest_factory.generate_long_and_short_configs()

    # ====================================================================================================
    # 3. 寻找最优参数
    # ====================================================================================================
    find_best_params(backtest_factory)

    # ====================================================================================================
    # 4. 准备多空，多头，空头的结果输出
    # ====================================================================================================
    # 将策略的资金曲线合并到一个表格中然后绘图
    account_df = None  # 集合所有资金曲线
    rtn_df = None  # 集合所有策略评价
    rtn_cols = []
    # 遍历策略
    for conf in backtest_factory.config_list:
        # 读取资金曲线
        result_folder = conf.get_result_folder()  # 自动生成当前存储结果的文件夹
        df = pd.read_csv(result_folder / '资金曲线.csv', encoding='utf-8-sig', parse_dates=['candle_begin_time'])
        # 读取策略评价
        rtn, _, _, _ = strategy_evaluate(df, net_col='净值', pct_col='涨跌幅')
        # 合并策略评价数据
        if rtn_df is None:
            rtn_df = rtn
        else:
            rtn_df = pd.concat([rtn_df, rtn], axis=1)
        # 合并资金曲线数据
        df = df[['candle_begin_time', '净值', '净值dd2here']]
        if '多空模拟' in result_folder.name:
            re_columns = {'净值': '净值_多空', '净值dd2here': '最大回撤_多空'}
            rtn_cols.append('多空模拟')
        elif '纯多模拟' in result_folder.name:
            re_columns = {'净值': '净值_多头', '净值dd2here': '最大回撤_多头'}
            rtn_cols.append('纯多模拟')
        else:
            re_columns = {'净值': '净值_空头', '净值dd2here': '最大回撤_空头'}
            rtn_cols.append('纯空模拟')
        df.rename(columns=re_columns, inplace=True)
        if account_df is None:
            account_df = df
        else:
            account_df = pd.merge(account_df, df, on=['candle_begin_time'], how='outer')

    rtn_df.columns = rtn_cols
    print('\n策略评价\n', rtn_df)

    # 准备绘图，添加BTC/ETH涨跌幅
    all_swap = pd.read_pickle(swap_path)
    btc_df = all_swap['BTC-USDT']
    account_df = pd.merge(left=account_df, right=btc_df[['candle_begin_time', 'close']], on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['BTC涨跌幅'] = account_df['close'].pct_change()
    account_df['BTC涨跌幅'].fillna(value=0, inplace=True)
    account_df['BTC资金曲线'] = (account_df['BTC涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['BTC涨跌幅']

    eth_df = all_swap['ETH-USDT']
    account_df = pd.merge(left=account_df, right=eth_df[['candle_begin_time', 'close']], on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['ETH涨跌幅'] = account_df['close'].pct_change()
    account_df['ETH涨跌幅'].fillna(value=0, inplace=True)
    account_df['ETH资金曲线'] = (account_df['ETH涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['ETH涨跌幅']

    # 生成画图数据字典，可以画出所有offset资金曲线以及各个offset资金曲线
    data_dict = {'净值_多空': '净值_多空', '净值_多头': '净值_多头', '净值_空头': '净值_空头',
                 'BTC资金曲线': 'BTC资金曲线', 'ETH资金曲线': 'ETH资金曲线'}
    right_axis = {'最大回撤_多空': '最大回撤_多空', '最大回撤_多头': '最大回撤_多头', '最大回撤_空头': '最大回撤_空头'}

    # 调用画图函数
    draw_equity_curve_plotly(account_df, data_dict=data_dict, date_col='candle_begin_time', right_axis=right_axis,
                             title='多空资金曲线集合', desc='', path=backtest_factory.result_folder / '资金曲线.html')
    logger.ok(f'生成多空资金曲线集合完成')


if __name__ == '__main__':
    version_prompt()
    logger.info(f'系统启动中，稍等...')

    # ===================================================================================================
    # 运行回测，生成多空，多头，空头的资金曲线集合
    # ====================================================================================================
    run()
