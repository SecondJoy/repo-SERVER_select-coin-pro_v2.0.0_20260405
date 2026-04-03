"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
import itertools

import numpy as np
import pandas as pd


# 计算策略评价指标
def strategy_evaluate(equity, net_col='多空资金曲线', pct_col='本周期多空涨跌幅'):
    """
    回测评价函数
    :param equity: 资金曲线数据
    :param net_col: 资金曲线列名
    :param pct_col: 周期涨跌幅列名
    :return:
    """
    # ===新建一个dataframe保存回测指标
    results = pd.DataFrame()

    # 将数字转为百分数
    def num_to_pct(value):
        return '%.2f%%' % (value * 100)

    # ===计算累积净值
    results.loc[0, '累积净值'] = round(equity[net_col].iloc[-1], 2)

    # ===计算年化收益
    annual_return = (equity[net_col].iloc[-1]) ** (
            '1 days 00:00:00' / (equity['candle_begin_time'].iloc[-1] - equity['candle_begin_time'].iloc[0]) * 365) - 1
    results.loc[0, '年化收益'] = num_to_pct(annual_return)

    # ===计算最大回撤，最大回撤的含义：《如何通过3行代码计算最大回撤》https://mp.weixin.qq.com/s/Dwt4lkKR_PEnWRprLlvPVw
    # 计算当日之前的资金曲线的最高点
    equity[f'{net_col.split("资金曲线")[0]}max2here'] = equity[net_col].expanding().max()
    # 计算到历史最高值到当日的跌幅，drowdwon
    equity[f'{net_col.split("资金曲线")[0]}dd2here'] = equity[net_col] / equity[f'{net_col.split("资金曲线")[0]}max2here'] - 1
    # 计算最大回撤，以及最大回撤结束时间
    end_date, max_draw_down = tuple(equity.sort_values(by=[f'{net_col.split("资金曲线")[0]}dd2here']).iloc[0][['candle_begin_time', f'{net_col.split("资金曲线")[0]}dd2here']])
    # 计算最大回撤开始时间
    start_date = equity[equity['candle_begin_time'] <= end_date].sort_values(by=net_col, ascending=False).iloc[0]['candle_begin_time']
    results.loc[0, '最大回撤'] = num_to_pct(max_draw_down)
    results.loc[0, '最大回撤开始时间'] = str(start_date)
    results.loc[0, '最大回撤结束时间'] = str(end_date)
    # ===年化收益/回撤比：我个人比较关注的一个指标
    results.loc[0, '年化收益/回撤比'] = round(annual_return / abs(max_draw_down), 2)
    mean_back_zf = 1 / (1 + equity[f'{net_col.split("资金曲线")[0]}dd2here']) - 1  # 回本涨幅
    mean_fix_zf = mean_back_zf.mean()  # 修复涨幅
    max_back_zf = 1 / (1 + max_draw_down) - 1  # 回本涨幅
    max_fix_zf = max_back_zf.mean()  # 修复涨幅
    results.loc[0, '修复涨幅（均/最大）'] = f"{num_to_pct(mean_fix_zf)} / {num_to_pct(max_fix_zf)}"
    results.loc[0, '修复时间（均/最大）'] = (
        f"{round(np.log10(1 + mean_fix_zf) / np.log10(1 + annual_return) * 365, 1)} / "
        f"{round(np.log10(1 + max_fix_zf) / np.log10(1 + annual_return) * 365, 1)}")
    # ===统计每个周期
    results.loc[0, '盈利周期数'] = len(equity.loc[equity[pct_col] > 0])  # 盈利笔数
    results.loc[0, '亏损周期数'] = len(equity.loc[equity[pct_col] <= 0])  # 亏损笔数
    results.loc[0, '胜率'] = num_to_pct(results.loc[0, '盈利周期数'] / len(equity))  # 胜率
    results.loc[0, '每周期平均收益'] = num_to_pct(equity[pct_col].mean())  # 每笔交易平均盈亏
    results.loc[0, '盈亏收益比'] = round(equity.loc[equity[pct_col] > 0][pct_col].mean() / equity.loc[equity[pct_col] <= 0][pct_col].mean() * (-1), 2)  # 盈亏比
    if 1 in equity['是否爆仓'].to_list():
        results.loc[0, '盈亏收益比'] = 0
    results.loc[0, '单周期最大盈利'] = num_to_pct(equity[pct_col].max())  # 单笔最大盈利
    results.loc[0, '单周期大亏损'] = num_to_pct(equity[pct_col].min())  # 单笔最大亏损

    # ===连续盈利亏损
    results.loc[0, '最大连续盈利周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(equity[pct_col] > 0, 1, np.nan))])  # 最大连续盈利次数
    results.loc[0, '最大连续亏损周期数'] = max(
        [len(list(v)) for k, v in itertools.groupby(np.where(equity[pct_col] <= 0, 1, np.nan))])  # 最大连续亏损次数

    # ===其他评价指标
    results.loc[0, '收益率标准差'] = num_to_pct(equity[pct_col].std())

    # ===每年、每月收益率
    temp = equity.copy()
    temp.set_index('candle_begin_time', inplace=True)
    year_return = temp[[pct_col]].resample(rule='A').apply(lambda x: (1 + x).prod() - 1)
    month_return = temp[[pct_col]].resample(rule='M').apply(lambda x: (1 + x).prod() - 1)
    quarter_return = temp[[pct_col]].resample(rule='Q').apply(lambda x: (1 + x).prod() - 1)

    def num2pct(x):
        if str(x) != 'nan':
            return str(round(x * 100, 2)) + '%'
        else:
            return x

    year_return['涨跌幅'] = year_return[pct_col].apply(num2pct)
    month_return['涨跌幅'] = month_return[pct_col].apply(num2pct)
    quarter_return['涨跌幅'] = quarter_return[pct_col].apply(num2pct)

    # # 对每月收益进行处理，做成二维表
    # month_return.reset_index(inplace=True)
    # month_return['year'] = month_return['candle_begin_time'].dt.year
    # month_return['month'] = month_return['candle_begin_time'].dt.month
    # month_return.set_index(['year', 'month'], inplace=True)
    # del month_return['candle_begin_time']
    # month_return_all = month_return[pct_col].unstack()
    # month_return_all.loc['mean'] = month_return_all.mean(axis=0)
    # month_return_all = month_return_all.apply(lambda x: x.apply(num2pct))

    return results.T, year_return, month_return, quarter_return
