"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
import time

import numba as nb
import numpy as np
import pandas as pd

from config import swap_path
from core.evaluate import strategy_evaluate
from core.figure import draw_equity_curve_plotly
from core.model.backtest_config import BacktestConfig
from core.simulator import Simulator
from core.utils.functions import load_min_qty
from core.utils.log_kit import logger
from update_min_qty import min_qty_path

pd.set_option('display.max_rows', 1000)
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行


def calc_equity(conf: BacktestConfig,
                pivot_dict_spot: dict,
                pivot_dict_swap: dict,
                df_spot_ratio: pd.DataFrame,
                df_swap_ratio: pd.DataFrame,
                leverage: float | pd.Series = None):
    """
    计算回测结果的函数
    :param conf: 回测配置
    :param pivot_dict_spot: 现货行情数据
    :param pivot_dict_swap: 永续合约行情数据
    :param df_spot_ratio: 现货目标资金占比
    :param df_swap_ratio: 永续合约目标资金占比
    :param leverage: 杠杆
    :return: 没有返回值
    """
    # ====================================================================================================
    # 1. 数据预检和准备数据
    # 数据预检，对齐所有数据的长度（防御性编程）
    # ====================================================================================================
    # noinspection PyUnresolvedReferences
    logger.debug('🛂 数据预检，对齐所有数据的长度（防御性编程）')
    if len(df_spot_ratio) != len(df_swap_ratio) or np.any(df_swap_ratio.index != df_spot_ratio.index):
        raise RuntimeError(f'数据长度不一致，现货数据长度：{len(df_spot_ratio)}, 永续合约数据长度：{len(df_swap_ratio)}')

    # 开始时间列
    candle_begin_times = df_spot_ratio.index.to_series().reset_index(drop=True)

    # 获取现货和永续合约的币种，并且排序
    spot_symbols = sorted(df_spot_ratio.columns)
    swap_symbols = sorted(df_swap_ratio.columns)

    # 裁切现货数据，保证open，close，vwap1m，对应的df中，现货币种、时间长度一致
    pivot_dict_spot = align_pivot_dimensions(pivot_dict_spot, spot_symbols, candle_begin_times)

    # 裁切合约数据，保证open，close，vwap1m，funding_fee对应的df中，合约币种、时间长度一致
    pivot_dict_swap = align_pivot_dimensions(pivot_dict_swap, swap_symbols, candle_begin_times)

    # 读入最小下单量数据
    spot_lot_sizes = read_lot_sizes(min_qty_path / '最小下单量_spot.csv', spot_symbols)
    swap_lot_sizes = read_lot_sizes(min_qty_path / '最小下单量_swap.csv', swap_symbols)

    pos_calc = conf.rebalance_mode.create(spot_lot_sizes.to_numpy(), swap_lot_sizes.to_numpy())

    # 确定rebalance接入的时间点
    if conf.is_day_period:
        require_rebalance = np.where(candle_begin_times.dt.hour == 23, 1, 0).astype(np.int8)
    else:
        require_rebalance = np.ones(len(candle_begin_times), dtype=np.int8)

    if leverage is None:
        leverage = conf.leverage

    if isinstance(leverage, pd.Series):
        leverages = leverage.to_numpy(dtype=np.float64)
    else:
        leverages = np.full(len(df_spot_ratio), leverage, dtype=np.float64)

    # ====================================================================================================
    # 2. 开始模拟交易
    # 开始策马奔腾啦 🐎
    # ====================================================================================================
    logger.debug('▶️ 开始模拟交易...')
    s_time = time.perf_counter()
    equities, turnovers, fees, funding_fees, margin_rates, long_pos_values, short_pos_values = start_simulation(
        init_capital=conf.initial_usdt,  # 初始资金，单位：USDT
        leverages=leverages,  # 杠杆
        spot_lot_sizes=spot_lot_sizes.to_numpy(),  # 现货最小下单量
        swap_lot_sizes=swap_lot_sizes.to_numpy(),  # 永续合约最小下单量
        spot_c_rate=conf.spot_c_rate,  # 现货杠杆率
        swap_c_rate=conf.swap_c_rate,  # 永续合约杠杆率
        spot_min_order_limit=float(conf.spot_min_order_limit),  # 现货最小下单金额
        swap_min_order_limit=float(conf.swap_min_order_limit),  # 永续合约最小下单金额
        min_margin_rate=conf.margin_rate,  # 最低保证金比例
        # 选股结果计算聚合得到的每个周期目标资金占比
        spot_ratio=df_spot_ratio[spot_symbols].to_numpy(),  # 现货目标资金占比
        swap_ratio=df_swap_ratio[swap_symbols].to_numpy(),  # 永续合约目标资金占比
        # 现货行情数据
        spot_open_p=pivot_dict_spot['open'].to_numpy(),  # 现货开盘价
        spot_close_p=pivot_dict_spot['close'].to_numpy(),  # 现货收盘价
        spot_vwap1m_p=pivot_dict_spot['vwap1m'].to_numpy(),  # 现货开盘一分钟均价
        # 永续合约行情数据
        swap_open_p=pivot_dict_swap['open'].to_numpy(),  # 永续合约开盘价
        swap_close_p=pivot_dict_swap['close'].to_numpy(),  # 永续合约收盘价
        swap_vwap1m_p=pivot_dict_swap['vwap1m'].to_numpy(),  # 永续合约开盘一分钟均价
        funding_rates=pivot_dict_swap['funding_rate'].to_numpy(),  # 永续合约资金费率
        pos_calc=pos_calc,  # 仓位计算
        require_rebalance=require_rebalance,  # 是否需要rebalance
    )
    logger.ok(f'完成模拟交易，花费时间: {time.perf_counter() - s_time:.3f}秒')

    # ====================================================================================================
    # 3. 回测结果汇总，并输出相关文件
    # ====================================================================================================
    account_df = pd.DataFrame({
        'candle_begin_time': candle_begin_times,
        'equity': equities,
        'turnover': turnovers,
        'fee': fees,
        'funding_fee': funding_fees,
        'marginRatio': margin_rates,
        'long_pos_value': long_pos_values,
        'short_pos_value': short_pos_values
    })

    account_df['净值'] = account_df['equity'] / conf.initial_usdt
    account_df['涨跌幅'] = account_df['净值'].pct_change()
    account_df.loc[account_df['marginRatio'] < conf.margin_rate, '是否爆仓'] = 1
    account_df['是否爆仓'].fillna(method='ffill', inplace=True)
    account_df['是否爆仓'].fillna(value=0, inplace=True)
    account_df['long_short_ratio'] = account_df['long_pos_value'] / (account_df['short_pos_value'] + 1e-8)
    account_df['leverage_ratio'] = (account_df['long_pos_value'] + account_df['short_pos_value']) / account_df['equity']

    account_df.set_index('candle_begin_time', inplace=True)
    df_swap_ratio = df_swap_ratio * conf.leverage
    df_spot_ratio = df_spot_ratio * conf.leverage
    account_df['symbol_long_num'] = df_spot_ratio[df_spot_ratio > 0].count(axis=1) + df_swap_ratio[df_swap_ratio > 0].count(axis=1)
    account_df['symbol_short_num'] = df_spot_ratio[df_spot_ratio < 0].count(axis=1) + df_swap_ratio[df_swap_ratio < 0].count(axis=1)
    
    # 计算全局多头最大ratio和对应symbol（现货+合约多头部分）
    long_combined = pd.concat([
        df_spot_ratio.fillna(0).add_prefix('SPOT_'),
        df_swap_ratio[df_swap_ratio > 0].fillna(0).add_prefix('SWAP_')
    ], axis=1)
    account_df['long_max_ratio'] = long_combined.max(axis=1).fillna(0)
    # 转换为字符串类型避免Categorical错误
    account_df['long_max_ratio_symbol'] = 'NO_LONG' if long_combined.empty else long_combined.idxmax(axis=1).astype(str).fillna('NO_LONG')
    account_df.loc[account_df['long_max_ratio'] <= 0, 'long_max_ratio_symbol'] = 'NO_LONG'
    
    # 计算空头最大ratio（绝对值最大）和对应symbol
    short_only = df_swap_ratio[df_swap_ratio < 0].add_prefix('SWAP_')  # 添加SWAP_前缀保持一致
    account_df['short_max_ratio'] = short_only.min(axis=1).fillna(0)
    # 转换为字符串类型避免Categorical错误
    account_df['short_max_ratio_symbol'] = 'NO_SHORT' if short_only.empty else short_only.idxmin(axis=1).astype(str).fillna('NO_SHORT')
    account_df['short_max_ratio_abs'] = account_df['short_max_ratio'].abs()
    
    # 计算前3名持仓 - 简洁实现（百分比格式）
    # 多头前3名：利用pandas的nlargest，转换为百分比显示
    account_df['top3_long'] = long_combined.apply(
        lambda row: "; ".join([f"{sym}({val*100:.2f}%)" for sym, val in row[row > 0].nlargest(3).items()]) or "NO_LONG", 
        axis=1
    )
    
    # 空头前3名：按绝对值排序（已包含SWAP_前缀），转换为百分比显示
    account_df['top3_short'] = short_only.apply(
        lambda row: "; ".join([f"{sym}({val*100:.2f}%)" for sym, val in row[row < 0].abs().nlargest(3).items()]) or "NO_SHORT", 
        axis=1
    )
    
    # 如果需要重置索引，可以执行以下操作
    account_df.reset_index(inplace=True)

    # 策略评价
    rtn, year_return, month_return, quarter_return = strategy_evaluate(account_df, net_col='净值', pct_col='涨跌幅')
    conf.set_report(rtn.T)

    return account_df, rtn, year_return, month_return, quarter_return


def show_plot_performance(conf: BacktestConfig, account_df, rtn, year_return, title_prefix='', **kwargs):
    # 计算仓位比例
    account_df['long_pos_ratio'] = account_df['long_pos_value'] / account_df['equity']
    account_df['short_pos_ratio'] = account_df['short_pos_value'] / account_df['equity']
    account_df['empty_ratio'] = (conf.leverage - account_df['long_pos_ratio'] - account_df['short_pos_ratio']).clip(lower=0)
    # 计算累计值，主要用于后面画图使用
    account_df['long_cum'] = account_df['long_pos_ratio']
    account_df['short_cum'] = account_df['long_pos_ratio'] + account_df['short_pos_ratio']
    account_df['empty_cum'] = conf.leverage  # 空仓占比始终为 1（顶部）

    all_swap = pd.read_pickle(swap_path)
    btc_df = all_swap['BTC-USDT']
    account_df = pd.merge(left=account_df,
                          right=btc_df[['candle_begin_time', 'close']],
                          on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['BTC涨跌幅'] = account_df['close'].pct_change()
    account_df['BTC涨跌幅'].fillna(value=0, inplace=True)
    account_df['BTC资金曲线'] = (account_df['BTC涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['BTC涨跌幅']

    logger.debug(f"""
策略评价
{rtn}
分年收益率
{year_return}
总手续费
{account_df['fee'].sum():,.2f}
""")

    eth_df = all_swap['ETH-USDT']
    account_df = pd.merge(left=account_df,
                          right=eth_df[['candle_begin_time', 'close']],
                          on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['ETH涨跌幅'] = account_df['close'].pct_change()
    account_df['ETH涨跌幅'].fillna(value=0, inplace=True)
    account_df['ETH资金曲线'] = (account_df['ETH涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['ETH涨跌幅']

    # 生成画图数据字典，可以画出所有offset资金曲线以及各个offset资金曲线
    data_dict = {'多空资金曲线': '净值'}
    for col_name, col_series in kwargs.items():
        account_df[col_name] = col_series
        data_dict[col_name] = col_name
    data_dict.update({'BTC资金曲线': 'BTC资金曲线', 'ETH资金曲线': 'ETH资金曲线'})
    right_axis = {'多空最大回撤': '净值dd2here'}

    # 如果画多头、空头资金曲线，同时也会画上回撤曲线
    pic_title = f"{title_prefix}CumNetVal:{rtn.at['累积净值', 0]}, Annual:{rtn.at['年化收益', 0]}, MaxDrawdown:{rtn.at['最大回撤', 0]}"
    pic_desc = conf.get_fullname()
    # 调用画图函数
    draw_equity_curve_plotly(account_df,
                             data_dict=data_dict,
                             date_col='candle_begin_time',
                             right_axis=right_axis,
                             title=pic_title,
                             desc=pic_desc,
                             path=conf.get_result_folder() / f'{title_prefix}资金曲线.html',
                             show_subplots=True)


def read_lot_sizes(path, symbols):
    """
    读取每个币种的最小下单量
    :param path: 文件路径
    :param symbols:  币种列表
    :return:
    """
    default_min_qty, min_qty_dict = load_min_qty(path)
    lot_sizes = 0.1 ** pd.Series(min_qty_dict)
    lot_sizes = lot_sizes.reindex(symbols, fill_value=0.1 ** default_min_qty)
    return lot_sizes


def align_pivot_dimensions(market_pivot_dict, symbols, candle_begin_times):
    """
    对不同维度的数据进行对齐
    :param market_pivot_dict: 原始数据，是一个dict哦
    :param symbols: 币种（列）
    :param candle_begin_times: 时间（行）
    :return:
    """
    return {k: df.loc[candle_begin_times, symbols] for k, df in market_pivot_dict.items()}


@nb.jit(nopython=True, boundscheck=True)
def start_simulation(init_capital, leverages, spot_lot_sizes, swap_lot_sizes, spot_c_rate, swap_c_rate,
                     spot_min_order_limit, swap_min_order_limit, min_margin_rate, spot_ratio, swap_ratio,
                     spot_open_p, spot_close_p, spot_vwap1m_p, swap_open_p, swap_close_p, swap_vwap1m_p,
                     funding_rates, pos_calc, require_rebalance):
    """
    模拟交易
    :param init_capital: 初始资金
    :param leverages: 杠杆
    :param spot_lot_sizes: spot 现货的最小下单量
    :param swap_lot_sizes: swap 合约的最小下单量
    :param spot_c_rate: spot 现货的手续费率
    :param swap_c_rate: swap 合约的手续费率
    :param spot_min_order_limit: spot 现货最小下单金额
    :param swap_min_order_limit: swap 合约最小下单金额
    :param min_margin_rate: 维持保证金率
    :param spot_ratio: spot 的仓位透视表 (numpy 矩阵)
    :param swap_ratio: swap 的仓位透视表 (numpy 矩阵)
    :param spot_open_p: spot 的开仓价格透视表 (numpy 矩阵)
    :param spot_close_p: spot 的平仓价格透视表 (numpy 矩阵)
    :param spot_vwap1m_p: spot 的 vwap1m 价格透视表 (numpy 矩阵)
    :param swap_open_p: swap 的开仓价格透视表 (numpy 矩阵)
    :param swap_close_p: swap 的平仓价格透视表 (numpy 矩阵)
    :param swap_vwap1m_p: swap 的 vwap1m 价格透视表 (numpy 矩阵)
    :param funding_rates: swap 的 funding rate 透视表 (numpy 矩阵)
    :param pos_calc: 仓位计算
    :param require_rebalance: 是否需要调仓
    :return:
    """
    # ====================================================================================================
    # 1. 初始化回测空间
    # 设置几个固定长度的数组变量，并且重置为0，到时候每一个周期的数据，都按照index的顺序，依次填充进去
    # ====================================================================================================
    n_bars = spot_ratio.shape[0]
    n_syms_spot = spot_ratio.shape[1]
    n_syms_swap = swap_ratio.shape[1]

    start_lots_spot = np.zeros(n_syms_spot, dtype=np.int64)
    start_lots_swap = np.zeros(n_syms_swap, dtype=np.int64)
    # 现货不设置资金费
    funding_rates_spot = np.zeros(n_syms_spot, dtype=np.float64)

    turnovers = np.zeros(n_bars, dtype=np.float64)
    fees = np.zeros(n_bars, dtype=np.float64)
    equities = np.zeros(n_bars, dtype=np.float64)  # equity after execution
    funding_fees = np.zeros(n_bars, dtype=np.float64)
    margin_rates = np.zeros(n_bars, dtype=np.float64)
    long_pos_values = np.zeros(n_bars, dtype=np.float64)
    short_pos_values = np.zeros(n_bars, dtype=np.float64)

    # ====================================================================================================
    # 2. 初始化模拟对象
    # ====================================================================================================
    sim_spot = Simulator(init_capital, spot_lot_sizes, spot_c_rate, start_lots_spot, spot_min_order_limit)
    sim_swap = Simulator(0, swap_lot_sizes, swap_c_rate, start_lots_swap, swap_min_order_limit)

    # ====================================================================================================
    # 3. 开始回测
    # 每次循环包含以下四个步骤：
    # 1. 模拟开盘on_open
    # 2. 模拟执行on_execution
    # 3. 模拟平仓on_close
    # 4. 设置目标仓位set_target_lots
    # 如下依次执行
    # t1: on_open -> on_execution -> on_close -> set_target_lots
    # t2: on_open -> on_execution -> on_close -> set_target_lots
    # t3: on_open -> on_execution -> on_close -> set_target_lots
    # ...
    # tN: on_open -> on_execution -> on_close -> set_target_lots
    # 并且在每一个t时刻，都会记录账户的截面数据，包括equity，funding_fee，margin_rate，等等
    # ====================================================================================================
    #
    for i in range(n_bars):
        """1. 模拟开盘on_open"""
        # 根据开盘价格，计算账户权益，当前持仓的名义价值，以及资金费
        equity_spot, _, pos_value_spot = sim_spot.on_open(spot_open_p[i], funding_rates_spot, spot_open_p[i])
        equity_swap, funding_fee, pos_value_swap = sim_swap.on_open(swap_open_p[i], funding_rates[i], swap_open_p[i])

        # 当前持仓的名义价值
        position_val = np.sum(np.abs(pos_value_spot)) + np.sum(np.abs(pos_value_swap))
        if position_val < 1e-8:
            # 没有持仓
            margin_rate = 10000.0
        else:
            margin_rate = (equity_spot + equity_swap) / float(position_val)

        # 当前保证金率小于维持保证金率，爆仓 💀
        if margin_rate < min_margin_rate:
            margin_rates[i] = margin_rate
            break

        """2. 模拟开仓on_execution"""
        # 根据开仓价格，计算账户权益，换手，手续费
        equity_spot, turnover_spot, fee_spot = sim_spot.on_execution(spot_vwap1m_p[i])
        equity_swap, turnover_swap, fee_swap = sim_swap.on_execution(swap_vwap1m_p[i])

        """3. 模拟K线结束on_close"""
        # 根据收盘价格，计算账户权益
        equity_spot_close, pos_value_spot_close = sim_spot.on_close(spot_close_p[i])
        equity_swap_close, pos_value_swap_close = sim_swap.on_close(swap_close_p[i])

        long_pos_value = (np.sum(pos_value_spot_close[pos_value_spot_close > 0]) + 
                          np.sum(pos_value_swap_close[pos_value_swap_close > 0]))
        
        short_pos_value = -(np.sum(pos_value_spot_close[pos_value_spot_close < 0]) + 
                            np.sum(pos_value_swap_close[pos_value_swap_close < 0]))
        
        # 把中间结果更新到之前初始化的空间
        funding_fees[i] = funding_fee
        equities[i] = equity_spot_close + equity_swap_close
        turnovers[i] = turnover_spot + turnover_swap
        fees[i] = fee_spot + fee_swap
        margin_rates[i] = margin_rate
        long_pos_values[i] = long_pos_value
        short_pos_values[i] = short_pos_value

        # 考虑杠杆
        equity_leveraged = (equity_spot_close + equity_swap_close) * leverages[i]

        """4. 计算目标持仓"""
        # 并不是所有的时间点都需要计算目标持仓，比如D持仓下，只需要在23点更新0点的目标持仓
        if require_rebalance[i] == 1:
            target_lots_spot, target_lots_swap = pos_calc.calc_lots(equity_leveraged, spot_close_p[i], sim_spot.lots,
                                                                    spot_ratio[i], swap_close_p[i], sim_swap.lots,
                                                                    swap_ratio[i])
            # 更新目标持仓
            sim_spot.set_target_lots(target_lots_spot)
            sim_swap.set_target_lots(target_lots_swap)

    return equities, turnovers, fees, funding_fees, margin_rates, long_pos_values, short_pos_values
