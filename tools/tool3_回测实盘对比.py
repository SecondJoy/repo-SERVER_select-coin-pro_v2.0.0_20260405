# -*- coding: utf-8 -*-
"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

import os
import warnings

import numpy as np

import tools.utils.pfunctions as pf
import tools.utils.tfunctions as tf

warnings.filterwarnings('ignore')
_ = os.path.abspath(os.path.dirname(__file__))  # 返回当前文件路径
root_path = os.path.abspath(os.path.join(_, '..'))  # 返回根目录文件夹


# ====== 实盘和回测资金曲线对比 主函数 ======
def curve_analysis(cfg):
    print("开始进行资金曲线和选币结果对比分析...")

    # ====== 绘制实盘和回测资金曲线对比图 ======

    # 提取字段
    backtest_name = cfg['backtest_name']
    start_time = cfg['start_time']
    end_time = cfg['end_time']

    # 整合资金曲线数据
    print("正在整合资金曲线数据...")
    equity_data = tf.process_equity_data(root_path, backtest_name, start_time, end_time)

    left_axis = {
        '回测净值': '回测净值',
        '实盘净值': '实盘净值',
        '对比资金曲线': '对比资金曲线'
    }
    print("正在绘制资金曲线对比图...")
    fig_list = []
    fig = pf.draw_equity_curve_plotly(equity_data, left_axis, date_col='time', title='实盘回测资金曲线对比')
    fig_list.append(fig)

    # ====== 绘制实盘和回测选币结果对比图 ======
    print("正在整合选币数据...")
    # 整合选币数据
    # 注意点：由于回测和实盘选币的差异，后续计算相似度时，分别考虑基于回测和实盘的结果，即分别以回测和实盘选币数量作为相似度计算的分母
    coin_selection_data = tf.process_coin_selection_data(root_path, backtest_name, start_time, end_time)

    left_axis = {
        f'回测-{backtest_name}选币数量': f'回测-{backtest_name}选币数量',
        f'实盘-{backtest_name}选币数量': f'实盘-{backtest_name}选币数量',
        '重复选币数量': '重复选币数量',
    }

    similarity_mean = coin_selection_data[f'相似度'].mean().round(2)

    right_axis = {
        f'相似度(均值：{similarity_mean})': '相似度',
    }

    print("正在绘制选币结果对比图...")
    fig = pf.draw_coins_difference(coin_selection_data, left_axis, date_col='candle_begin_time', right_axis=right_axis,
                                   title='实盘回测选币结果对比'
                                   )
    fig_list.append(fig)

    # 实盘回测独有选币整理
    table_data = coin_selection_data[['candle_begin_time', '回测独有选币', '实盘独有选币']]
    table_data['candle_begin_time'] = (table_data['candle_begin_time'].map(
        lambda x: x.strftime('%Y-%m-%d %H:00:00')))
    table_data['回测独有选币'] = table_data['回测独有选币'].apply(
        lambda x: ', '.join(sorted(x)) if isinstance(x, set) else x
    )
    table_data['实盘独有选币'] = table_data['实盘独有选币'].apply(
        lambda x: ', '.join(sorted(x)) if isinstance(x, set) else x
    )
    table_data.replace('', np.nan, inplace=True)
    table_data = table_data.dropna(subset=['回测独有选币', '实盘独有选币'], how='all')
    table_data.replace(np.nan, '', inplace=True)
    fig = pf.draw_coins_table(table_data, table_data.columns, title='选币结果明细表')
    fig_list.append(fig)

    # ====== 生成最终报告 ======
    # 结果保存路径
    output_dir = os.path.join(root_path, 'data/分析结果/实盘回测对比', backtest_name)
    os.makedirs(output_dir, exist_ok=True)
    html_name = f'{backtest_name}实盘回测对比.html'
    pf.merge_html_flexible(fig_list, os.path.join(output_dir, html_name),
                           title=f'{backtest_name}实盘回测对比分析')

    print("资金曲线和选币结果对比分析全部完成。")
    print("""
实盘回测不一致的常见原因：
    1.分钟偏移的原因导致因子值不一样。可以先把小资金的实盘的分钟偏移调为0m，对比结束之后再调整为其他的分钟偏移
    2.回测/实盘数据有问题。重新下载一份回测数据，并且重新一下实盘数据，再做对比
    3.实盘回测策略配置不一样。检查选币因子(因子文件及配置)、过滤因子(因子文件及配置)、持仓周期、offset、选币数量等配置
    4.实盘K线数量不够。修改实盘所需的K线数量
    5.一些新币、下架币、拉黑币的问题。导致实盘和回测存在差异。
    """)

    # 绘制某个币的因子值
    while True:
        coin = input("请输入要查看因子的币种(例如 CATI )，退出输入 q 即可：")
        if coin == 'q':
            break
        coin = coin.strip().upper()
        if not coin.endswith('USDT'):
            coin += 'USDT'
        coin = coin.replace('-', '')

        # 读取所有回测因子
        backtest_factor_path = os.path.join(root_path, 'data/cache/')
        backtest_factors_name_list = [factor_name[7:-4] for factor_name in os.listdir(backtest_factor_path) if
                                      (factor_name.endswith('.pkl') & factor_name.startswith('factor_'))]
        # 读取所有实盘因子
        trading_factor_path = os.path.join(root_path, f'data/回测结果/{backtest_name}/实盘结果/runtime')
        trading_factors_name_list = [factor_name[12:-4] for factor_name in os.listdir(trading_factor_path) if
                                     (factor_name.endswith('.pkl') & factor_name.startswith('all_factors_'))]
        trading_factors_name_list.remove('kline')

        # 检查是否实盘或者回测缺少某个因子
        def find_missing_elements(list_a, list_b):
            set_a = set(list_a)
            set_b = set(list_b)
            a_missing = set_b - set_a  # list_a中缺少的元素
            b_missing = set_a - set_b  # list_b中缺少的元素
            return {'coins_list': list(set_a | set_b),
                    "list_a缺少的元素": list(a_missing),
                    "list_b缺少的元素": list(b_missing)
                    }

        result = find_missing_elements(backtest_factors_name_list, trading_factors_name_list)

        if result["list_a缺少的元素"]:
            raise ValueError(f"回测缺少因子：{result['list_a缺少的元素']}")
        if result["list_b缺少的元素"]:
            raise ValueError(f"实盘缺少因子：{result['list_b缺少的元素']}")

        factors_data = tf.process_backtest_trading_factors(root_path, result['coins_list'],
                                                           backtest_name, coin)
        if isinstance(factors_data, bool) and not factors_data:
            continue

        # 画图
        col_right_axis = [col for col in factors_data.columns if col not in ['实盘_close(左轴)', '回测_close(左轴)']]
        fig = pf.draw_line_plotly(x=factors_data.index, y1=factors_data[['实盘_close(左轴)', '回测_close(左轴)']],
                                  y2=factors_data[col_right_axis],
                                  if_log=True, title=f'实盘回测{coin}因子值对比')
        # 输出路径
        factor_output_dir = os.path.join(root_path, 'data/分析结果/实盘回测同币因子值对比')
        os.makedirs(factor_output_dir, exist_ok=True)

        html_name = f'实盘回测{coin}因子值对比.html'
        pf.merge_html_flexible([fig], os.path.join(factor_output_dir, html_name), title='')
        print(f"实盘回测{coin}因子值对比分析完成")


if __name__ == "__main__":
    # ====== 使用说明 ======
    "https://bbs.quantclass.cn/thread/54137"

    # ====== 配置信息 ======
    # 注意点：首先，需要将策略对应的实盘结果放置同一策略的回测目录下，展示一下目录结构
    # data / 回测结果 / 浪淘沙策略 /
    #                            │
    #                            ├── 资金曲线.csv(回测资金曲线结果)
    #                            │
    #                            │── 选币结果.pkl(回测选币结果)
    #                            │
    #                            ├── 实盘结果(文件夹)
    #                            │   ├── select_coin(选币文件夹)
    #                            │   │   └── 例如 2025-03-04_00.pkl
    #                            │   ├── runtime(因子文件夹)
    #                            │   │   └── 例如 all_factor_Cci_525.pkl
    #                            │   └── 账户信息 (实盘资金曲线文件夹)
    #                            │       └── equity.csv (实盘资金曲线结果)
    #                            │
    #                            └── 一些关于回测的其他文件，比如 年度账户收益.csv

    config = {
        # 注意点：需要保证回测的日期与实盘匹配，由于实盘是最新的，所有回测结果可能需要更新，即更新历史数据，重新回测。
        'start_time': '2025-03-19',  # 对比开始时间
        'end_time': '2025-03-20 23:00:00',  # 对比结束时间
        'backtest_name': '浪淘沙策略',  # 回测策略名称，用于读取 data\回测结果 目录下，该策略对应的结果
    }

    # 开始分析
    curve_analysis(config)
