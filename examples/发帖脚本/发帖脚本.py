# -*- coding: utf-8 -*-
"""
中性策略框架 | 邢不行 | 2024分享会
author: 邢不行
微信: xbx1717
"""
import os
import pandas as pd
from config import backtest_path

backtest_name = '混合策略仓位管理'
backtest_file_path = os.path.join(backtest_path, backtest_name)

try:
    equity_path = os.path.join(backtest_file_path, '资金曲线.csv')
    equity_df = pd.read_csv(equity_path, encoding='utf-8-sig', parse_dates=['candle_begin_time'])
    equity_df = equity_df[['candle_begin_time', '涨跌幅']]

    evaluate_path = os.path.join(backtest_file_path, '策略评价.csv')
    evaluate_df = pd.read_csv(evaluate_path, encoding='utf-8-sig')
except:
    print('=' * 50)
    print('请确定是否存在对应文件')
    print('=' * 50)
    import traceback
    traceback.print_exc()
    exit()

pro_max = equity_df.sort_values('涨跌幅', ascending=False).head(5)
pro_min = equity_df.sort_values('涨跌幅', ascending=True).head(5)
pro_max['涨跌幅'] = pro_max['涨跌幅'].apply(lambda x: str(round(100 * x, 2)) + '%')
pro_min['涨跌幅'] = pro_min['涨跌幅'].apply(lambda x: str(round(100 * x, 2)) + '%')

# 低版本的pandas，不支持 index 参数
# 安装高版本 pandas 命令 ： pip install pandas==1.5.3
tx_evaluate = evaluate_df.to_markdown()

tx_pro_max = pro_max.to_markdown(index=False)
tx_pro_min = pro_min.to_markdown(index=False)

with open('样本模板.txt', 'r', encoding='utf8') as file:
    bbs_post = file.read()
    bbs_post = bbs_post % (tx_evaluate, tx_pro_max, tx_pro_min)
    print(bbs_post)
