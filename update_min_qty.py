"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
from pathlib import Path

import ccxt
import pandas as pd

from core.utils.path_kit import get_folder_path

min_qty_path = Path(get_folder_path('data', 'min_qty'))


def update(proxies=None):
    # 初始化交易所
    if proxies is None:
        proxies = {}
    exchange = ccxt.binance({'proxies': proxies})

    for _ in ['swap', 'spot']:
        # 获取交易规则
        if _ == 'swap':
            data = exchange.fapiPublicGetExchangeInfo()
        else:
            data = exchange.publicGetExchangeInfo()

        # 获取BUSD和USDT的交易对
        _symbol_list = [x for x in data['symbols'] if x['symbol'].endswith('BUSD') or x['symbol'].endswith('USDT')]

        # 获取需要的最小下单量数据
        min_qty_list = []
        for symbol in _symbol_list:
            min_qty_list.append({
                '币种': symbol['symbol'].replace('USDT', '-USDT'),
                '最小下单量': symbol['filters'][1]['minQty']
            })

        # 转成df
        new_df = pd.DataFrame(min_qty_list)

        # 文件路径
        file_path = min_qty_path / f'最小下单量_{_}.csv'

        # 读取旧的数据
        if file_path.exists():
            old_df = pd.read_csv(file_path, encoding='gbk')
        else:
            old_df = pd.DataFrame()

        # 数据合并
        all_data_df = pd.concat([new_df, old_df], ignore_index=True)
        # 去重
        all_data_df.drop_duplicates(subset=['币种'], inplace=True)
        all_data_df.to_csv(file_path, encoding='gbk', index=False)

        print(all_data_df)
        print(f'【{_}】最小下单量更新完成')
        print('-' * 32)
        print('\n')


if __name__ == '__main__':
    update()

    # 如果你本地开了Clash的代理，可以这样更新
    # update({'http': 'http://127.0.0.1:1087', 'https': 'http://127.0.0.1:1087'})
