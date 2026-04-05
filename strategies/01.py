#JoyAddded CTA币池因子趋势强度评估_20260329直播
strategy_list = []

# 策略1 - Mtm_cubic_v2
long_factor_name_list = ["VolumeMeanRatio", ]
short_factor_name_list = ["VolumeMeanRatio", ]
long_select_coin_num = 20  # 多1
short_select_coin_num = 20  # 空1
# long_range =  list(range(20, 22, 2))
# short_range = list(range(20, 22, 2))
long_range = [20]
short_range = [20]

strategy_list +=[
    {
        "strategy": f"Strategy_{factor_name}_L_{x}",
        "offset_list":[0],
        "hold_period": '24H',
        "is_use_spot": False,
        "market": "swap_swap",
        'cap_weight':1,
        'long_cap_weight': 1,
        'short_cap_weight': 0,
        'long_select_coin_num': long_select_coin_num,
        'short_select_coin_num': 0,

        "factor_list": [
            (factor_name, False, x, 1),
        ],
        "filter_list": [
            ('QuoteVolumeMean', 48, 'pct:<=0.5', False),
        ],
        "use_custom_func": False,
    }
    for x in long_range
    for factor_name in long_factor_name_list
]

strategy_list += [
    {
        "strategy": f"Strategy_{factor_name}_S_{x}",
        "offset_list": [0],
        "hold_period": '24H',
        "is_use_spot": False,
        "market": "swap_swap",
        'cap_weight': 1,
        'long_cap_weight': 0,
        'short_cap_weight': 1,
        'long_select_coin_num':0 ,
        'short_select_coin_num':short_select_coin_num,
        "factor_list": [
            (factor_name, False, x, 1),
        ],
        "filter_list": [
            ('QuoteVolumeMean', 48, 'pct:<=0.5', False),

        ],
        "use_custom_func": False,
    }
    for x in short_range
    for factor_name in short_factor_name_list
]
