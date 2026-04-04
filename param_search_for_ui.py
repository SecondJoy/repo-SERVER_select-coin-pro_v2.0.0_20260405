"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

import copy
import itertools
import json
import re
import sys
import warnings
from multiprocessing import freeze_support

import pandas as pd

from core.backtest import find_best_params
from core.model.backtest_config import BacktestConfigFactory
from core.utils.log_kit import logger
from core.utils.path_kit import get_file_path
from core.version import version_prompt

# ====================================================================================================
# ** 脚本运行前配置 **
# 主要是解决各种各样奇怪的问题们
# ====================================================================================================
# region 脚本运行前准备
warnings.filterwarnings("ignore")  # 过滤一下warnings，不要吓到老实人

# pandas相关的显示设置，基础课程都有介绍
pd.set_option("display.max_rows", 1000)
pd.set_option("expand_frame_repr", False)  # 当列太多时不换行
pd.set_option("display.unicode.ambiguous_as_wide", True)  # 设置命令行输出时的列对齐功能
pd.set_option("display.unicode.east_asian_width", True)


def dict_itertools(dict_):
    filter_dict = {k: v for k, v in dict_.items() if isinstance(v, list) and len(v) > 0}
    keys = list(filter_dict.keys())
    values = list(filter_dict.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def __parse_path_expression(path_expr):
    """解析路径表达式，如 'factor_list[0][2][0]'

    Args:
        path_expr: 路径表达式字符串

    Returns:
        tuple: (base_key, indices)
        - base_key: 基础键名，如 'factor_list'
        - indices: 索引列表，如 [0, 2, 0]
    """
    # 使用正则表达式匹配基础键名和所有索引
    match = re.match(r"^([^[]+)((?:\[\d+\])+)$", path_expr)
    if not match:
        return path_expr, []

    base_key = match.group(1)
    indices_str = match.group(2)

    # 提取所有数字索引
    indices = [int(idx) for idx in re.findall(r"\[(\d+)\]", indices_str)]

    return base_key, indices


def __set_nested_value(obj, base_key, indices, value):
    """根据路径设置嵌套数据结构中的值

    Args:
        obj: 目标对象（字典）
        base_key: 基础键名
        indices: 索引列表
        value: 要设置的值
    """
    if base_key not in obj:
        return

    current = obj[base_key]

    # 导航到最后一层的父级
    for idx in indices[:-1]:
        if isinstance(current, list) and 0 <= idx < len(current):
            current = current[idx]
        else:
            return

    # 设置最后一层的值
    final_idx = indices[-1]
    if 0 <= final_idx < len(current):
        if isinstance(current, (list, tuple)):
            current = list(current)
            current[final_idx] = value
            current = tuple(current)
        else:
            current[final_idx] = value

    obj[base_key][indices[:-1][0]] = current


def convert_range_params(data):
    """转换range格式的参数为列表

    Args:
        data: 配置数据（通常是字典）

    Returns:
        转换后的数据
    """
    if isinstance(data, dict):
        # 检查是否是range格式 {"start": x, "end": y, "step": z}
        if all(key in data for key in ["start", "end", "step"]):
            start = data["start"]
            end = data["end"]
            step = data["step"]
            return list(range(start, end, step))
        else:
            # 递归处理字典中的每个值
            return {k: convert_range_params(v) for k, v in data.items()}
    elif isinstance(data, list):
        # 递归处理列表中的每个元素
        return [convert_range_params(item) for item in data]
    else:
        # 其他类型直接返回
        return data


def convert_lists_to_tuples(data, target_fields=None):
    """将指定字段中的列表里的列表转换为元组

    Args:
        data: 字典数据
        target_fields: 需要处理的字段集合，默认为None时处理所有字段

    Returns:
        处理后的数据
    """
    if not isinstance(data, dict):
        return data

    # 默认的元组字段
    if target_fields is None:
        target_fields = {
            "factor_list",
            "long_factor_list",
            "short_factor_list",
            "filter_list",
            "long_filter_list",
            "short_filter_list",
            "filter_list_post",
            "long_filter_list_post",
            "short_filter_list_post",
        }

    # 深拷贝以避免修改原数据
    result = copy.deepcopy(data)

    for field in target_fields:
        if field in result and isinstance(result[field], list):
            result[field] = [
                tuple(item) if isinstance(item, list) else item
                for item in result[field]
            ]

    return result


if __name__ == "__main__":
    if "win" in sys.platform:
        freeze_support()
    version_prompt()
    logger.info(f"系统启动中，稍等...")

    # ====================================================================================================
    # 1. 配置需要遍历的参数
    # ====================================================================================================
    with open(get_file_path("config.json"), "r", encoding="utf-8") as f:
        batch = json.load(f)
    backtest_name = batch.get("search_name", "遍历")

    # 转换range格式的参数为列表
    batch = convert_range_params(batch)

    strategies = []
    stg = batch.get("strategy_info", {})
    # 转换列表中的列表为元组
    stg = convert_lists_to_tuples(stg)
    if stg:
        strategy_list = [stg]
    else:
        print(
            '未找到需要遍历的策略，重新在网页上编辑遍历试试。\n 问题排查:检查 config.json 中的 "strategy_info" 数据'
        )
        exit(1)
    for params_dict in dict_itertools(batch):
        # 使用config.py中的strategy_list作为模板，更新对应的参数
        current_strategy_list = copy.deepcopy(strategy_list)
        for strategy in current_strategy_list:
            # 更新可遍历的参数
            for param_key, param_value in params_dict.items():
                if not param_value:  # 跳过空值
                    continue

                # 检查是否是路径表达式
                base_key, indices = __parse_path_expression(param_key)

                if indices:  # 如果是路径表达式（包含索引）
                    __set_nested_value(strategy, base_key, indices, param_value)
                    logger.info(f"更新路径表达式 {param_key}: {param_value}")
                else:  # 传统的直接键值对
                    # 处理传统的参数更新逻辑
                    strategy[param_key] = param_value

        strategies.append(current_strategy_list)

    factory = BacktestConfigFactory(backtest_name=backtest_name)
    factory.generate_configs_by_strategies(strategies=strategies)

    # ====================================================================================================
    # 2. 执行遍历
    # ====================================================================================================
    find_best_params(factory)
