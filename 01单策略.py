#JoyAddded CTA币池因子趋势强度评估_20260329直播
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

# Monkey Patch: 替换 BacktestConfig 内部引用的 StrategyConfig 为 CTA 专用版本
import core.model.backtest_config
from cta_core.model.strategy_config import StrategyConfig as CTAStrategyConfig
core.model.backtest_config.StrategyConfig = CTAStrategyConfig

from cta_core.backtest import run_backtest, BacktestConfig, logger
from cta_core.version import version_prompt

# ====================================================================================================
# ** 脚本运行前配置 **
# 主要是解决各种各样奇怪的问题，确保脚本能在不同环境下正常运行
# ====================================================================================================
warnings.filterwarnings('ignore')  # 过滤掉所有的warnings，以避免不必要的警告信息干扰用户

# pandas相关的显示设置，优化控制台输出
pd.set_option('display.max_rows', 1000)  # 设置DataFrame显示的最大行数为1000
pd.set_option('expand_frame_repr', False)  # 当列数超过屏幕宽度时，不自动换行
pd.set_option('display.unicode.ambiguous_as_wide', True)  # 处理模糊宽度字符的显示，使列对齐更美观
pd.set_option('display.unicode.east_asian_width', True)  # 处理东亚字符宽度，确保在命令行输出时的表格对齐


if __name__ == '__main__':
    version_prompt()  # 输出当前版本信息
    logger.info(f'系统启动中，稍等...')

    # 从配置文件中初始化回测配置
    backtest_config = BacktestConfig.init_from_config()

    # 输出回测配置信息，便于确认当前设置
    backtest_config.info()

    # 执行回测，开始策略框架的运行
    run_backtest(backtest_config)
