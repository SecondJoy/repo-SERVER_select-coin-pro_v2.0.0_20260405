#JoyAddded CTA币池因子趋势强度评估_20260329直播
"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

from pandas import show_versions

from core.utils.log_kit import logger, divider

sys_version = '0.10'
sys_name = 'cta-select-coin'
build_version = f'v{sys_version}.20250930'


def version_prompt():
    show_versions()
    divider('[SYSTEM INFO]', '-', with_timestamp=False)
    logger.debug(f'# VERSION:\t{sys_name}({sys_version})')
    logger.debug(f'# BUILD:\t{build_version}')
    divider('[SYSTEM INFO]', '-', with_timestamp=False)
