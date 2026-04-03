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

sys_version = '2.0.0'
sys_name = 'select-coin-pro'
build_version = f'v{sys_version}.20260206'


def version_prompt():
    show_versions()
    divider('[SYSTEM INFO]', '-', with_timestamp=False)
    logger.debug(f'# VERSION:\t{sys_name}({sys_version})')
    logger.debug(f'# BUILD:\t{build_version}')
    divider('[SYSTEM INFO]', '-', with_timestamp=False)
