"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
from enum import Enum


class AccountType(Enum):
    PORTFOLIO_MARGIN = 'portfolio_margin'
    STANDARD = 'standard'

    def __str__(self):
        match self.value:
            case 'portfolio_margin':
                return '统一账户'
            case 'standard':
                return '普通账户'

    @classmethod
    def translate(cls, value):
        match value:
            case '统一账户':
                return cls.PORTFOLIO_MARGIN
            case '普通账户':
                return cls.STANDARD
            case _:
                print('目前仅支持以下账户类型：["统一账户", "普通账户"]')
                raise ValueError(f'{value} 是暂不支持的账户类型')


# Account Position
# Account Overview
# Order
