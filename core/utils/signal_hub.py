"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""
import importlib


class SignalHub:
    _signal_cache = {}

    # noinspection PyTypeChecker
    @classmethod
    def get_by_name(cls, signal_name):
        if signal_name in SignalHub._signal_cache:
            return SignalHub._signal_cache[signal_name]

        try:
            # 构造模块名
            module_name = f"signals.{signal_name}"

            # 动态导入模块
            signal_module = importlib.import_module(module_name)

            # 创建一个包含模块变量和函数的字典
            signal_content = {
                name: getattr(signal_module, name) for name in dir(signal_module)
                if not name.startswith("__")
            }

            # 创建一个包含这些变量和函数的对象
            signal_instance = type(signal_name, (), signal_content)
            signal_instance.module_name = module_name

            # 缓存策略对象
            cls._signal_cache[signal_name] = signal_instance

            return signal_instance
        except ModuleNotFoundError:
            raise ValueError(f"Signal {signal_name} not found.")
        except AttributeError:
            raise ValueError(f"Error accessing signal content in module {signal_name}.")


# 使用示例
if __name__ == "__main__":
    factor = SignalHub.get_by_name("MovingAverage")
