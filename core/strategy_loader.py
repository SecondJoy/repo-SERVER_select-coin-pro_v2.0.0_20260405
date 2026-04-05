#JoyAddded CTA币池因子趋势强度评估_20260329直播
"""
策略配置加载模块
支持从strategies文件夹中读取策略配置
"""
import os
import importlib.util
from pathlib import Path
from typing import List, Dict, Any


def load_strategy_from_file(strategy_file: str) -> List[Dict[str, Any]]:
    """
    从指定的策略文件中加载策略配置
    
    Args:
        strategy_file: 策略文件名（如 "动量f1001.py"）
        
    Returns:
        策略配置列表
    """
    # 获取strategies目录路径
    current_dir = Path(__file__).parent.parent
    strategies_dir = current_dir / 'strategies'
    
    # 构建完整文件路径
    file_path = strategies_dir / strategy_file
    
    # 检查文件是否存在
    if not file_path.exists():
        raise FileNotFoundError(f"策略文件不存在: {file_path}")
    
    # 动态导入模块
    spec = importlib.util.spec_from_file_location("strategy_module", file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载策略文件: {file_path}")
    
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)
    
    # 获取策略列表
    if hasattr(strategy_module, 'strategy_list'):
        return strategy_module.strategy_list
    else:
        raise AttributeError(f"策略文件中未找到 'strategy_list' 变量: {file_path}")


def load_strategy_by_backtest_name(backtest_name: str) -> List[Dict[str, Any]]:
    """
    根据回测名称加载策略配置
    
    Args:
        backtest_name: 回测名称（如 "动量f1001"）
        
    Returns:
        策略配置列表
    """
    strategy_filename = f"{backtest_name}.py"
    return load_strategy_from_file(strategy_filename)


def list_strategy_files() -> List[str]:
    """
    列出strategies目录下的所有Python文件
    
    Returns:
        策略文件名列表
    """
    current_dir = Path(__file__).parent.parent
    strategies_dir = current_dir / 'strategies'
    
    if not strategies_dir.exists():
        return []
    
    strategy_files = []
    for file_path in strategies_dir.glob("*.py"):
        if file_path.name != "__init__.py":
            strategy_files.append(file_path.name)
    
    return strategy_files


# 示例用法
if __name__ == "__main__":
    # 列出所有策略文件
    files = list_strategy_files()
    print("可用的策略文件:", files)
    
    # 加载特定策略文件
    if "动量f1001.py" in files:
        strategies = load_strategy_from_file("动量f1001.py")
        print(f"加载的策略数量: {len(strategies)}")
        for i, strategy in enumerate(strategies[:3]):  # 只显示前3个
            print(f"策略 {i+1}: {strategy['strategy']}")