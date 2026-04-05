#JoyAdded 马代代码
from pathlib import Path
from typing import List

import pandas as pd

from config import backtest_path, backtest_iter_path
from core.utils.path_kit import get_folder_path


def list_all_factors() -> List[str]:
    """
    TODO: 加入截面因子
    """
    factors_folder = Path(get_folder_path("factors"))
    section_factors_folder = Path(get_folder_path("sections"))
    return [f.stem for f in factors_folder.glob("*.py") if f.stem != "__init__"] + [f.stem for f in
                                                                               section_factors_folder.glob("*.py") if
                                                                               f.stem != "__init__"]


def list_all_strategy() -> List[str]:
    return [f.stem for f in backtest_path.iterdir() if f.is_dir()]


def list_all_search() -> List[str]:
    return [f.stem for f in backtest_iter_path.iterdir() if f.is_dir()]


def rank_top_n_value(source: pd.DataFrame, col_name: str, ascending: bool, top_n: int,
                     group_by_col: str = 'candle_begin_time') -> pd.DataFrame:
    if ascending:
        res = source.groupby(group_by_col).apply(lambda x: x.nsmallest(top_n, col_name).iloc[-1], include_groups=False)
    else:
        res = source.groupby(group_by_col).apply(lambda x: x.nlargest(top_n, col_name).iloc[-1], include_groups=False)
    return res.reset_index(level=0)


def _get_small_percent_min(group, percent, col_name):
    n = max(int(len(group) * percent), 1)  # 计算前10%的数量
    return group.nsmallest(n, col_name) if n > 0 else pd.DataFrame()  # 获取前10%的行


def _get_large_percent_min(group, percent, col_name):
    n = max(int(len(group) * percent), 1)  # 计算前10%的数量
    return group.nlargest(n, col_name) if n > 0 else pd.DataFrame()  # 获取前10%的行


def rank_top_n_pct(source: pd.DataFrame, col_name: str, ascending: bool, percent: float,
                   group_by_col: str = 'candle_begin_time') -> pd.DataFrame:
    if ascending:
        res = source.groupby(group_by_col).apply(lambda x: _get_small_percent_min(x, percent, col_name).iloc[-1],
                                                 include_groups=False)
    else:
        res = source.groupby(group_by_col).apply(lambda x: _get_large_percent_min(x, percent, col_name).iloc[-1],
                                                 include_groups=False)
    return res.reset_index(level=0)


def custom_fig_for_display(plot_fig, pic_size: List[int] = None):
    if pic_size is None:
        pic_size = [800, 600]
    plot_fig.update_layout(
        plot_bgcolor='rgba(173, 216, 230, 0.3)',  # 透明淡蓝色（lightblue with 30% opacity）
        paper_bgcolor='#f0f0f0',  # 图表外部背景
        width=pic_size[0],  # 增加宽度（单位：像素）
        height=pic_size[1],  # 增加高度（单位：像素）
        font_color='black',  # 全局字体颜色设置为黑色
        legend=dict(
            font=dict(color='black'),  # 明确设置图例文字为黑色
            title_font=dict(color='black')  # 图例标题（如果有）为黑色
        )
    )
    # 设置坐标轴文字颜色为黑色
    plot_fig.update_xaxes(
        title_font_color='black',
        tickfont_color='black'
    )
    plot_fig.update_yaxes(
        title_font_color='black',
        tickfont_color='black'
    )
    return plot_fig
