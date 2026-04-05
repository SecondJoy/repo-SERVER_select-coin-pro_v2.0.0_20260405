#JoyAdded 马代代码
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly import subplots
from plotly.subplots import make_subplots
from config import swap_path
from core.evaluate import strategy_evaluate
from core.model.backtest_config import BacktestConfig
import plotly.express as px
from plotly.colors import qualitative as qcolor

from tools.utils.pfunctions import float_num_process
from plotly.colors import qualitative as qcolor  # 如果上面已导入过就不用重复


def draw_equity_curve_plotly_xbx(df, data_dict, date_col=None, right_axis=None, pic_size=None, chg=False, title=None,
                                 desc=None, show_subplots=False, right_is_line: bool = False):
    """
    绘制策略曲线
    :param right_is_line:
    :param df: 包含净值数据的df
    :param data_dict: 要展示的数据字典格式：｛图片上显示的名字:df中的列名｝
    :param date_col: 时间列的名字，如果为None将用索引作为时间列
    :param right_axis: 右轴数据 ｛图片上显示的名字:df中的列名｝
    :param pic_size: 图片的尺寸
    :param chg: datadict中的数据是否为涨跌幅，True表示涨跌幅，False表示净值
    :param title: 标题
    :param desc: 图片描述
    :param show_subplots: 是否展示子图，显示多空仓位比例
    :return:
    """
    if pic_size is None:
        pic_size = [1500, 920]

    draw_df = df.copy()

    # 设置时间序列
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index

    # 创建子图
    row = 3 if show_subplots else 1
    row_heights = [0.8, 0.1, 0.1] if show_subplots else [1]
    specs = [[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]] if show_subplots else [
        [{"secondary_y": True}]]
    fig = make_subplots(
        rows=row, cols=1,
        shared_xaxes=True,  # 共享 x 轴，主，子图共同变化
        vertical_spacing=0.03,  # 减少主图和子图之间的间距
        row_heights=row_heights,  # 主图高度占 80%，子图高度占 10%
        specs=specs
    )

    # 主图：绘制左轴数据
    for key in data_dict:
        if chg:
            draw_df[data_dict[key]] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        fig.add_trace(go.Scatter(x=time_data, y=draw_df[data_dict[key]], name=key), row=1, col=1)

    # 绘制右轴数据
    if right_axis:
        key = list(right_axis.keys())[0]
        if right_is_line is True:
            fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                     yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴
        else:
            fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                     marker=dict(color='rgba(220, 220, 220, 0.8)'),
                                     # marker_color='orange',
                                     opacity=0.1,
                                     line=dict(width=0),
                                     fill='tozeroy',
                                     yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴
        for i, key in enumerate(list(right_axis.keys())[1:]):
            fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                     marker=dict(color=f'rgba({100 + i * 50}, {100 + i * 20}, {100 + i * 30}, 0.8)'),
                                     opacity=0.2, line=dict(width=0),
                                     fill='tozeroy',
                                     yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴

    if show_subplots:
        # 子图：按照 matplotlib stackplot 风格实现堆叠图
        # 最下面是多头仓位占比
        fig.add_trace(go.Scatter(
            x=time_data,
            y=draw_df['long_cum'],
            mode='lines',
            line=dict(width=0),
            fill='tozeroy',
            fillcolor='rgba(30, 177, 0, 0.6)',
            name='多头仓位占比',
            hovertemplate="多头仓位占比: %{customdata:.4f}<extra></extra>",
            customdata=draw_df['long_pos_ratio']  # 使用原始比例值
        ), row=2, col=1)

        # 中间是空头仓位占比
        fig.add_trace(go.Scatter(
            x=time_data,
            y=draw_df['short_cum'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(255, 99, 77, 0.6)',
            name='空头仓位占比',
            hovertemplate="空头仓位占比: %{customdata:.4f}<extra></extra>",
            customdata=draw_df['short_pos_ratio']  # 使用原始比例值
        ), row=2, col=1)

        # 最上面是空仓占比
        fig.add_trace(go.Scatter(
            x=time_data,
            y=draw_df['empty_cum'],
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 46, 77, 0.6)',
            name='空仓占比',
            hovertemplate="空仓占比: %{customdata:.4f}<extra></extra>",
            customdata=draw_df['empty_ratio']  # 使用原始比例值
        ), row=2, col=1)

        # 子图：右轴绘制 long_short_ratio 曲线
        fig.add_trace(go.Scatter(
            x=time_data,
            y=draw_df['symbol_long_num'],
            name='多头选币数量',
            mode='lines',
            line=dict(color='rgba(30, 177, 0, 0.6)', width=2)
        ), row=3, col=1)

        fig.add_trace(go.Scatter(
            x=time_data,
            y=draw_df['symbol_short_num'],
            name='空头选币数量',
            mode='lines',
            line=dict(color='rgba(255, 99, 77, 0.6)', width=2)
        ), row=3, col=1)

    fig.update_layout(template="none", width=pic_size[0], height=pic_size[1], title_text=title,
                      hovermode="x unified", hoverlabel=dict(bgcolor='rgba(255,255,255,0.5)', ),
                      annotations=[
                          dict(
                              text=desc,
                              xref='paper',
                              yref='paper',
                              x=0.5,
                              y=1.05,
                              showarrow=False,
                              font=dict(size=12, color='black'),
                              align='center',
                              bgcolor='rgba(255,255,255,0.8)',
                          )
                      ]
                      )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label="线性 y轴",
                         method="relayout",
                         args=[{"yaxis.type": "linear"}]),
                    dict(label="Log y轴",
                         method="relayout",
                         args=[{"yaxis.type": "log"}]),
                ])],
    )

    fig.update_yaxes(
        showspikes=True, spikemode='across', spikesnap='cursor', spikedash='solid', spikethickness=1,  # 峰线
    )
    fig.update_xaxes(
        showspikes=True, spikemode='across+marker', spikesnap='cursor', spikedash='solid', spikethickness=1,  # 峰线
    )
    return fig


def show_market( account_df,  factor_colum: dict[str,str],
                debug: bool = False, **kwargs, ):
    all_swap = pd.read_pickle(swap_path)
    btc_df = all_swap['BTC-USDT']
    account_df = pd.merge(left=account_df,
                          right=btc_df[['candle_begin_time', 'close']],
                          on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['BTC涨跌幅'] = account_df['close'].pct_change()
    account_df['BTC涨跌幅'].fillna(value=0, inplace=True)
    account_df['BTC资金曲线'] = (account_df['BTC涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['BTC涨跌幅']



    data_dict = factor_colum
    right_axis = {'BTC资金曲线': 'BTC资金曲线'}

    # 如果画多头、空头资金曲线，同时也会画上回撤曲线
    pic_title = f"市场{factor_colum}图"
    pic_desc = "市场图"
    # 调用画图函数
    return draw_equity_curve_plotly_xbx(account_df,
                                        data_dict=data_dict,
                                        date_col='candle_begin_time',
                                        right_axis=right_axis,
                                        title=pic_title,
                                        desc=pic_desc,
                                        pic_size=[1000, 600],
                                        show_subplots=False,
                                        right_is_line=True)


def show_plot_performance_with_factor(conf: BacktestConfig, account_df, rtn, factor_colum: str, title_prefix='',
                                      debug: bool = False, **kwargs, ):
    account_df['long_pos_ratio'] = account_df['long_pos_value'] / account_df['equity']
    account_df['short_pos_ratio'] = account_df['short_pos_value'] / account_df['equity']
    account_df['empty_ratio'] = (conf.leverage - account_df['long_pos_ratio'] - account_df['short_pos_ratio']).clip(
        lower=0)
    # 计算累计值，主要用于后面画图使用
    account_df['long_cum'] = account_df['long_pos_ratio']
    account_df['short_cum'] = account_df['long_pos_ratio'] + account_df['short_pos_ratio']
    account_df['empty_cum'] = conf.leverage  # 空仓占比始终为 1（顶部）

    all_swap = pd.read_pickle(swap_path)
    btc_df = all_swap['BTC-USDT']
    account_df = pd.merge(left=account_df,
                          right=btc_df[['candle_begin_time', 'close']],
                          on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['BTC涨跌幅'] = account_df['close'].pct_change()
    account_df['BTC涨跌幅'].fillna(value=0, inplace=True)
    account_df['BTC资金曲线'] = (account_df['BTC涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['BTC涨跌幅']

    account_df['涨跌幅'].fillna(value=0, inplace=True)
    account_df['净值'] = (account_df['涨跌幅'] + 1).cumprod()
    rtn, _, _, _ = strategy_evaluate(account_df, net_col='净值', pct_col='涨跌幅')

    data_dict = {'多空资金曲线': '净值'}
    data_dict.update({'BTC资金曲线': 'BTC资金曲线'})
    right_axis = {'因子值': factor_colum}

    # 如果画多头、空头资金曲线，同时也会画上回撤曲线
    pic_title = f"{title_prefix}CumNetVal:{rtn.at['累积净值', 0]}, Annual:{rtn.at['年化收益', 0]}, MaxDrawdown:{rtn.at['最大回撤', 0]}"
    pic_desc = conf.get_fullname()
    # 调用画图函数
    return draw_equity_curve_plotly_xbx(account_df,
                                        data_dict=data_dict,
                                        date_col='candle_begin_time',
                                        right_axis=right_axis,
                                        title=pic_title,
                                        desc=pic_desc,
                                        pic_size=[1000, 600],
                                        show_subplots=True,
                                        right_is_line=True)


def show_plot_performance(conf: BacktestConfig, account_df, rtn=None, title_prefix='', debug: bool = False, **kwargs):
    # 计算仓位比例
    account_df['long_pos_ratio'] = account_df['long_pos_value'] / account_df['equity']
    account_df['short_pos_ratio'] = account_df['short_pos_value'] / account_df['equity']
    account_df['empty_ratio'] = (conf.leverage - account_df['long_pos_ratio'] - account_df['short_pos_ratio']).clip(
        lower=0)
    # 计算累计值，主要用于后面画图使用
    account_df['long_cum'] = account_df['long_pos_ratio']
    account_df['short_cum'] = account_df['long_pos_ratio'] + account_df['short_pos_ratio']
    account_df['empty_cum'] = conf.leverage  # 空仓占比始终为 1（顶部）

    all_swap = pd.read_pickle(swap_path)
    btc_df = all_swap['BTC-USDT']
    account_df = pd.merge(left=account_df,
                          right=btc_df[['candle_begin_time', 'close']],
                          on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['BTC涨跌幅'] = account_df['close'].pct_change()
    account_df['BTC涨跌幅'].fillna(value=0, inplace=True)
    account_df['BTC资金曲线'] = (account_df['BTC涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['BTC涨跌幅']

    eth_df = all_swap['ETH-USDT']
    account_df = pd.merge(left=account_df,
                          right=eth_df[['candle_begin_time', 'close']],
                          on=['candle_begin_time'],
                          how='left')
    account_df['close'].fillna(method='ffill', inplace=True)
    account_df['ETH涨跌幅'] = account_df['close'].pct_change()
    account_df['ETH涨跌幅'].fillna(value=0, inplace=True)
    account_df['ETH资金曲线'] = (account_df['ETH涨跌幅'] + 1).cumprod()
    del account_df['close'], account_df['ETH涨跌幅']

    if rtn is None:
        account_df['涨跌幅'].fillna(value=0, inplace=True)
        account_df['净值'] = (account_df['涨跌幅'] + 1).cumprod()
        rtn, _, _, _ = strategy_evaluate(account_df, net_col='净值', pct_col='涨跌幅')

    # 生成画图数据字典，可以画出所有offset资金曲线以及各个offset资金曲线
    data_dict = {'多空资金曲线': '净值'}
    for col_name, col_series in kwargs.items():
        account_df[col_name] = col_series
        data_dict[col_name] = col_name
    data_dict.update({'BTC资金曲线': 'BTC资金曲线', 'ETH资金曲线': 'ETH资金曲线'})
    right_axis = {'多空最大回撤': '净值dd2here'}

    # 如果画多头、空头资金曲线，同时也会画上回撤曲线
    pic_title = f"{title_prefix}CumNetVal:{rtn.at['累积净值', 0]}, Annual:{rtn.at['年化收益', 0]}, MaxDrawdown:{rtn.at['最大回撤', 0]}"
    pic_desc = conf.get_fullname()
    # 调用画图函数
    return draw_equity_curve_plotly_xbx(account_df,
                                        data_dict=data_dict,
                                        date_col='candle_begin_time',
                                        right_axis=right_axis,
                                        title=pic_title,
                                        desc=pic_desc,
                                        show_subplots=True)


def draw_params_bar_plotly(draw_df: pd.DataFrame, name: str):
    rows = len(draw_df.columns)
    s = (1 / (rows - 1)) * 0.5
    fig = subplots.make_subplots(rows=rows, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=s)
    # 提取 index 的数字部分并排序，只保留数字作为 index
    import re
    def extract_number(idx):
        match = re.search(r'(\d+)$', str(idx))
        return int(match.group(1)) if match else float('inf')
    # 生成新 index
    sorted_idx = sorted(draw_df.index, key=extract_number)
    new_idx = [str(extract_number(idx)) for idx in sorted_idx]
    draw_df_sorted = draw_df.loc[sorted_idx].copy()
    draw_df_sorted.index = new_idx
    for i, col_name in enumerate(draw_df_sorted.columns):
        trace = go.Bar(x=draw_df_sorted.index, y=draw_df_sorted[col_name], name=f"{col_name}")
        fig.add_trace(trace, i + 1, 1)
        fig.update_xaxes(showticklabels=True, row=i + 1, col=1)
    for i, col_name in enumerate(draw_df_sorted.columns):
        fig.update_xaxes(title_text=col_name, row=i + 1, col=1)
    fig.update_layout(height=200 * rows, showlegend=True, title_text=name)
    return fig


def draw_params_heatmap_plotly(draw_df: pd.DataFrame, title: str = None):
    """
    生成热力图
    """
    x, y = draw_df.shape
    if x > y:
        draw_df = draw_df.T

    draw_df.replace(np.nan, '', inplace=True)
    # 修改temp的index和columns为str
    draw_df.index = draw_df.index.astype(str)
    draw_df.columns = draw_df.columns.astype(str)
    return px.imshow(
        draw_df,
        title=title,
        text_auto="",
        aspect="auto",
        color_continuous_scale='Turbo',
        # color_continuous_scale=[ 'blue','green','orange','yellow','red'],
        zmin=draw_df.min().min(),
        zmax=draw_df.max().max()
    )

def draw_bar_plotly(x, y, text_data=None, title='', pic_size=[1800, 600]):
    """
    柱状图画图函数
    :param x: 放到X轴上的数据
    :param y: 放到Y轴上的数据
    :param text_data: text说明数据
    :param title: 图标题
    :param pic_size: 图大小
    :return:
        返回柱状图
    """

    # 创建子图
    fig = make_subplots()

    y_ = y.map(float_num_process, na_action='ignore')

    if text_data is not None:
        text_values = [
            f"{x_val}<br>{text_val}"  # <br>实现换行显示
            for x_val, text_val in zip(x, text_data)
        ]
    else:
        # 仅显示数值（带千分位格式）
        text_values = [f"{x_val}" for x_val in x]

    # 添加柱状图轨迹（统一蓝色）
    fig.add_trace(go.Bar(
        x=x,  # X轴数据
        y=y,  # Y轴数据
        text=y_,  # Y轴文本
        name=x.name if hasattr(x, "name") else "value",
        marker=dict(color="#1f77b4")  # 统一设置为蓝色
    ), row=1, col=1)

    # 更新X轴的tick
    fig.update_xaxes(
        tickmode='array',
        tickvals=x,
        ticktext=text_values,
    )

    # 更新布局
    fig.update_layout(
        plot_bgcolor='rgb(255, 255, 255)',  # 设置绘图区背景色
        width=pic_size[0],  # 宽度
        height=pic_size[1],  # 高度
        title={
            'text': title,  # 标题文本
            'x': 0.377,  # 标题相对于绘图区的水平位置
            'y': 0.9,  # 标题相对于绘图区的垂直位置
            'xanchor': 'center',  # 标题的水平对齐方式
            'font': {'color': 'green', 'size': 20}  # 标题的颜色和大小
        },
        xaxis=dict(domain=[0.0, 0.73]),  # 设置 X 轴的显示范围
        showlegend=True,  # 是否显示图例
        legend=dict(
            x=0.8,  # 图例相对于绘图区的水平位置
            y=1.0,  # 图例相对于绘图区的垂直位置
            bgcolor='white',  # 图例背景色
            bordercolor='gray',  # 图例边框颜色
            borderwidth=1  # 图例边框宽度
        )
    )
    return fig

def draw_line_plotly(x, y1, y2=pd.DataFrame(), update_xticks=False, if_log=False, title='', pic_size=[1400, 420]):
    """
    折线画图函数
    :param x: X轴数据
    :param y1: 左轴数据
    :param y2: 右轴数据
    :param update_xticks: 是否更新x轴刻度
    :param if_log: 是否需要log轴
    :param title: 图标题
    :param pic_size: 图片大小
    :return:
        返回折线图
    """

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    # 颜色方案：显式上色，避免黑白
    palette = (qcolor.Plotly + qcolor.D3 + qcolor.Set2 + qcolor.Set3 + qcolor.Dark24 + qcolor.Light24)

    for i, col in enumerate(y1.columns):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y1[col],
                name=col,
                line={'width': 2, 'color': palette[i % len(palette)]},
                mode="lines"
            ),
            row=1, col=1, secondary_y=False
        )

    if not y2.empty:
        for j, col in enumerate(y2.columns):
            color_idx = (len(palette) // 2 + j) % len(palette)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y2[col],
                    name=col,
                    line={'dash': 'dot', 'width': 2, 'color': palette[color_idx]},
                    mode="lines"
                ),
                row=1, col=1, secondary_y=True
            )

    if update_xticks:
        fig.update_xaxes(tickmode='array', tickvals=x)

    fig.update_layout(
        template="plotly_white",
        plot_bgcolor='rgb(255, 255, 255)',
        width=pic_size[0],
        height=pic_size[1],
        title={
            'text': f'{title}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'green', 'size': 20}
        },
        # 将图例放到上方、横向排列，并预留顶部空间
        legend=dict(
            orientation='h',        # 横向图例
            yanchor='bottom',
            y=1.02,                 # 放在绘图区上方
            xanchor='left',
            x=0.0,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        ),
        margin=dict(t=90, r=20, b=40, l=60),  # 顶部留白，避免图例压线
        hovermode="x unified",
        hoverlabel=dict(bgcolor='rgba(255,255,255,0.5)')
    )

    return fig
