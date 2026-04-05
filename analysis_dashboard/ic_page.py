#JoyAdded 马代代码
# # Python
import dash
import pandas as pd
from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from mytools.func import cal_next
from mytools.models import get_ohlc_data
from mytools.streamlist_tools import list_all_factors
import tools.utils.tfunctions as tf
from mytools.xbx import draw_bar_plotly, draw_line_plotly

dash.register_page(__name__, path="/ic", name="Hello")

# 顶层 layout（或 layout() 函数都可以）
all_ready_factors = [{'label': f, 'value': f} for f in list_all_factors()]
bool_option = [{'label': "是", 'value': True}, {'label': "否", 'value': False}]
layout = html.Div([
    html.H1("IC分析"),
    dbc.Row([
        dbc.Col([
            html.Label("因子"),
            dcc.Dropdown(
                id='ic_factor',
                options=all_ready_factors,
                placeholder='因子',
                style={'width': '100%'}
            )
        ], width=2),
        dbc.Col([
            html.Label("参数"),
            dcc.Input(id="ic_parameter", type='number', style={'width': '100%'})
        ], width=2),
        dbc.Col([
            html.Label("持币时间(小时)"),
            dcc.Input(id="ic_hold_period", type='number', value=1, style={'width': '100%'})
        ], width=2),
        dbc.Col([
            html.Label("现货"),
            dcc.Dropdown(
                id='ic_use_spot',
                options=bool_option,
                value=True,
                style={'width': '100%'}
            )
        ], width=2),
        dbc.Col([
            html.Label("组数"),
            dcc.Input(id="ic_group_num", type='number', value=10, style={'width': '100%'})
        ], width=2)
    ]),
    html.Hr(),
    dbc.Button("提交分析", id="submit-analysis", color="primary", className="mt-3"),
    html.Hr(),
    dcc.Loading(
        id="ic-plot-loading",
        type="circle",  # 可选："default", "circle", "dot", "graph"
        children=html.Div(id="ic-plot-loading-finished")
    ),
    dbc.Row([

        dcc.Graph(id='ic-plot'),
        dcc.Graph(id='ic-group-plot'),
        dcc.Graph(id='ic-group_curve_plot')
    ], id="ic-plots-group", style={'display': 'none'}),

])


def compose_ic_content(IC):
    """
    整合各个offset的IC数据并计算相关的IC指标
    :param IC_list: 各个offset的IC数据
    :return:返回IC数据、IC字符串
    """

    # 将各个offset的数据合并 并 整理
    IC = IC.sort_values('candle_begin_time').reset_index(drop=True)

    # 计算累计RankIC；注意：因为我们考虑了每个offset，所以这边为了使得各个不同period之间的IC累计值能够比较，故除以offset的数量
    # ===计算IC的统计值，并进行约等
    # =IC均值
    IC_mean = IC['RankIC'].mean()
    # =IC标准差
    IC_std = IC['RankIC'].std()
    # =ICIR
    ICIR = IC_mean / IC_std
    # =IC胜率
    # 如果累计IC为正，则计算IC为正的比例
    if IC['RankIC_cumsum'].iloc[-1] > 0:
        IC_ratio = f"{(IC['RankIC'] > 0).sum() / len(IC) * 100 * 100 :.2f}%"
    # 如果累计IC为负，则计算IC为负的比例
    else:
        IC_ratio = f"{(IC['RankIC'] < 0).sum() / len(IC) * 100 :.2f}%"

    # 将上述指标合成一个字符串，加入到IC图中
    return f'IC均值：{IC_mean:6f}，IC标准差：{IC_std:6f}，ICIR：{ICIR:6f}，IC胜率：{IC_ratio}'





def draw_ic_plot(df, data_dict, date_col=None, right_axis=None, pic_size=[1500, 800], chg=False,
                 title=None):
    """
    绘制策略曲线
    :param df: 包含净值数据的df
    :param data_dict: 要展示的数据字典格式：｛图片上显示的名字:df中的列名｝
    :param date_col: 时间列的名字，如果为None将用索引作为时间列
    :param right_axis: 右轴数据 ｛图片上显示的名字:df中的列名｝
    :param pic_size: 图片的尺寸
    :param chg: datadict中的数据是否为涨跌幅，True表示涨跌幅，False表示净值
    :param title: 标题
    :param path: 图片路径
    :param show: 是否打开图片
    :return:
    """
    draw_df = df.copy()

    # 设置时间序列
    if date_col:
        time_data = draw_df[date_col]
    else:
        time_data = draw_df.index

    # 绘制左轴数据
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for key in list(data_dict.keys()):
        if chg:
            draw_df[data_dict[key]] = (draw_df[data_dict[key]] + 1).fillna(1).cumprod()
        if '回撤曲线' in key:
            fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                     #  marker=dict(color='rgba(220, 220, 220, 0.8)'),
                                     opacity=0.1, line=dict(width=0),
                                     fill='tozeroy',
                                     yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴
        else:
            fig.add_trace(go.Bar(x=time_data, y=draw_df[data_dict[key]], name=key,
                                 # marker_color='orange',  # 设置颜色
                                 # marker_line_color='orange'  # 设置柱状图边框的颜色
                                 ))

    # 绘制右轴数据
    if right_axis:
        for key in list(right_axis.keys()):
            if '回撤曲线' in key:
                fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                         #  marker=dict(color='rgba(220, 220, 220, 0.8)'),
                                         opacity=0.1, line=dict(width=0),
                                         fill='tozeroy',
                                         yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴
            else:
                fig.add_trace(go.Scatter(x=time_data, y=draw_df[right_axis[key]], name=key + '(右轴)',
                                         #  marker=dict(color='rgba(220, 220, 220, 0.8)'),
                                         marker_color='orange',  # 设置颜色
                                         yaxis='y2'))  # 标明设置一个不同于trace1的一个坐标轴

    fig.update_layout(template="none", width=pic_size[0], height=pic_size[1], title_text=title,
                      hovermode="x unified", hoverlabel=dict(bgcolor='rgba(255,255,255,0.5)', ),
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
    # plot(figure_or_data=fig, filename=path, auto_open=False)

    fig.update_yaxes(
        showspikes=True, spikemode='across', spikesnap='cursor', spikedash='solid', spikethickness=1,  # 峰线
    )
    fig.update_xaxes(
        showspikes=True, spikemode='across+marker', spikesnap='cursor', spikedash='solid', spikethickness=1,  # 峰线
    )
    return fig


@callback(
    [
        Output("ic-plot", "figure"),
        Output("ic-group-plot", "figure"),
        Output("ic-group_curve_plot", "figure"),
        Output("ic-plots-group", "style"),
        Output('ic-plot-loading-finished', 'children'),
    ],
    [
        Input("submit-analysis", "n_clicks"),
    ],
    [
        State("ic_factor", "value"),
        State("ic_parameter", "value"),
        State("ic_hold_period", "value"),
        State("ic_use_spot", "value"),
        State("ic_group_num", "value"),

    ]
)
def plot_ci(n_clicks, factor, parameter, hold_period, use_spot, group_num):
    if not n_clicks:
        return None, None, None, {'display': 'none'}, "未计算"
    factor_list = [(factor, parameter)]

    ohlc_data = get_ohlc_data(use_spot)
    factors = ohlc_data.calculate_factors(factor_list, use_spots=use_spot)

    factors = factors.groupby("symbol").apply(lambda group: cal_next(group, hold_period)).reset_index(drop=True)
    # factors[factors['symbol'] == "BTC-USDT"].to_csv("BTC.csv", index=False, encoding="utf-8-sig")
    ic_factor_name = f"{factor}_{parameter}"
    factors.dropna(subset=['ret_next'], inplace=True)
    kline = factors[factors['candle_begin_time'] >= '2021-01-01']
    corr = kline.groupby('candle_begin_time')[[ic_factor_name, 'ret_next']].corr(method='spearman')
    corr = corr[corr['ret_next'] != 1]['ret_next']
    corr = corr.reset_index().drop('level_1', axis=1).rename(columns={'ret_next': 'RankIC'})
    corr['RankIC_cumsum'] = corr['RankIC'].cumsum()
    left_axis = {
        'RankIC': 'RankIC',
    }
    right_axis = {
        '累计RankIC': 'RankIC_cumsum'
    }
    content = compose_ic_content(corr)
    ic_plot = draw_ic_plot(corr, data_dict=left_axis, date_col='candle_begin_time', right_axis=right_axis)
    # Process the inputs here

    group_curve, bar_df, labels = tf.group_analysis(factors, ic_factor_name, bins=group_num, method="pct")
    if not use_spot:
        labels += ['多空净值']
    bar_df = bar_df[bar_df['groups'].isin(labels)]
    # 构建因子值标识列表
    factor_labels = ['因子值最小'] + [''] * (group_num - 2) + ['因子值最大']
    if not use_spot:
        factor_labels.append('')
    bar_df['因子值标识'] = factor_labels
    group_fig = draw_bar_plotly(x=bar_df['groups'], y=bar_df['asset'], text_data=bar_df['因子值标识'],
                                title='分组净值')
    group_curve = group_curve.resample('D').last()
    cols_list = [col for col in group_curve.columns if '第' in col]
    y2_data = group_curve[['多空净值']] if not use_spot else pd.DataFrame()
    group_curve_fig = draw_line_plotly(x=group_curve.index, y1=group_curve[cols_list], y2=y2_data, if_log=True,
                                       title='分组资金曲线')
    ic_plot.write_html("ic_plot.html")
    group_curve_fig.write_html("group_curve_fig.html")
    group_fig.write_html("group_fig.html")
    return ic_plot, group_fig, group_curve_fig, {'display': 'block'}, content
