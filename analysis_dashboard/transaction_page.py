from typing import List

import dash
from dash import html, dcc, dash_table, callback, callback_context
import dash_bootstrap_components as dbc
from dash_table import FormatTemplate
from plotly.subplots import make_subplots

from core.utils.factor_hub import FactorHub
from mytools.func import reload_module_py, diff_transaction, cleanup_analysis_cache
from mytools.streamlist_tools import list_all_factors, list_all_strategy
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import ctx
from dash.dependencies import Input, Output, State

import config
from mytools.models import BackTestingResults, plotly_transactions, get_ohlc_data
from mytools.xbx import show_plot_performance

dash.register_page(__name__, path='/transaction', name='交易记录')


def create_transaction_table():
    return dash_table.DataTable(
        id='transaction-table',
        columns=[
            {'name': '币种', 'id': 'symbol'},
            {'name': '利润', 'id': 'profit', "type": "numeric", "format": FormatTemplate.percentage(2)},
            {'name': '开始时间', 'id': 'start_time'},
            {'name': '结束时间', 'id': 'end_time'},
            {'name': '方向', 'id': 'direction'},
            {'name': '偏移', 'id': 'offset'},
            {'name': '现货', 'id': 'is_spot'},
            {'name': '策略', 'id': 'strategy'}
        ],
        style_table={'height': '400px', 'overflowY': 'auto'},
        style_cell={
            'textAlign': 'left',
            'minWidth': '10px',
            'maxWidth': '100px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'color': 'black'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(230, 230, 230)'
            },
            {
                'if': {'filter_query': '{profit} contains "-"', 'column_id': 'profit'},
                'color': 'red'
            }
        ],
        row_selectable='single',  # 只允许选择一行
        selected_rows=[],  # 初始时没有选中的行
        page_size=12,  # 每页显示20条记录
        sort_action='native'  # 启用本地排序功能
    )


def create_drawback_table():
    return dash_table.DataTable(
        id='stg_drawback_table',
        columns=[
            {'name': '开始时间', 'id': 'start'},
            {'name': '谷底时间', 'id': 'valley'},
            {'name': '结束时间', 'id': 'end'},
            {'name': '持续小时', 'id': 'hours'},
            {'name': '回撤幅度', 'id': 'drawdown'}
        ],
        style_table={'height': '400px', 'overflowY': 'auto'},
        style_cell={
            'textAlign': 'left',
            'minWidth': '10px',
            'maxWidth': '100px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'color': 'black'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(230, 230, 230)'
            }
        ],
        row_selectable='single',  # 只允许选择一行
        selected_rows=[],  # 初始时没有选中的行
        page_size=12,  # 每页显示20条记录
        sort_action='native'  # 启用本地排序功能
    )


def create_distribution_table():
    return dash_table.DataTable(
        id='transaction-distribution-table',
        columns=[
            {'name': '区域', 'id': 'bin'},
            {'name': '做多', 'id': 'long'},
            {'name': '做空', 'id': 'short'},
        ],
        style_table={'height': '400px', 'overflowY': 'auto'},
        style_cell={
            'textAlign': 'left',
            'minWidth': '10px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'color': 'black'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        row_selectable='multi',  # 允许多选或不选
        selected_rows=[],  # 初始时没有选中的行
        sort_action='native'  # 启用本地排序功能
    )


def create_transaction_range_slider():
    return dcc.RangeSlider(
        id='transaction-range-slider',
        min=0,
        max=1,
        value=[0, 1],
        marks={},
        step=86400,
        tooltip={
            "placement": "top",
            "always_visible": True,
            "template": "时间点: %{value}"  # 使用template而不是value
        }

    )


def create_parameter_table():
    return dash_table.DataTable(
        id='parameter-display-table',
        columns=[
            {
                "name": "因子",
                "id": "factor",
                "type": "text",  # 使用 text 类型
            },
            {
                "name": "参数",
                "id": "parameter",
                "type": "numeric"
            },
        ],
        page_size=5,
        style_cell={'textAlign': 'left', 'padding': '5px'},
        style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
        style_table={
            'marginLeft': '10',
            'marginRight': 'auto',
            'overflowX': 'auto'
        },
        row_deletable=True,
    )


all_ready_factors = [{'label': f, 'value': f} for f in list_all_factors()]
# 可编辑表格页面布局
layout = html.Div([
    html.H1(id='strategy-display'),

    create_transaction_range_slider(),
    # 原来的 dbc.Row([...]) 替换为更紧凑的水平排列
    html.Div(
        [
            dbc.Button("资金曲线", id="transaction-equity-curve", n_clicks=0),
            dbc.Button("月度曲线", id="transaction-month-return", n_clicks=0),
            dbc.Button("回撤", id="open-drawback-button", n_clicks=0),
            dbc.Button("下载交易记录", id="download-transactions-btn", n_clicks=0),
            dcc.Download(id="download-transactions"),
        ],
        style={
            'display': 'flex',
            'alignItems': 'center',
            'gap': '8px',  # 按钮之间的间距（可调成 4px/6px）
            'marginTop': '12px',
            'marginBottom': '12px'
        }
    ),
    dcc.Store(id='stg-drawback-store'),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("回测区间")),
            dbc.ModalBody(create_drawback_table()),
        ],
        id="drawback-table-modal",
        size="xl",
        is_open=False,
    ),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("月收益")),
            dbc.ModalBody(dcc.Graph(id='transaction-month-return-plot'), ),
        ],
        id="transaction-month-return-model",
        size="xl",
        is_open=False,
    ),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("资金曲线")),
            dbc.ModalBody(dcc.Graph(id='transaction-equity-curve-plot'), ),
        ],
        id="transaction-equity-curve-model",
        size="xl",
        is_open=False,
    ),
    dbc.Row(
        [
            dbc.Label(id="transaction-statistic", children="交易记录统计"),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(dbc.Label(id="transaction-default-factor-label", children="默认因子"), md=2),
            dbc.Col(
                dcc.Dropdown(
                    id='transaction-default-factor',
                    options=all_ready_factors,
                    placeholder='因子',
                ), md=5),
            dbc.Col(
                dcc.Input(
                    id='transaction-default-factor-parameter',
                    type='number',
                    placeholder='Enter a number',
                ), md=3),
            dbc.Col([html.Button("刷新", id="refresh-default-factors-btn", n_clicks=0)], md=3)
        ], style={'marginBottom': '12px'}
    ),
    dbc.Row(
        [
            dbc.Col(create_distribution_table(), md=3),
            dbc.Col(create_transaction_table(), md=9),
        ], style={'marginTop': '12px'}
    ),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("资金曲线")),
            dbc.ModalBody(
                dcc.Loading(
                    id="transaction-stg-compare-loading",
                    type="circle",
                    children=dcc.Graph(id='transaction-stg-compare-median-plot'),
                )
            ),
        ],
        id="transaction-stg-compare-plot-model",
        size="xl",
        is_open=False,
    ),
    dbc.Row([
        dbc.Col([
            html.Label("杠杆"),
            dcc.Dropdown(
                id='transaction-other-strategy-dropdown',
                placeholder='策略',
                style={'width': '100%'},
                clearable=False,
            )
        ], md=3),
        dbc.Col([
            dbc.Button("中性比较", id="transaction-compare-stg-plot-curve-btn", color="primary", className="mt-3"),
        ], md=2),
        dbc.Col([
            html.Label("比较类型"),
            dcc.Dropdown(
                id='transaction-compare-stg—type-dropdown',
                options=[{'label': '相同买入', 'value': "same_buy"}, {'label': '完全不同', 'value': "diff"}],
                value="same_buy",
                placeholder='策略',
                style={'width': '100%'},
                clearable=False,
            ),
        ], md=3),
        dbc.Col([
            dbc.Button("交易比较", id="transaction-compare-transaction-button", color="primary", className="mt-3"),
        ], md=2),
    ]),
    html.Hr(),
    dcc.Loading(
        id="loading-plot-transaction",
        type="default",
        children=html.Div(id="loading-plot-transaction-finish")
    ),
    # 合并图表和价格按钮到一个div
    html.Div(
        id='transactions-plot-group',
        style={'display': 'none'},
        children=[
            html.Div([
                dcc.DatePickerRange(
                    id='transaction-hour-range-picker',
                    min_date_allowed=datetime.datetime(2018, 1, 1),
                    max_date_allowed=datetime.datetime.today(),
                    start_date=None,
                    end_date=None,
                    display_format='YYYY-MM-DD',
                    style={'marginRight': '16px'},
                ),
                dbc.Button('往前加一周', id='btn-prev-week', n_clicks=0, style={'marginRight': '4px'}),
                dbc.Button('往前加一天', id='btn-prev-day', n_clicks=0, style={'marginRight': '8px'}),
                dbc.Button('往后加一天', id='btn-next-day', n_clicks=0, style={'marginRight': '4px'}),
                dbc.Button('往后加一周', id='btn-next-week', n_clicks=0),
            ], id="transaction-timerange-div", style={'marginTop': '12px', 'marginBottom': '12px'}),
            dcc.Graph(id='transactions-plot', style={'height': '600px', 'width': '100%', 'margin': '0 auto'}),
            html.Div([
                dbc.Button("TradingView", id="open-tradingview-btn", n_clicks=0),
            ], style={'marginTop': '12px', 'marginBottom': '32px'}),
        ]
    ),
    html.Div(
        id='transactions-analysis-div',
        style={'marginTop': '12px', 'display': 'none'},
        children=dbc.Row(
            [
                dbc.Col([
                    html.Div([
                        html.Label("", id="peak-hour-label", style={'marginBottom': '10px'}),
                    ]),
                    dbc.Row(
                        [
                            dbc.Col([
                                dcc.Dropdown(
                                    id='new-factor',
                                    options=all_ready_factors,
                                    placeholder='因子',
                                    style={'display': 'inline-block', 'width': '100%'}
                                )], md=8),
                            dbc.Col([html.Button("刷新", id="refresh-factors-btn", n_clicks=0)], md=3)
                        ],

                    ),
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(
                                id='new-factor-parameter',
                                type='number',
                                placeholder='Enter a number',
                                style={'display': 'inline-block', 'width': '100%'}
                            )]),
                        dbc.Col([
                            dcc.Input(
                                id='n-hour-later',
                                type='number',
                                placeholder='延后小时',
                                value=0,
                                style={'display': 'inline-block', 'width': '100%'}
                            )]),
                    ]),
                    dbc.Row(
                        [
                            dbc.Col([html.Button('添加参数', id='add-parameter-button')]),
                            dbc.Col([html.Button('重画图形', id='plot-new-factor-button')]),
                            dbc.Col([html.Button('因子计算', id='calculate-new-button')])
                        ]
                    ),
                ], md=3),

                dbc.Col(create_parameter_table(), md=9),
                dbc.Modal(
                    [
                        dbc.ModalHeader(dbc.ModalTitle("Header")),
                        dbc.ModalBody([
                            dcc.Loading(
                                id="new-parameter-one-transaction-plot-loading",
                                type="circle",  # 可选："default", "circle", "dot", "graph"
                                children=html.Div(id="new-parameter-one-transaction-plot-finished")
                            ),
                            dcc.Graph(id='new-parameter-one-transaction-plot')
                        ]),
                    ],
                    id="transaction-with-new-parameter-modal",
                    size="xl",
                    is_open=False,
                ),
                html.Div(id='table-output', style={'marginTop': '20px'}),
                html.Hr(),
                dcc.Loading(
                    id="loading-new-parameter-loading",
                    type="circle",  # 可选："default", "circle", "dot", "graph"
                    children=html.Div(id="loading-new-parameter-finished")
                ),
                dcc.Graph(id='new-factor-plot'),
            ],
        )),
])


def plot_factor_data(factor_data, x_param: tuple[str, int], start_time, end_time,
                     tag: str, factor_side="last") -> pd.DataFrame:
    res = factor_data[(factor_data['candle_begin_time'] >= pd.to_datetime(start_time)) & (
            factor_data['candle_begin_time'] <= pd.to_datetime(end_time))]
    group_by_symbol = res.groupby('symbol').agg(
        x=(f"{x_param[0]}_{x_param[1]}", factor_side),
        close_start=('close', 'first'),
        close_end=('close', 'last'),
    )

    res: pd.DataFrame = group_by_symbol.reset_index()
    res['x'] = res["x"].rank(ascending=True)
    res['y'] = res['close_end'] / res['close_start'] - 1
    res.drop(columns=['close_start', 'close_end'], inplace=True)
    res["tag"] = tag
    res = res.sort_values('x')

    return res


def cal_factor(name, p, symbol, use_spot, trade_data) -> pd.Series:
    factor = f"{name}_{p}"
    return trade_data.cal_factor_one_symbol(symbol, [(name, p)], use_spot=use_spot)[['candle_begin_time', factor]]


def return_distribution(transactions):
    # 定义区间
    bins = [-float('inf')] + \
           [round(x, 1) for x in np.arange(-0.5, 0.5 + 0.1, 0.1)] + \
           [float('inf')]

    def count_profit(series):
        # 统计每个区间的个数
        counts = pd.cut(series, bins=bins, right=False).value_counts().sort_index()

        # 创建结果字典
        result = {}

        # 小于-0.5的部分
        result['< -0.5'] = counts.iloc[0]

        # -0.5到0.5之间的区间
        for i in range(1, len(counts) - 1):
            interval = f'[{bins[i]}, {bins[i + 1]})'
            result[interval] = counts.iloc[i]

        # 大于0.5的部分
        result['> 0.5'] = counts.iloc[-1]

        return result

    long_position = transactions[transactions['direction'] == 1]
    short_position = transactions[transactions['direction'] == -1]
    long_profit_bin = count_profit(long_position['profit'])
    short_profit_bin = count_profit(short_position['profit'])
    res = pd.DataFrame({"long": long_profit_bin, "short": short_profit_bin, }).reset_index()
    res = res.rename(columns={'index': 'bin'})
    return res


@callback(
    Output('transaction-range-slider', 'tooltip'),
    [Input('transaction-range-slider', 'value')]
)
def update_transaction_range_slider_tooltip(value):
    if not value:
        return {"always_visible": True, "placement": "top"}
    return {
        "always_visible": True,
        "placement": "top",
        "template": f"{pd.Timestamp.fromtimestamp(value[0]).strftime('%Y-%m-%d')} 到 {pd.Timestamp.fromtimestamp(value[1]).strftime('%Y-%m-%d')}"
    }


@callback(
    Output('new-factor', 'options'),
    Output('transaction-default-factor', 'options'),
    Output('transaction-default-factor', 'value'),
    Output('transaction-default-factor-parameter', 'value'),
    [
        Input('refresh-factors-btn', 'n_clicks'),
        Input('refresh-default-factors-btn', 'n_clicks'),
        Input('select-strategy-store', 'data'),
    ],
)
def refresh_factor_options(n_clicks, default_n_click, select_strategy):
    """
    触发点：点击刷新按钮或策略切换时。
    作用：尝试重载 mytools.factors 包下的子模块（如果存在），
    然后重新计算 list_all_factors() 并返回用于 dcc.Dropdown 的 options 列表。
    """
    try:

        reload_module_py("factors")
        reload_module_py("sections")
        results = BackTestingResults(select_strategy)
        stg_config = results.config.strategy_list[0]
        factor_list = stg_config.long_factor_list
        if factor_list is None:
            factor_list = stg_config.short_factor_list

        factor_config = factor_list[0]
        factor_name = factor_config.name
        factor_parameter = factor_config.param
        FactorHub._factor_cache.clear()
        # 重新读取因子列表
        new_factors = [{'label': f, 'value': f} for f in list_all_factors()]
        return new_factors, new_factors, factor_name, factor_parameter
    except Exception:
        return all_ready_factors if all_ready_factors else []


@callback(
    [
        Output('transaction-range-slider', 'min'),
        Output('transaction-range-slider', 'max'),
        Output('transaction-range-slider', 'marks'),
        Output('transaction-range-slider', 'value'),
        Output("stg_drawback_table", "selected_rows"),
        Output('drawback-table-modal', 'is_open'),
    ],
    [
        Input('transaction-store', 'data'),
        Input("stg_drawback_table", "selected_rows"),
        Input("stg_drawback_table", "data"),
        Input("open-drawback-button", "n_clicks")
    ],
    prevent_initial_call=True
)
def refresh_time_select_slider(transaction_data, selected_rows, draw_back_data, n_clicks):
    # 没有触发源（极少见）
    model_open = False

    prop_id = ctx.triggered[0]['prop_id']  # e.g. "drawback-list-button.n_clicks"
    trigger_id = prop_id.split('.')[0]

    if trigger_id == 'open-drawback-button':
        model_open = True

    if not transaction_data:
        return 0, 0, {}, [0, 1], [], model_open
    # 获取时间范围
    times = [pd.Timestamp(trans['start_time']) for trans in transaction_data]
    min_time = min(times)
    max_time = max(times)

    # 转换为时间戳
    min_ts = int(min_time.timestamp())
    max_ts = int(max_time.timestamp())

    # 生成月份标记
    dates = pd.date_range(start=min_time, end=max_time, freq='M')
    marks = {
        int(d.timestamp()): {
            'label': d.strftime('%Y-%m'),
            'style': {
                'font-size': '12px',
                'text-align': 'center'
            }
        }
        for d in dates[::3]  # 每三个月显示一次
    }
    if not selected_rows:
        return min_ts, max_ts, marks, [min_ts, max_ts], [], model_open
    else:
        selected_row = draw_back_data[selected_rows[0]]
        start_ = selected_row['start']
        start = int(pd.Timestamp(start_).timestamp())
        end = int(pd.Timestamp(selected_row['end']).timestamp())
        return min_ts, max_ts, marks, [start, end], [], model_open


@callback(
    Output('transaction-table', 'data'),
    Output('transaction-statistic', 'children'),
    [Input('transaction-distribution-table', 'selected_rows'),
     Input('transaction-distribution-table', 'data'),
     Input('transaction-store', 'data'),
     Input('transaction-range-slider', 'value')]
)
def filter_transactions(selected_rows, bin_data, transaction_data, time_range):
    start_time = pd.Timestamp.fromtimestamp(time_range[0])
    end_time = pd.Timestamp.fromtimestamp(time_range[1])

    if transaction_data is None:
        return None, None
    # Filter transactions by time range
    filtered_by_time = [trans for trans in transaction_data
                        if start_time <= pd.Timestamp(trans['start_time']) <= end_time]

    if not selected_rows or not bin_data:
        return filtered_by_time, cal_transaction_statistic(
            filtered_by_time)  # Return time-filtered data if no bins selected

    # 获取选中的区间
    selected_bins = [bin_data[i]['bin'] for i in selected_rows]
    # 创建过滤条件
    filtered_data = []
    for trans in filtered_by_time:
        profit = trans['profit']  # 转换百分比为小数
        if profit is None:  # 这个为最后的时间
            continue
        verify = False
        # 检查profit是否在任何选中的区间内
        for bin_range in selected_bins:
            if bin_range == '< -0.5' and profit < -0.5:
                verify = True
                break
            elif bin_range == '> 0.5' and profit > 0.5:
                verify = True
                break
            elif bin_range.startswith('['):
                # 处理类似 "[0.1, 0.2)" 的区间
                start, end = map(float, bin_range.strip('[]()').split(','))
                if start <= profit < end:
                    verify = True
                    break
        if verify:
            filtered_data.append(trans)
    return filtered_data, cal_transaction_statistic(filtered_data)


def cal_transaction_statistic(transaction: List) -> str:
    if len(transaction) == 0:
        return "当前无交易记录"
    durations = []
    for trans in transaction:
        durations.append(pd.Timestamp(trans['end_time']) - pd.Timestamp(trans['start_time']))
    duration_hour = np.array([(d.total_seconds() / 3600) for d in durations])
    avg = np.mean(duration_hour)
    median = np.median(duration_hour)
    max_v = np.max(duration_hour)
    min_v = np.min(duration_hour)
    var = np.var(duration_hour)
    result = (
        f"持仓时长统计：平均 {avg:.2f} 小时，"
        f"中位数 {median:.2f} 小时，"
        f"最大 {max_v:.2f} 小时，"
        f"最小 {min_v:.2f} 小时，"
        f"方差 {var:.2f}"
    )
    return result


@callback(
    Output('transaction-distribution-table', 'data'),
    Input('transaction-store', 'data'),
    Input('transaction-range-slider', 'value')
)
def update_transaction_distribution_table(data, time_range):
    start_time = pd.Timestamp.fromtimestamp(time_range[0])
    end_time = pd.Timestamp.fromtimestamp(time_range[1])

    if data is None:
        return None
    # Filter transactions by time range
    filtered_by_time = [trans for trans in data
                        if start_time <= pd.Timestamp(trans['start_time']) <= end_time]
    trans_df = pd.DataFrame(filtered_by_time)
    distribution = return_distribution(trans_df)
    return distribution.to_dict('records')


# 回调2：处理策略加载 - 正常加载逻辑
@callback(
    Output('transaction-store', 'data', allow_duplicate=True),
    Output('stg-drawback-store', 'data', allow_duplicate=True),
    Output('loading-strategy-finish', 'children'),
    Input('select-strategy-store', 'data'),
    prevent_initial_call=True
)
def load_transaction_data(select_strategy):
    """加载策略的交易数据（不比较）"""
    if select_strategy is None:
        return dash.no_update, dash.no_update, dash.no_update

    results = BackTestingResults(select_strategy)
    export_root = config.backtest_path / select_strategy
    transaction_root = export_root / "transactions"
    if not transaction_root.exists():
        transaction_root.mkdir(parents=True)
    cache_path = transaction_root / f"{select_strategy}.parquet"

    if cache_path.exists():
        trans_df = pd.read_parquet(cache_path)
        return trans_df.to_dict('records'), results.drawdown_list().to_dict(
            "records"), f"加载策略:{select_strategy}完成"

    trans_df = results.transaction_df(merge_trans=True)
    trans_df = trans_df[
        ['symbol', 'profit', 'start_time', 'end_time', 'start_close', 'end_close', 'direction', 'offset',
         'is_spot', 'strategy']]
    trans_df = trans_df.sort_values(by='profit')
    trans_df.to_parquet(cache_path)
    return trans_df.to_dict('records'), results.drawdown_list().to_dict(
        "records"), f"策略:{select_strategy}加载完成"


# 回调3：处理交易比较 - 比较逻辑
@callback(
    [
        Output('transaction-store', 'data', allow_duplicate=True),
        Output('loading-strategy-finish', 'children', allow_duplicate=True)
    ],
    Input('transaction-compare-transaction-button', 'n_clicks'),
    [
        State('select-strategy-store', 'data'),
        State('transaction-compare-stg—type-dropdown', 'value'),
        State('transaction-other-strategy-dropdown', 'value'),
    ],
    prevent_initial_call=True
)
def compare_transactions(trigger_data, select_strategy, compare_type, other_strategy):
    """比较两个策略的交易记录"""
    if select_strategy is None or not trigger_data:
        return dash.no_update, dash.no_update
    try:
        base_result = BackTestingResults(name=select_strategy)
        compare_result = BackTestingResults(name=other_strategy)
        trans_df = diff_transaction(
            base_result.transaction_df(use_cache=True),
            compare_result.transaction_df(use_cache=True),
            diff_type=compare_type,
        )
        trans_df.sort_values('start_time', inplace=True)
        return trans_df.to_dict('records'), f"策略:{select_strategy}与{other_strategy}比较完成"
    except Exception as e:
        print(f"比较交易时出错: {e}")
        return dash.no_update, dash.no_update, f"比较失败: {e}"


@callback(
    [
        Output('transactions-plot', 'figure'),
        Output('transactions-plot-group', 'style'),
        Output('loading-plot-transaction-finish', 'children'),
        Output('transactions-analysis-div', 'style'),
        Output('peak-hour-label', 'children'),
        Output("n-hour-later", "value"),
        Output('transactions-plot', 'config'),
    ],
    Input('transaction-table', 'selected_rows'),
    Input('transaction-table', 'data'),
    Input('select-strategy', 'value'),
    Input('transaction-hour-range-picker', 'start_date'),
    Input('transaction-hour-range-picker', 'end_date'),
    State('transaction-default-factor', 'value'),
    State('transaction-default-factor-parameter', 'value'),
    prevent_initial_call=True
)
def plot_transaction(selected_rows, all_transaction, strategy_name, picker_start, picker_end, default_factor,
                     default_factor_parameter):
    if not selected_rows:
        # Return empty figure when no rows are selected
        return px.scatter(title="请选择一条交易记录进行查看"), {'display': 'none'}, "请选择一条交易记录进行查看", {
            'display': 'none', 'marginTop': '12px'}, "无数据", 0, None

    try:
        selected_row = all_transaction[selected_rows[0]]
        symbol = selected_row['symbol']
        direction = "做多" if selected_row['direction'] == 1 else "做空"
        start_time = pd.Timestamp(selected_row['start_time'])
        end_time = pd.Timestamp(selected_row['end_time'])
        results = BackTestingResults(strategy_name)
        stg_config = results.config.strategy_list[0]
        factor_name = default_factor
        factor_parameter = default_factor_parameter
        factor_col_name = f"{factor_name}_{factor_parameter}"
        draw_cols = [factor_col_name]

        market = stg_config.market
        if market == 'swap_swap':
            use_spot = False
        elif market is None:
            use_spot = stg_config.is_use_spot
        else:
            use_spot = True
        ohlc_data = get_ohlc_data(use_spot)
        data = ohlc_data.get_data_by_symbol(symbol, use_spot)
        factor_df = cal_factor(factor_name, factor_parameter, symbol, use_spot, ohlc_data)
        data = pd.merge(data, factor_df, on='candle_begin_time', how='left')
        picker_start_dt = pd.Timestamp(picker_start)
        picker_end_dt = pd.Timestamp(picker_end)
        plot_df = data[(data['candle_begin_time'] >= picker_start_dt) & (data['candle_begin_time'] <= picker_end_dt)]
        back_hour = (start_time - picker_start_dt).total_seconds() / 3600
        # Find timestamp of maximum close value
        transaction_df = plot_df[plot_df['candle_begin_time'] >= start_time]
        transaction_df = transaction_df[transaction_df['candle_begin_time'] <= end_time]
        max_close_idx = transaction_df['close'].idxmax()
        max_close_time = transaction_df.loc[max_close_idx, 'candle_begin_time']
        max_during_hour = int((max_close_time - start_time).total_seconds() / 3600)
        # 保证为0
        max_during_hour = max_during_hour if max_during_hour > 0 else 0
        config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"{symbol}-{factor_name}-{factor_parameter}-{start_time.strftime('%Y-%m-%d')}-{end_time.strftime('%Y-%m-%d')}",
                "height": None,
                "width": None,
                "scale": 1
            },
            "displayModeBar": True
        }
        trans_plot = plotly_transactions(plot_df, draw_cols, back_hour, start_time, end_time, symbol, None,
                                         "持仓时间", "持仓", True)
        return trans_plot, {'display': 'block',
                            'marginBottom': '12px'}, f"{symbol}-{direction}-{start_time}-{end_time}", {
            'display': 'block',
            'marginTop': '12px'}, f"最高价时间: {max_close_time},距离开始为{max_during_hour}小时", max_during_hour, config
    except Exception as e:
        print(f"出错:{e}")
        return px.scatter(title="请选择一条交易记录进行查看"), {'display': 'none'}, "请选择一条交易记录进行查看", {
            'display': 'none', 'marginTop': '12px'}, "无数据", 0, None


@callback(
    Output('strategy-display', 'children'),
    Input('select-strategy-store', 'data')
)
def display_strategy(value):
    if value is None:
        return "当前未选择策略"
    return f"当前选择的策略: {value}"


@callback(
    Output('parameter-display-table', 'data'),
    [Input('add-parameter-button', 'n_clicks')],
    [State('new-factor', 'value'),
     State('new-factor-parameter', 'value'),
     State('parameter-display-table', 'data')]
)
def update_parameter_table(n_clicks, factor, parameter, existing_data):
    if n_clicks is None:  # 页面初始化时不触发
        return existing_data if existing_data else []

    if not factor or parameter is None:  # 检查输入是否有效
        return existing_data if existing_data else []

    # 如果现有数据为None，初始化为空列表
    if existing_data is None:
        existing_data = []

    # 创建新行
    new_row = {"factor": factor, "parameter": parameter}

    # 添加新行到现有数据
    existing_data.append(new_row)

    return existing_data


@callback(
    Output("stg_drawback_table", "data"),
    [
        Input("stg-drawback-store", "data")
    ]
)
def refresh_drawback_table(data):
    return data


@callback(
    [
        Output('new-factor-plot', 'figure'),
        Output('new-factor-plot', 'style'),
        Output('loading-new-parameter-finished', 'children'),
    ],
    [
        Input("calculate-new-button", "n_clicks"),
        Input("n-hour-later", "value"),
    ],
    [
        State("parameter-display-table", "data"),
        State("transaction-table", "selected_rows"),
        State("transaction-table", "data"),
        State('select-strategy', 'value'),
    ]
)
def plot_new_factor_bar(n_clicks, hour_late, parameter_table, select_transaction_idx, transaction_data,
                        strategy_name):
    if not n_clicks:
        return None, {'display': 'none', 'height': '600px'}, "填写因子"

    if not select_transaction_idx:
        return None, {'display': 'none', 'height': '600px'}, "请选择一条交易记录进行查看"

    if parameter_table is None or len(parameter_table) == 0:
        return None, {'display': 'none', 'height': '600px'}, "填写因子"
    factor_list = []
    results = BackTestingResults(strategy_name)
    stg_config = results.config.strategy_list[0]
    market = stg_config.market
    if market == 'swap_swap':
        use_spot = False
    elif market is None:
        use_spot = stg_config.is_use_spot
    else:
        use_spot = True
    for p in parameter_table:
        factor_name = p['factor']
        factor_parameter = p['parameter']
        factor_list.append((factor_name, factor_parameter))
    source = get_ohlc_data(use_spot).calculate_factors(factor_list, use_spots=use_spot)
    select_row = transaction_data[select_transaction_idx[0]]
    start_time = pd.Timestamp(select_row['start_time'])
    end_time = start_time + datetime.timedelta(hours=hour_late)
    highlight_symbol = select_row['symbol']

    plot_data = []
    for factor in factor_list:
        plot_entry = plot_factor_data(source, factor, start_time, end_time, f"{factor[0]}-{factor[1]}")
        plot_data.append(plot_entry)
    plot_df = pd.concat(plot_data, ignore_index=True)
    # 绘制线图
    # 创建颜色分组：高亮symbol为一组，其他按tag分组
    plot_df['color_group'] = plot_df.apply(
        lambda row: f"🔥 {highlight_symbol}" if row['symbol'] == highlight_symbol
        else f"📊 {row['tag']}", axis=1
    )
    fig = px.bar(
        plot_df,
        x='x',
        y='y',
        facet_row="tag",
        hover_data=['symbol'],
        color="color_group"  # plotly会自动为不同的color_group分配不同颜色
    )

    fig.update_layout(annotations=[], overwrite=True)
    fig.update_yaxes(matches=None, showticklabels=True, title=None)
    fig.update_xaxes(matches=None, showticklabels=True)
    fig.update_layout(
        showlegend=True,
        margin=dict(t=10, l=10, b=10, r=10)
    )
    return fig, {'display': 'block', 'height': '600px'}, "新因子计算完毕"


@callback(
    [
        Output("transaction-with-new-parameter-modal", "is_open"),
        Output('new-parameter-one-transaction-plot', 'figure'),
        Output('new-parameter-one-transaction-plot-finished', 'children'),
        Output('new-parameter-one-transaction-plot', 'config'),
    ],
    [
        Input("plot-new-factor-button", "n_clicks"),

    ],
    [
        State("new-factor", "value"),
        State("new-factor-parameter", "value"),
        State('transaction-table', 'selected_rows'),
        State('transaction-table', 'data'),
        State('select-strategy', 'value'),
        State('transaction-hour-range-picker', 'start_date'),
        State('transaction-hour-range-picker', 'end_date'),
    ]
)
def plot_new_factor_parameter(n_clicks, factor, parameter, selected_rows, all_transaction, strategy_name, picker_start,
                              picker_end):
    if not n_clicks or len(selected_rows) == 0:
        return False, None, "", None
    selected_row = all_transaction[selected_rows[0]]
    symbol = selected_row['symbol']
    direction = "做多" if selected_row['direction'] == 1 else "做空"
    start_time = pd.Timestamp(selected_row['start_time'])
    end_time = pd.Timestamp(selected_row['end_time'])
    draw_cols = [factor]

    results = BackTestingResults(strategy_name)
    stg_config = results.config.strategy_list[0]

    market = stg_config.market
    if market == 'swap_swap':
        use_spot = False
    elif market is None:
        use_spot = stg_config.is_use_spot
    else:
        use_spot = True

    ohlc_data = get_ohlc_data(use_spot)
    data = ohlc_data.get_data_by_symbol(symbol, use_spot)
    data[factor] = cal_factor(factor, parameter, symbol, use_spot, ohlc_data)
    picker_start_dt = pd.Timestamp(picker_start)
    picker_end_dt = pd.Timestamp(picker_end)
    plot_df = data[(data['candle_begin_time'] >= picker_start_dt) & (data['candle_begin_time'] <= picker_end_dt)]
    back_hour = (start_time - picker_start_dt).total_seconds() / 3600

    # 保证为0
    config = {
        "toImageButtonOptions": {
            "format": "png",
            "filename": f"{symbol}-{factor}-{parameter}-{start_time.strftime('%Y-%m-%d')}-{end_time.strftime('%Y-%m-%d')}",
            "height": None,
            "width": None,
            "scale": 1
        },
        "displayModeBar": True
    }
    trans_plot = plotly_transactions(plot_df, draw_cols, back_hour, start_time, end_time, symbol, None,
                                     "持仓时间", "持仓", True)
    return True, trans_plot, f"{symbol}-{direction}-{start_time}-{end_time}", config


@callback(
    [
        Output("transaction-equity-curve-model", "is_open"),
        Output("transaction-equity-curve-plot", "figure"),
    ],
    Input("transaction-equity-curve", "n_clicks"),
    [
        State('transaction-range-slider', 'value'),
        State('select-strategy', 'value')
    ],
    prevent_initial_call=True
)
def open_equity_modal(n_clicks, time_range, select_strategy):
    if not n_clicks:
        return False, None
    # 只要点了“资金曲线”按钮，就打开对应的 Modal
    start_time = pd.Timestamp.fromtimestamp(time_range[0])
    end_time = pd.Timestamp.fromtimestamp(time_range[1])
    results = BackTestingResults(select_strategy)
    equity_df = results.equity
    plot_df = equity_df[
        (equity_df['candle_begin_time'] >= pd.to_datetime(start_time)) &
        (equity_df['candle_begin_time'] <= pd.to_datetime(end_time))
        ]
    bt_config = results.config
    fig = show_plot_performance(bt_config, plot_df, debug=False)
    # 调整图表布局以适配 Modal
    # 更紧凑的布局，避免撑宽
    fig.update_layout(
        height=600,
        width=1000,
        autosize=True,
    )
    fig.update_xaxes(title=None, automargin=True)
    fig.update_yaxes(automargin=True)

    return True, fig


@callback(
    [
        Output("transaction-month-return-model", "is_open"),
        Output("transaction-month-return-plot", "figure"),

    ],
    Input("transaction-month-return", "n_clicks"),
    [
        State('select-strategy', 'value')
    ],
    prevent_initial_call=True
)
def open_month_modal(n_clicks, select_stg):
    if not n_clicks:
        return False, None

    results = BackTestingResults(select_stg)
    equity_df = results.equity.copy()

    if equity_df.empty:
        return True, px.imshow([[None]], x=[1], y=["无数据"], color_continuous_scale="RdYlGn", origin="upper")

    # 统一时间列并按区间过滤
    equity_df['candle_begin_time'] = pd.to_datetime(equity_df['candle_begin_time'])

    if equity_df.empty:
        fig = px.imshow([[np.nan]], x=[1], y=["无数据"], color_continuous_scale="RdYlGn", origin="upper")
        fig.update_layout(height=500, width=900, margin=dict(t=10, l=10, b=10, r=10))
        return True, fig

    # 自动选择净值列名称
    candidate_cols = ['equity', 'net_value', 'nav', 'value']
    value_col = next((c for c in candidate_cols if c in equity_df.columns), None)
    if value_col is None:
        fig = px.imshow([[np.nan]], x=[1], y=["缺少净值列"], color_continuous_scale="RdYlGn", origin="upper")
        fig.update_layout(height=500, width=900, margin=dict(t=10, l=10, b=10, r=10))
        return True, fig

    # 计算月收益：���月末/每月初 - 1
    equity_df['year'] = equity_df['candle_begin_time'].dt.year
    equity_df['month'] = equity_df['candle_begin_time'].dt.month
    month_ret = equity_df.groupby(['year', 'month'])[value_col].agg(['first', 'last']).reset_index()
    month_ret['ret'] = month_ret['last'] / month_ret['first'] - 1

    # 透视为 年 x 月 的矩阵
    # 透视为 年 x 月 的矩阵
    pv = month_ret.pivot(index='year', columns='month', values='ret').sort_index(ascending=False)
    all_months = list(range(1, 12 + 1))
    pv = pv.reindex(columns=all_months)

    # 对称色轴
    values = pv.values

    # 计算最大最小值（可根据实际需求选择绝对值或直接 min/max）
    vmax = float(np.nanmax(values))
    vmin = float(np.nanmin(values))
    vmax = vmax if vmax < 1.5 else 1.5
    vmin = vmin if vmin > -0.6 else -0.6

    # 文本矩阵（百分比显示）
    values = pv.values
    text_matrix = np.empty_like(values, dtype=object)
    it = np.nditer(values, flags=['multi_index'])
    while not it.finished:
        val = it[0]
        text_matrix[it.multi_index] = "" if pd.isna(val) else f"{float(val) * 100:.1f}%"
        it.iternext()

    y_labels = pv.index.astype(str).tolist()  # y轴显示年份
    months_text = [f"{m}月" for m in all_months]  # x轴显示月份

    # 使用 go.Heatmap，明确类目轴
    import plotly.graph_objects as go
    fig = go.Figure(data=go.Heatmap(
        z=pv.values,
        x=months_text,  # 横轴使用月份："1月", "2月", ..., "12月"
        y=y_labels,  # 纵轴使用年份："2023", "2024", 等
        text=text_matrix,  # 方格内文本
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmin=vmin,
        zmax=vmax,
        hovertemplate="年份: %{y}<br>月份: %{x}<br>月收益: %{text}<extra></extra>",
        colorbar=dict(title="月收益", tickformat=".0%")
    ))

    # 坐标轴强制为类目并显示刻度
    fig.update_xaxes(
        type="category",
        categoryorder="array",
        categoryarray=months_text,  # x轴显示月份
        tickmode="array",
        tickvals=months_text,
        ticktext=months_text,
        showticklabels=True,
        title=None
    )
    fig.update_yaxes(
        type="category",
        categoryorder="array",
        categoryarray=y_labels,  # y轴显示年份
        tickmode="array",
        tickvals=y_labels,
        ticktext=y_labels,
        showticklabels=True,
        title="年份"
    )

    return True, fig


@callback(
    Output('open-tradingview-btn', 'n_clicks'),
    [Input('open-tradingview-btn', 'n_clicks')],
    [State('transaction-table', 'selected_rows'), State('transaction-table', 'data')],
    prevent_initial_call=True
)
def open_tradingview(n_clicks, selected_rows, all_transaction):
    if not n_clicks or not selected_rows:
        return dash.no_update
    selected_row = all_transaction[selected_rows[0]]
    symbol = selected_row['symbol']
    # 处理 symbol 格式，如 VIDT-USDT -> VIDTUSDT
    symbol_clean = symbol.replace('-', '').replace('_', '').upper()
    tv_symbol = f"BINANCE:{symbol_clean}"
    url = f"https://www.tradingview.com/chart/?symbol={tv_symbol}"
    import webbrowser
    webbrowser.open_new_tab(url)
    return 0


@callback(
    Output("download-transactions", "data"),
    Input("download-transactions-btn", "n_clicks"),
    State("transaction-table", "data"),
    State("select-strategy-store", "data"),
    prevent_initial_call=True
)
def download_transactions(n_clicks, table_data, strategy_name):
    if not n_clicks or not table_data or not strategy_name:
        return dash.no_update
    df = pd.DataFrame(table_data)
    time_col = pd.to_datetime(df['start_time'])
    start_time = time_col.min().strftime("%Y%m%d")
    end_time = time_col.max().strftime("%Y%m%d")
    filename = f"transactions_{strategy_name}_{start_time}_{end_time}.csv"
    return dcc.send_data_frame(df.to_csv, filename, index=False, encoding="utf-8-sig")


@callback(
    [
        Output('transaction-hour-range-picker', 'start_date'),
        Output('transaction-hour-range-picker', 'end_date'),
        Output('btn-prev-week', 'n_clicks'),
        Output('btn-prev-day', 'n_clicks'),
        Output('btn-next-day', 'n_clicks'),
        Output('btn-next-week', 'n_clicks'),
    ],

    Input('transaction-table', 'selected_rows'),
    Input('btn-prev-week', 'n_clicks'),
    Input('btn-prev-day', 'n_clicks'),
    Input('btn-next-day', 'n_clicks'),
    Input('btn-next-week', 'n_clicks'),

    [
        State('transaction-table', 'data'),
        State('transaction-hour-range-picker', 'start_date'),
        State('transaction-hour-range-picker', 'end_date'),
    ]
)
def set_transaction_hour_range(selected_rows, btn_prev_week, btn_prev_day, btn_next_day, btn_next_week, table_data,
                               pick_start_date, pick_end_date):
    if not selected_rows or not table_data:
        return None, None, 0, 0, 0, 0

    select_row_change = False
    triggered = callback_context.triggered
    if triggered:
        prop_id = triggered[0]['prop_id']
        if prop_id == 'transaction-table.selected_rows':
            select_row_change = True
    selected_row = table_data[selected_rows[0]]
    if select_row_change or (pick_start_date is None and pick_end_date is None):
        start_time = pd.Timestamp(selected_row['start_time'])
        end_time = pd.Timestamp(selected_row['end_time'])
        gap = end_time - start_time
        delta_hour = max(gap * 0.5, pd.Timedelta(hours=24 * 3))
        start_date = start_time - delta_hour
        end_date = end_time + delta_hour
    else:
        start_date = pd.Timestamp(pick_start_date)
        end_date = pd.Timestamp(pick_end_date)

    if btn_prev_week == 1:
        start_date = start_date - datetime.timedelta(days=7)

    if btn_prev_day == 1:
        start_date = start_date - datetime.timedelta(days=1)

    if btn_next_day == 1:
        end_date = end_date + datetime.timedelta(days=1)

    if btn_next_week == 1:
        end_date = end_date + datetime.timedelta(days=7)
    return start_date, end_date, 0, 0, 0, 0


@callback(
    Output('transaction-other-strategy-dropdown', 'options'),
    Output('transaction-other-strategy-dropdown', 'value'),  # 修正为连字符，匹配布局里的 id
    Input('select-strategy-store', 'data'),
)
def refresh_strategy_list(select_stg):
    stg_display = [s for s in list_all_strategy() if s != select_stg]
    select_compare_stg = stg_display[-1] if stg_display else None
    return [{'label': f, 'value': f} for f in stg_display], select_compare_stg


@callback(
    Output('transaction-stg-compare-median-plot', 'figure'),
    Input('transaction-compare-stg-plot-curve-btn', 'n_clicks'),
    State('select-strategy-store', 'data'),
    State('transaction-other-strategy-dropdown', 'value'),
    prevent_initial_call=True,
)
def plot_stg_diff(plot_click, select_stg, compare_stg):
    # This callback only computes and returns the figure. The Modal open is handled client-side
    if not plot_click:
        return dash.no_update
    base_result: BackTestingResults = BackTestingResults(name=select_stg)
    compare_result: BackTestingResults = BackTestingResults(name=compare_stg)
    # 计算多空策略
    result_df = calculate_market_neutral_portfolio(base_result.equity, compare_result.equity, rebalance_hours=4,
                                                   initial_balance=config.initial_usdt)
    # 计算性能指标（打印用于调试）
    performance = calculate_performance_metrics(result_df, initial_balance=config.initial_usdt)
    print(performance)
    compare_stg_plot = plot_strategy_analysis(result_df, select_stg, compare_stg, initial_balance=config.initial_usdt)
    return compare_stg_plot


# Client-side callback: open the Modal immediately when the button is clicked
dash.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) { return false; }
        return true;
    }
    """,
    Output('transaction-stg-compare-plot-model', 'is_open'),
    Input('transaction-compare-stg-plot-curve-btn', 'n_clicks')
)


def calculate_market_neutral_portfolio(df_a: pd.DataFrame, df_b: pd.DataFrame, rebalance_hours=4,
                                       initial_balance=10000):
    """
    计算市场中性的多空策略：50%做多A，50%做空B，定期重新平衡
    """
    # 重命名列以符合您的数据格式
    df_a = df_a.rename(columns={'candle_begin_time': 'timestamp', '净值': 'nav', '涨跌幅': 'return'})
    df_b = df_b.rename(columns={'candle_begin_time': 'timestamp', '净值': 'nav', '涨跌幅': 'return'})

    # 确保时间列格式正确
    df_a['timestamp'] = pd.to_datetime(df_a['timestamp'])
    df_b['timestamp'] = pd.to_datetime(df_b['timestamp'])

    # 设置时间索引
    df_a.set_index('timestamp', inplace=True)
    df_b.set_index('timestamp', inplace=True)

    # 合并数据，确保时间对齐
    merged_df = pd.DataFrame({
        'nav_a': df_a['nav'],
        'nav_b': df_b['nav'],
        'return_a': df_a['return'],
        'return_b': df_b['return']
    }).dropna()

    # 确保数据按时间排序
    merged_df = merged_df.sort_index()

    print(f"数据时间范围: {merged_df.index[0]} 到 {merged_df.index[-1]}")
    print(f"总数据点数: {len(merged_df)}")
    print(f"初始资金: {initial_balance}")

    # 初始化投资组合
    portfolio_records = []

    # 初始资金分配
    long_value = initial_balance * 0.5  # 做多A的资金
    short_value = initial_balance * 0.5  # 做空B的资金
    total_value = initial_balance

    # 记录持仓
    portfolio_records.append({
        'timestamp': merged_df.index[0],
        'portfolio_value': total_value,
        'long_value': long_value,
        'short_value': short_value,
        'long_position': 0.5,  # 做多A的仓位比例
        'short_position': -0.5,  # 做空B的仓位比例
        'return_a': 0,
        'return_b': 0,
        'portfolio_return': 0,
        'is_rebalance': True
    })

    last_rebalance_time = merged_df.index[0]

    # 遍历每个时间点
    for i in range(1, len(merged_df)):
        current_time = merged_df.index[i]
        prev_record = portfolio_records[-1]

        # 获取当前周期的收益率
        return_a = merged_df['return_a'].iloc[i] if not pd.isna(merged_df['return_a'].iloc[i]) else 0
        return_b = merged_df['return_b'].iloc[i] if not pd.isna(merged_df['return_b'].iloc[i]) else 0

        # 计算多空头寸的价值变化
        new_long_value = prev_record['long_value'] * (1 + return_a)
        new_short_value = prev_record['short_value'] * (1 - return_b)

        new_total_value = new_long_value + new_short_value

        # 计算新的仓位比例
        new_long_position = new_long_value / new_total_value
        new_short_position = -new_short_value / new_total_value

        # 检查是否需要重新平衡
        hours_since_rebalance = (current_time - last_rebalance_time).total_seconds() / 3600
        is_rebalance = hours_since_rebalance >= rebalance_hours

        if is_rebalance:
            # 重新平衡：调整到各50%的比例
            rebalanced_long_value = new_total_value * 0.5
            rebalanced_short_value = new_total_value * 0.5

            final_long_value = rebalanced_long_value
            final_short_value = rebalanced_short_value
            final_total_value = final_long_value + final_short_value

            last_rebalance_time = current_time
        else:
            # 不重新平衡，保持当前价值
            final_long_value = new_long_value
            final_short_value = new_short_value
            final_total_value = new_total_value

        # 计算最终仓位比例
        final_long_position = final_long_value / final_total_value
        final_short_position = -final_short_value / final_total_value

        # 计算投资组合收益率
        portfolio_return = (final_total_value - prev_record['portfolio_value']) / prev_record['portfolio_value']

        # 记录当前状态
        portfolio_records.append({
            'timestamp': current_time,
            'portfolio_value': final_total_value,
            'long_value': final_long_value,
            'short_value': final_short_value,
            'long_position': final_long_position,
            'short_position': final_short_position,
            'return_a': return_a,
            'return_b': return_b,
            'portfolio_return': portfolio_return,
            'is_rebalance': is_rebalance
        })

    # 转换为DataFrame
    result_df = pd.DataFrame(portfolio_records)
    result_df.set_index('timestamp', inplace=True)

    # 添加原始净值数据到结果中
    result_df['nav_a'] = merged_df['nav_a']
    result_df['nav_b'] = merged_df['nav_b']

    # 计算回撤
    cumulative_max = result_df['portfolio_value'].expanding().max()
    result_df['当前回撤'] = (result_df['portfolio_value'] - cumulative_max) / cumulative_max
    result_df['历史最高值'] = cumulative_max

    return result_df


def plot_strategy_analysis(result_df, stg_a_name: str, stg_b_name: str, initial_balance=10000, ):
    """
    使用Plotly绘制多空策略分析图表

    参数:
        result_df: 包含策略结果的DataFrame
        initial_balance: 初始资金

    返回:
        fig: Plotly Figure对象
    """

    # 创建3x2子图
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '多空策略资金曲线',
            '仓位比例变化',
            '当前回撤',
            '策略收益率分布',
            '重新平衡点标记',
            '与原始策略对比'
        ),
        specs=[[{}, {}], [{}, {}], [{}, {}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # 子图1：多空策略整体表现
    fig.add_trace(
        go.Scatter(
            x=result_df.index,
            y=result_df['portfolio_value'],
            name='多空策略总价值',
            line=dict(color='black', width=2),
            mode='lines'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=result_df.index,
            y=result_df['long_value'],
            name=f'做多{stg_a_name}部分',
            line=dict(color='green', width=1, dash='dash'),
            opacity=0.7,
            mode='lines'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=result_df.index,
            y=result_df['short_value'],
            name=f'做空{stg_b_name}部分',
            line=dict(color='red', width=1, dash='dash'),
            opacity=0.7,
            mode='lines'
        ),
        row=1, col=1
    )

    fig.update_xaxes(title_text='时间', row=1, col=1)
    fig.update_yaxes(title_text='价值', row=1, col=1)

    # 子图2：仓位比例变化
    fig.add_trace(
        go.Scatter(
            x=result_df.index,
            y=result_df['long_position'],
            name=f'做多{stg_a_name}仓位',
            line=dict(color='green', width=2),
            mode='lines'
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=result_df.index,
            y=result_df['short_position'],
            name=f'做空{stg_b_name}仓位',
            line=dict(color='red', width=2),
            mode='lines'
        ),
        row=1, col=2
    )
    # 添加目标仓位线
    fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.5,
                  annotation_text="目标多头仓位", row=1, col=2)
    fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.5,
                  annotation_text="目标空头仓位", row=1, col=2)

    fig.update_xaxes(title_text='时间', row=1, col=2)
    fig.update_yaxes(title_text='仓位比例', row=1, col=2)

    # 子图3：当前回撤
    fig.add_trace(
        go.Scatter(
            x=result_df.index,
            y=result_df['当前回撤'],
            name='当前回撤',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='darkred', width=1),
            mode='lines'
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_color="black", line_width=1, opacity=0.5, row=2, col=1)

    fig.update_xaxes(title_text='时间', row=2, col=1)
    fig.update_yaxes(title_text='回撤幅度', row=2, col=1)

    # 子图4：收益率分布（直方图）
    returns = result_df['portfolio_return'].dropna()
    fig.add_trace(
        go.Histogram(
            x=returns,
            name='收益率分布',
            nbinsx=50,
            marker=dict(color='blue', line=dict(color='black', width=1)),
            opacity=0.7,
            showlegend=False
        ),
        row=2, col=2
    )

    fig.update_xaxes(title_text='收益率', row=2, col=2)
    fig.update_yaxes(title_text='频数', row=2, col=2)

    # 子图5：重新平衡点标记
    fig.add_trace(
        go.Scatter(
            x=result_df.index,
            y=result_df['portfolio_value'],
            name='投资组合价值',
            line=dict(color='black', width=1),
            mode='lines'
        ),
        row=3, col=1
    )

    rebalance_points = result_df[result_df['is_rebalance'] == True]
    fig.add_trace(
        go.Scatter(
            x=rebalance_points.index,
            y=rebalance_points['portfolio_value'],
            name='重新平衡点',
            mode='markers',
            marker=dict(color='orange', size=8),
            showlegend=True
        ),
        row=3, col=1
    )

    fig.update_xaxes(title_text='时间', row=3, col=1)
    fig.update_yaxes(title_text='投资组合价值', row=3, col=1)

    # 子图6：原始策略对比
    if 'nav_a' in result_df.columns and 'nav_b' in result_df.columns:
        # 归一化到同一起点
        norm_a = result_df['nav_a'] / result_df['nav_a'].iloc[0] * initial_balance
        norm_b = result_df['nav_b'] / result_df['nav_b'].iloc[0] * initial_balance

        fig.add_trace(
            go.Scatter(
                x=result_df.index,
                y=norm_a,
                name=f'策略{stg_a_name}(纯做多）',
                line=dict(color='green'),
                opacity=0.7,
                mode='lines'
            ),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=result_df.index,
                y=norm_b,
                name=f'策略{stg_b_name}(纯做多）',
                line=dict(color='red'),
                opacity=0.7,
                mode='lines'
            ),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=result_df.index,
                y=result_df['portfolio_value'],
                name='多空策略',
                line=dict(color='black', width=2),
                mode='lines'
            ),
            row=3, col=2
        )
    else:
        fig.add_annotation(
            text='原始净值数据不可用',
            xref='paper', yref='paper',
            x=0.75, y=0.15,
            showarrow=False,
            font=dict(size=14, color='gray')
        )

    fig.update_xaxes(title_text='时间', row=3, col=2)
    fig.update_yaxes(title_text='价值', row=3, col=2)

    # 更新整体布局
    fig.update_layout(
        title_text='多空策略分析',
        height=1200,
        width=1500,
        showlegend=True,
        hovermode='x unified',
        font=dict(family='Arial, sans-serif')
    )

    # 为所有子图添加网格
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def calculate_performance_metrics(portfolio_df, initial_balance=10000):
    """计算投资组合性能指标"""

    returns = portfolio_df['portfolio_return'].dropna()

    if len(returns) == 0:
        return {
            'annual_return': 0,
            'annual_volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'current_drawdown': 0,
            'final_value': initial_balance,
            'total_return': 0
        }

    # 计算总收益率
    total_return = (portfolio_df['portfolio_value'].iloc[-1] - initial_balance) / initial_balance

    # 年化收益率（按小时数据计算）
    total_hours = len(portfolio_df)
    if total_hours > 24 * 365:  # 如果数据超过一年，使用年化计算
        years = total_hours / (24 * 365)
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = total_return * (24 * 365) / total_hours if total_hours > 0 else 0

    # 年化波动率
    annual_volatility = returns.std() * np.sqrt(24 * 365)

    # 夏普比率（假设无风险利率为0）
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

    # 最大回撤
    max_drawdown = portfolio_df['当前回撤'].min()
    current_drawdown = portfolio_df['当前回撤'].iloc[-1]

    # 多空收益分析
    long_total_return = (portfolio_df['long_value'].iloc[-1] - initial_balance * 0.5) / (initial_balance * 0.5)
    short_total_return = (portfolio_df['short_value'].iloc[-1] - initial_balance * 0.5) / (initial_balance * 0.5)

    performance = {
        '起始资金': initial_balance,
        '最终价值': portfolio_df['portfolio_value'].iloc[-1],
        '总收益率': total_return,
        '年化收益率': annual_return,
        '年化波动率': annual_volatility,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '当前回撤': current_drawdown,
        '做多部分收益': long_total_return,
        '做空部分收益': short_total_return,
        '多空收益差': long_total_return - short_total_return,
        '总交易小时数': total_hours,
        '重新平衡次数': portfolio_df['is_rebalance'].sum()
    }

    return performance
