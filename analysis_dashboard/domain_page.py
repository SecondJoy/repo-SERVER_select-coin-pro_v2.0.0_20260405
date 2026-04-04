# Python
from typing import List

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import html, dcc, callback, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
from pandas import DataFrame
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots

import config
from mytools.func import reload_module_py, add_n_hour_return, cache_data, bin_group, daily_bin_counts, BinInfo
from mytools.models import BackTestingResults,BASE_MERGE_COL, get_ohlc_data
from mytools.streamlist_tools import list_all_factors

dash.register_page(__name__, path="/domain_page", name="分域分析")


def create_domain_time_range_slider():
    return dcc.RangeSlider(
        id='domain-time-range-slider',
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


cut_type_options = [
    {'label': '等宽分箱', 'value': 'cut'},
    {'label': '等频分箱', 'value': 'qcut'}
]

cut_group_period = [
    {'label': '小时', 'value': 'hour'},
    {'label': '全部', 'value': 'all'}
]

# 顶层 layout（或 layout() 函数都可以）
all_ready_factors = [{'label': f, 'value': f} for f in list_all_factors()]
layout = html.Div([
    html.H1("分域分析"),
    dbc.Row([
        create_domain_time_range_slider()
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("N小时后收益"),
            dcc.Input(id="hour-later", type='number', value=3)
        ], md=2),
        dbc.Col([
            html.Label("范围选择"),
            dcc.Dropdown(
                id="coin-scope-select",
                options=[
                    {'label': '全部', 'value': 'all'},
                    {'label': '现货', 'value': 'spot'},
                    {'label': '合约', 'value': 'futures'},
                    {'label': '策略', 'value': 'strategy'},
                ],
                value='strategy')
        ], md=3)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("因子"),
            dcc.Dropdown(
                id='domain-first-factor',
                placeholder='因子',
                style={'width': '100%'},
                value=[],
                options=all_ready_factors,
                clearable=False
            )
        ], width=5),
        dbc.Col([
            html.Label("参数"),
            dcc.Input(id="domain-first-parameter", type='number', style={'width': '100%'})
        ], width=2),
        dbc.Col([
            html.Label("分箱方式"),
            dcc.Dropdown(
                id='domain-first-cut-type',
                placeholder='分箱方式',
                style={'width': '100%'},
                value='cut',
                options=cut_type_options,
                clearable=False
            )
        ], width=2),
        dbc.Col([
            html.Label("分箱周期"),
            dcc.Dropdown(
                id='domain-first-cut-period',
                placeholder='分箱方式',
                style={'width': '100%'},
                value='all',
                options=cut_group_period,
                clearable=False
            )
        ], width=2)
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("因子"),
            dcc.Dropdown(
                id='domain-second-factor',
                placeholder='因子',
                style={'width': '100%'},
                value=[],
                options=all_ready_factors,
                clearable=True
            )
        ], width=5),
        dbc.Col([
            html.Label("参数"),
            dcc.Input(id="domain-second-parameter", type='number', style={'width': '100%'})
        ], width=2),
        dbc.Col([
            html.Label("分箱方式"),
            dcc.Dropdown(
                id='domain-second-cut-type',
                placeholder='分箱方式',
                style={'width': '100%'},
                value='cut',
                options=cut_type_options,
                clearable=False
            )
        ], width=2),
        dbc.Col([
            html.Label("分箱周期"),
            dcc.Dropdown(
                id='domain-second-cut-period',
                placeholder='分箱方式',
                style={'width': '100%'},
                value='all',
                options=cut_group_period,
                clearable=False
            )
        ], width=2)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button("提交分析", id="domain-analysis-button", color="primary", className="mt-3",
                       style={"marginRight": "10px"}),
            dbc.Button("刷新因子", id="domain-refresh-factor-button", color="primary", className="mt-3"),
        ], width=5),
    ]),
    dbc.Row([
        # Wrap coin diff plot with a Loading component and a text placeholder for messages
        dcc.Loading(
            id='domain-time-curve',
            type='default',
            children=[
                dcc.Graph(id='domain-time-curve-plot', style={'display': 'none'}),
                html.Div(id='domain-time-curve-text', style={'marginTop': '6px', 'color': '#666'})
            ]
        ),
    ],
        id="domain-time-curve-row",
    ),
    dbc.Row([
        # Wrap equity relation plot with a Loading component and a text placeholder for messages
        dcc.Loading(
            id='domain-num-loading',
            type='default',
            children=[
                dcc.Graph(id='domain-num-plot', style={'display': 'none'}),
                html.Div(id='domain-num-text', style={'marginTop': '6px', 'color': '#666'})
            ]
        ),
    ],
        id="domain-num-row",
    ),
    dbc.Row([
        # Wrap coin diff plot with a Loading component and a text placeholder for messages
        dcc.Loading(
            id='domain-profit-diff',
            type='default',
            children=[
                dcc.Graph(id='domain-profit-plot', style={'display': 'none'}),
                html.Div(id='domain-profit-text', style={'marginTop': '6px', 'color': '#666'})
            ]
        ),
    ],
        id="domain-profit-row",
    ),
])


# 回调：按钮点击后显示两行
@callback(
    Output('domain-num-plot', 'figure'),
    Output('domain-profit-plot', 'figure'),
    Output('domain-time-curve-plot', 'figure'),
    Output('domain-num-plot', 'style'),
    Output('domain-profit-plot', 'style'),
    Output('domain-time-curve-plot', 'style'),
    Input('domain-analysis-button', 'n_clicks'),
    State('select-strategy-store', 'data'),
    State('hour-later', 'value'),
    State('domain-first-factor', 'value'),
    State('domain-first-parameter', 'value'),
    State('domain-second-factor', 'value'),
    State('domain-second-parameter', 'value'),
    State('coin-scope-select', 'value'),
    State('domain-time-range-slider', 'value'),
    State('domain-first-cut-type', 'value'),
    State('domain-first-cut-period', 'value'),
    State('domain-second-cut-type', 'value'),
    State('domain-second-cut-period', 'value'),
    prevent_initial_call=True,
)
def analysis_domain(n_clicks, select_stg, hour_later, first_factor, first_parameter, second_factor, second_parameter,
                    coin_scope, time_range, first_cut_type, first_cut_period, second_cut_type, second_cut_period):
    """
    关于分组的逻辑。暂时定成，如果是hour的，那么就是全局的排序
    如果使用非小时的话。那么就是全部的排序
    """
    # Only run when the button is clicked (other controls are now State)
    if n_clicks is None:
        return None, None, None, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
    factors_infos = [(first_factor, first_parameter)]
    first_col_name = f"{first_factor}_{first_parameter}"
    return_col_name = f"return_{hour_later}h"
    merge_col_list = BASE_MERGE_COL + [first_col_name, return_col_name]
    second_col_name = None
    if second_factor:
        factors_infos.append((second_factor, second_parameter))
        second_col_name = f"{second_factor}_{second_parameter}"
        merge_col_list.append(second_col_name)

    start_ts, end_ts = (None, None)
    if time_range and isinstance(time_range, (list, tuple)) and len(time_range) == 2:
        try:
            start_ts, end_ts = int(time_range[0]), int(time_range[1])
            start_time = pd.to_datetime(start_ts, unit='s')
            end_time = pd.to_datetime(end_ts, unit='s')
        except Exception:
            start_time, end_time = None, None
    else:
        start_time, end_time = None, None

    cache_name = f"{hour_later}_{first_factor}_{first_parameter}_{second_factor}_{second_parameter}"
    factor_with_return = cache_data(load_return_data, cache_name, False, 168, factors_infos, hour_later,
                                    'close', merge_col_list)
    ohlc = get_ohlc_data()
    ohlc_df = ohlc.all_in_one()
    factor_with_return = pd.merge(ohlc_df, factor_with_return, on=BASE_MERGE_COL,
                                  how='inner')
    first_bin_col = f"{first_col_name}_bin"
    second_bin_col = f"{second_col_name}_bin"
    first_bin_info = BinInfo(cut_type=first_cut_type, group_period=first_cut_period)
    first_bin_df = bin_group(factor_with_return, value_col=first_col_name, new_col=first_bin_col,
                             bin_info=first_bin_info)
    second_bin_df = None
    if second_factor:
        second_bin_info = BinInfo(cut_type=second_cut_type, group_period=second_cut_period)
        second_bin_df = bin_group(factor_with_return, value_col=second_col_name,
                                  new_col=second_bin_col,
                                  bin_info=second_bin_info)
    stg_res = BackTestingResults(name=select_stg)
    if coin_scope == 'strategy':

        stg_select_coins = stg_res.select_coins
        first_bin_df = first_bin_df[BASE_MERGE_COL + [first_col_name, first_bin_col, return_col_name]]
        first_bin_df = pd.merge(stg_select_coins, first_bin_df, on=BASE_MERGE_COL, how='inner')
        if second_factor:
            second_bin_df = second_bin_df[BASE_MERGE_COL + [second_col_name, second_bin_col, return_col_name]]
            second_bin_df = pd.merge(stg_select_coins, second_bin_df, on=BASE_MERGE_COL, how='inner')
    elif coin_scope == 'spot':
        analysis_df = factor_with_return[factor_with_return['is_spot'] == 1]
        first_bin_df = first_bin_df[BASE_MERGE_COL + [first_col_name, first_bin_col, return_col_name]]
        first_bin_df = pd.merge(analysis_df, first_bin_df, on=BASE_MERGE_COL, how='inner')
        if second_factor:
            second_bin_df = second_bin_df[BASE_MERGE_COL + [second_col_name, second_bin_col, return_col_name]]
            second_bin_df = pd.merge(analysis_df, second_bin_df, on=BASE_MERGE_COL, how='inner')
    elif coin_scope == 'futures':
        analysis_df = factor_with_return[factor_with_return['is_spot'] == 0]
        first_bin_df = first_bin_df[BASE_MERGE_COL + [first_col_name, first_bin_col, return_col_name]]
        first_bin_df = pd.merge(analysis_df, first_bin_df, on=BASE_MERGE_COL, how='inner')
        if second_factor:
            second_bin_df = second_bin_df[BASE_MERGE_COL + [second_col_name, second_bin_col, return_col_name]]
            second_bin_df = pd.merge(analysis_df, second_bin_df, on=BASE_MERGE_COL, how='inner')
    equity = stg_res.equity
    if start_time is not None and end_time is not None:
        # 确保为 pd.Timestamp
        if not isinstance(start_time, pd.Timestamp):
            start_time = pd.to_datetime(start_time)
        if not isinstance(end_time, pd.Timestamp):
            end_time = pd.to_datetime(end_time)
        first_bin_df = first_bin_df[
            (first_bin_df['candle_begin_time'] >= start_time) &
            (first_bin_df['candle_begin_time'] <= end_time)
            ]
        equity = equity[
            (equity['candle_begin_time'] >= start_time) &
            (equity['candle_begin_time'] <= end_time)
            ]
        if second_factor:
            second_bin_df = second_bin_df[
                (second_bin_df['candle_begin_time'] >= start_time) &
                (second_bin_df['candle_begin_time'] <= end_time)
                ]
    if second_factor:
        count_fig, profile_fig = create_tow_factor_figs(first_bin_df, second_bin_df, first_col_name, second_col_name,
                                                        return_col_name)
    else:
        count_fig, profile_fig = create_single_factor_figs(first_bin_df, first_col_name, return_col_name)

    time_curve_fig = create_time_curve(first_bin_df, first_col_name, equity, first_bin_info)
    return count_fig, profile_fig, time_curve_fig, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}


def create_tow_factor_figs(first_bin_df: pd.DataFrame, second_bin_df: pd.DataFrame, first_col_name: str,
                           second_col_name: str,
                           return_col_name: str) -> tuple[
    Figure, Figure]:
    first_bin_col = f"{first_col_name}_bin"
    second_bin_col = f"{second_col_name}_bin"
    bin_data = pd.merge(first_bin_df, second_bin_df, on=BASE_MERGE_COL, how='inner')
    # 丢弃任一分组为空的数据，然后按 (first_bin, second_bin) 计数
    grouped = bin_data.dropna(subset=[first_bin_col, second_bin_col])
    if grouped.empty:
        # 返回两个空图以避免回调报错
        return go.Figure(), go.Figure()

    counts = grouped.groupby([first_bin_col, second_bin_col]).size().reset_index(name='count')
    counts_pivot = counts.pivot(index=first_bin_col, columns=second_bin_col, values='count').fillna(0)

    # 交集样本数热力图
    count_fig = px.imshow(
        counts_pivot,
        labels={'x': second_col_name, 'y': first_col_name, 'color': 'count'},
        x=counts_pivot.columns.astype(str),
        y=counts_pivot.index.astype(str),
        title='两个因子分组交集样本数热力图',
        aspect='auto',
        color_continuous_scale='Viridis'
    )

    # 计算每个交集的平均收益并绘制热力图
    if return_col_name in grouped.columns:
        avg = grouped.groupby([first_bin_col, second_bin_col])[return_col_name].mean().reset_index(name='avg_return')
        avg_pivot = avg.pivot(index=first_bin_col, columns=second_bin_col, values='avg_return').fillna(0)
        profile_fig = px.imshow(
            avg_pivot,
            labels={'x': second_col_name, 'y': first_col_name, 'color': 'avg_return'},
            x=avg_pivot.columns.astype(str),
            y=avg_pivot.index.astype(str),
            title='两个因子分组交集平均收益热力图',
            aspect='auto',
            color_continuous_scale='RdBu'
        )
    else:
        profile_fig = go.Figure()

    return count_fig, profile_fig


def create_single_factor_figs(first_bin_df: pd.DataFrame, first_col_name: str, return_col_name: str) -> tuple[
    Figure, Figure]:
    bin_col = f"{first_col_name}_bin"
    bin_counts = first_bin_df.dropna(subset=[bin_col]).groupby(bin_col).size().reset_index(name='count')
    bin_counts = bin_counts.sort_values(by=bin_col)
    profile_data = first_bin_df.dropna(subset=[bin_col]).groupby(bin_col)[return_col_name].mean().reset_index(
        name='avg_return')
    # 未点击或 n_clicks 为 None
    count_fig = px.bar(bin_counts, x=bin_col, y='count', labels={'count': '样本数', bin_col: '分组'}, title='个数统计')
    profile_fig = px.bar(profile_data, x=bin_col, y='avg_return', labels={'avg_return': '平均收益', bin_col: '分组'},
                         title='收益分组')
    return count_fig, profile_fig


def create_time_curve(analysis_df: DataFrame, first_col_name: str, equity_df: pd.DataFrame,
                      bin_info: BinInfo) -> Figure:
    # daily_bin_counts_df = daily_bin_counts(analysis_df[['candle_begin_time', 'symbol', first_col_name]],
    #                                        date_col='candle_begin_time', value_col=first_col_name)
    first_bin_col = f"{first_col_name}_bin"
    daily_bin_counts_df= analysis_df.groupby(['candle_begin_time', first_bin_col]).size().unstack(fill_value=0).sort_index()
    numeric_cols = daily_bin_counts_df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) < 3:
        # 数据不够时返回空图避免出错
        return go.Figure()
    low_cols = numeric_cols[:3]
    mid_cols = numeric_cols[3:7]
    high_cols = numeric_cols[-3:]
    zero_series = pd.Series(0, index=daily_bin_counts_df.index)
    high_series = daily_bin_counts_df[high_cols].sum(axis=1) if high_cols else zero_series.copy()
    mid_series = daily_bin_counts_df[mid_cols].sum(axis=1) if mid_cols else zero_series.copy()
    low_series = daily_bin_counts_df[low_cols].sum(axis=1) if low_cols else zero_series.copy()
    num_count_df = pd.DataFrame({
        '低': low_series,
        '中': mid_series,
        '高': high_series
    })
    num_count_df_percentage = num_count_df.div(num_count_df.sum(axis=1), axis=0) * 100
    df = num_count_df_percentage.reset_index()
    x_col = df.columns[0]

    # 指定从底到顶的顺序
    order = ['低', '中', '高']

    # 若需要 100% 堆叠，取消下面注释并使用 values = pct
    # pct = df[order].div(df[order].sum(axis=1), axis=0) * 100
    # values = pct
    values = df[order]

    # 计算累积值（按 order 顺序）
    cum = values.cumsum(axis=1)

    # Make subplot: enable secondary_y on first row so we can plot 回撤 on right axis
    time_curve_fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,  # 共享 x 轴，主，子图共同变化
        vertical_spacing=0.03,  # 减少主图和子图之间的间距
        row_heights=[0.5, 0.5],  # 主图高度占 80%，子图高度占 10%
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
    )
    time_curve_fig.update_layout(height=800)

    time_curve_fig.add_trace(go.Scatter(x=df[x_col], y=equity_df['净值'], name="净值"), row=1, col=1)
    # 右轴: 回撤，绑定到次坐标轴（secondary_y=True）。不要在 trace 内部使用 yaxis='y2'，而是通过 add_trace 的 secondary_y 参数绑定。
    time_curve_fig.add_trace(
        go.Scatter(x=df[x_col], y=equity_df['净值dd2here'], name='回撤(右轴)',
                   marker=dict(color='rgba(220, 220, 220, 0.8)'),
                   opacity=0.1, line=dict(width=0), fill='tozeroy'),
        row=1, col=1, secondary_y=True
    )

    for i, cat in enumerate(order):
        y = cum[cat]
        fill_mode = 'tozeroy' if i == 0 else 'tonexty'  # 第一个填到零，其余填到上一个 trace
        time_curve_fig.add_trace(go.Scatter(
            x=df[x_col],
            y=y,
            mode='lines',
            name=cat,
            fill=fill_mode
        ), row=2, col=1)

    time_curve_fig.update_layout(title='高中低堆叠面积图',
                                 xaxis_title=x_col, yaxis_title='占比')
    return time_curve_fig


def load_return_data(factor_infos: List[tuple[str, int]], hours_left=168, close_col='high',
                     merge_col: List[str] | None = None) -> pd.DataFrame:
    ohlc = get_ohlc_data()
    start_time = config.start_date
    end_date = config.end_date
    factors = ohlc.calculate_factors(factor_infos, use_spots=True)
    factors = factors[(factors['candle_begin_time'] >= pd.to_datetime(start_time)) & (
            factors['candle_begin_time'] <= pd.to_datetime(end_date))]
    data_with_return = add_n_hour_return(factors, n=hours_left, close_col=close_col)
    if merge_col is None:
        return data_with_return
    else:
        return data_with_return[merge_col]


@callback(
    Output('domain-first-factor', 'options'),
    Output('domain-second-factor', 'options'),
    Output('domain-first-factor', 'value'),
    Output('domain-first-parameter', 'value'),
    [
        Input('domain-refresh-factor-button', 'n_clicks'),
        Input('select-strategy-store', 'data'),
    ],
)
def refresh_factor_options(n_clicks, select_strategy):
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

        # 重新读取因子列表
        new_factors = [{'label': f, 'value': f} for f in list_all_factors()]
        return new_factors, new_factors, factor_name, factor_parameter
    except Exception:
        return all_ready_factors, all_ready_factors, None, None


# python
@callback(
    [
        Output('domain-time-range-slider', 'min'),
        Output('domain-time-range-slider', 'max'),
        Output('domain-time-range-slider', 'marks'),
        Output('domain-time-range-slider', 'value'),
    ],
    Input('select-strategy-store', 'data'),
)
def refresh_time_select_slider(select_stg):
    try:
        min_time = pd.to_datetime(config.start_date)
        max_time = pd.to_datetime(config.end_date)
    except Exception:
        # 回退到合理默认范围，避免出错
        min_time = pd.to_datetime("2020-01-01")
        max_time = pd.to_datetime("2020-12-31")

    # 转换为时间戳（秒）
    min_ts = int(min_time.timestamp())
    max_ts = int(max_time.timestamp())

    if min_ts >= max_ts:
        # 确保有一个有效范围
        max_ts = min_ts + 86400

    # 生成月份标记，按月的第一个日（MS），但画面上只显示每三个月一个标签以避免拥挤
    dates = pd.date_range(start=min_time, end=max_time, freq='MS')
    marks = {}
    total = len(dates)
    for i, d in enumerate(dates):
        ts = int(d.timestamp())
        if total <= 6 or (i % 3 == 0):
            marks[ts] = {
                'label': d.strftime('%Y-%m'),
                'style': {
                    'font-size': '12px',
                    'text-align': 'center'
                }
            }

    return min_ts, max_ts, marks, [min_ts, max_ts]


@callback(
    Output('domain-time-range-slider', 'tooltip'),
    [Input('domain-time-range-slider', 'value')]
)
def update_transaction_range_slider_tooltip(value):
    if not value:
        return {"always_visible": True, "placement": "top"}
    return {
        "always_visible": True,
        "placement": "top",
        "template": f"{pd.Timestamp.fromtimestamp(value[0]).strftime('%Y-%m-%d')} 到 {pd.Timestamp.fromtimestamp(value[1]).strftime('%Y-%m-%d')}"
    }
