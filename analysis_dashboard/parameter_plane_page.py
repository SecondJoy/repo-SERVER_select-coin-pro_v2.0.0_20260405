# Python
import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px  # 新增：用于取丰富的调色板
import plotly.graph_objs as go
from dash import html, dcc, callback, Output, Input, State
from dash.exceptions import PreventUpdate

import config
from mytools.models import SearchResult
from mytools.streamlist_tools import list_all_search
from mytools.xbx import draw_params_heatmap_plotly, draw_params_bar_plotly

dash.register_page(__name__, path="/parameter_plane", name="Hello")

# 新增：统一上色函数
def make_colorful(fig: go.Figure, heatmap_scale: str = "Turbo") -> go.Figure:
    # 组合多套高饱和的离散色板，尽量避免重复
    palette = (
        px.colors.qualitative.Set3
        + px.colors.qualitative.Bold
        + px.colors.qualitative.Dark24
        + px.colors.qualitative.Pastel
        + px.colors.qualitative.Vivid
    )
    color_idx = 0
    for tr in fig.data:
        t = getattr(tr, "type", None)
        # 离散系列（柱/线/散点）统一按序着色
        if t in ("bar", "scatter", "scattergl"):
            tr.update(marker=dict(color=palette[color_idx % len(palette)]))
            # 线图也设置线条颜色
            if hasattr(tr, "line"):
                tr.update(line=dict(color=palette[color_idx % len(palette)]))
            color_idx += 1
        # 热力图设置更鲜艳的色板并显示色条
        elif t in ("heatmap", "heatmapgl"):
            tr.update(colorscale=heatmap_scale, reversescale=False)
            tr.update(colorbar=dict(title="值", titleside="right"))
    # 全局默认色序（用于 legend 等）
    fig.update_layout(colorway=palette)
    return fig


# 顶层 layout（或 layout() 函数都可以）
avail_stg = [{'label': f, 'value': f} for f in list_all_search()]
bool_option = [{'label': "是", 'value': True}, {'label': "否", 'value': False}]
layout = html.Div([
    html.H1("热力图"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id='parameter-stg-select',
                options=avail_stg,
                placeholder='策略',
            )
        ], width=10)
    ]
    ),
    dbc.Row([
        dbc.Col([
            html.Label("因子"),
            dcc.Dropdown(
                id='parameter_plane_factor',
                placeholder='因子',
                style={'width': '100%'},
                multi=True,
                maxHeight=200,
                value=[],
                clearable=True
            )
        ], width=2),
        dbc.Col([
            html.Label("持币时间(小时)"),
            dcc.Dropdown(
                id='parameter_plane_hold_period',
                placeholder='因子',
                style={'width': '100%'},
                maxHeight=200,
                clearable=True
            )
        ], width=2),
        dbc.Col([
            html.Label("选币数"),
            dcc.Dropdown(
                id='parameter_plane_select_num',
                placeholder='因子',
                style={'width': '100%'},
                maxHeight=200,
                clearable=True
            )
        ], width=2),
        dbc.Col([
            html.Label("现货"),
            dcc.Dropdown(
                id='parameter_plane_use_spot',
                options=bool_option,
                value=True,
                style={'width': '100%'}
            )
        ], width=2),
    ]),
    html.Hr(),
    dbc.Button("提交分析", id="parameter-plane-submit-analysis", color="primary", className="mt-3"),
    html.Hr(),
    dcc.Loading(
        id="parameter-plane-plot-loading",
        type="circle",  # 可选："default", "circle", "dot", "graph"
        children=html.Div(id="parameter-plane-plot-loading-finished")
    ),
    dbc.Row([

        dcc.Graph(id='parameter-plane-plot'),
    ], id="parameter-plane-plots-group", style={'display': 'none'}),

])


@callback(
    [
        Output("parameter_plane_factor", "options"),
        Output("parameter_plane_hold_period", "options"),
        Output("parameter_plane_select_num", "options"),
    ],
    Input("parameter-stg-select", "value"),
    prevent_initial_call=True,
)
def update_parameter_plan_data(select_search):
    root = config.backtest_iter_path / select_search
    results = SearchResult(root)

    all_df = results.to_df()
    hold_periods = all_df['hold_period'].unique()
    select_nums = all_df['long_select_num'].unique()
    if len(select_nums) == 1 and select_nums == [0]:
        select_nums = all_df['short_select_num'].unique()
    factor_option = [{'label': f, 'value': f} for f in results.get_factors()]
    hold_period_option = [{'label': f, 'value': f} for f in hold_periods]
    select_nums_option = [{'label': f, 'value': f} for f in select_nums]
    return factor_option, hold_period_option, select_nums_option


@callback(
    Output("parameter_plane_factor", "value"),
    Input("parameter_plane_factor", "value"),
    prevent_initial_call=True,
)
def enforce_max_two(selected):
    if selected is None:
        raise PreventUpdate
    if len(selected) <= 2:
        raise PreventUpdate
    return selected[-2:]


@callback(
    [
        Output("parameter-plane-plot-loading-finished", "children"),
        Output("parameter-plane-plot", "figure"),
        Output("parameter-plane-plots-group", "style"),
    ],
    Input("parameter-plane-submit-analysis", "n_clicks"),
    [
        State("parameter-stg-select", "value"),
        State("parameter_plane_factor", "value"),
        State("parameter_plane_hold_period", "value"),
        State("parameter_plane_select_num", "value"),
    ],
    prevent_initial_call=True,
)
def plot_plane_page(n_clicks, select_search, factor, hold_period, select_num):
    if not n_clicks:
        return "无事", None, {'display': 'none'}
    root = config.backtest_iter_path / select_search
    results = SearchResult(root)
    time_list = results.get_times()
    all_df = results.to_df()
    plot_data = all_df.loc[all_df['hold_period'] == hold_period]
    long_select_num = all_df['long_select_num'].unique()
    if len(long_select_num) == 1 and long_select_num == [0]:
        plot_data = plot_data.loc[plot_data['short_select_num'] == select_num]
    else:
        plot_data = plot_data.loc[plot_data['long_select_num'] == select_num]
    # 2 个因子 -> 热力图（多色）
    if select_num == 2:
        factor1 = factor[0]
        factor2 = factor[1]
        idx = plot_data.groupby([factor1, factor2])['all'].idxmax()
        plot_data = plot_data.loc[idx].copy()
        plot_data = plot_data[[factor1, factor2] + ['all'] + time_list].copy()
        plot_data = plot_data.dropna(subset=[factor1, factor2])

        for time in ['all'] + time_list:
            temp = pd.pivot_table(plot_data, index=factor1, columns=factor2, values=time)
            temp = temp.dropna()
            if temp.empty:
                return "过滤之后无数据", None, {'display': 'none'}
            fig = draw_params_heatmap_plotly(temp, title=time)
            fig = make_colorful(fig, heatmap_scale="Turbo")  # 应用彩色热力图
            return "参数平原", fig, {'display': 'block'}

    # 1 个因子 -> 柱状图（多色）
    elif select_num == 1:
        factor_name = factor[0]
        plot_data = all_df.loc[all_df['hold_period'] == hold_period]
        plot_data = plot_data.dropna(subset=[factor_name])
        if plot_data.empty:
            return "过滤之后无数据", None, {'display': 'none'}
        idx = plot_data.groupby(factor_name)['all'].idxmax()
        plot_data = plot_data.loc[idx].copy()
        plot_data = plot_data[[factor_name] + ['all'] + time_list].copy()
        plot_data[factor_name] = plot_data[factor_name].map(lambda x: f'{factor_name}_{x}')
        plot_data = plot_data.set_index(factor_name)
        evaluation_indicator = '年化收益'
        fig = draw_params_bar_plotly(plot_data, evaluation_indicator)
        fig = make_colorful(fig)  # 应用彩色系列
        return "参数平原", fig, {'display': 'block'}

    return "选择因子数量不对", None, {'display': 'none'}
