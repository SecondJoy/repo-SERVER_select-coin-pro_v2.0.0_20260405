#JoyAdded 马代代码
# # Python
from pathlib import Path

import dash
import pandas as pd
from dash import html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

import config
from core.backtest import simu_re_timing
from core.model.timing_signal import TimingSignal
from core.select_coin import agg_multi_strategy_ratio
from core.utils.path_kit import get_folder_path
from mytools.func import reload_module_py
from mytools.models import BackTestingResults
from mytools.xbx import show_plot_performance

dash.register_page(__name__, path="/leverage_page", name="赌狗的快乐")

# 顶层 layout（或 layout() 函数都可以）
layout = html.Div([
    html.H1("赌狗的快乐"),
    dbc.Row([
        dbc.Col([
            html.Label("杠杆"),
            dcc.Dropdown(
                id='leverage-signal',
                # options=[],
                placeholder='signal',
                style={'width': '100%'}
            )
        ], md=3),
        dbc.Col([
            html.Label("参数(多个用逗号相隔)"),
            dcc.Input(id="leverage-parameter", style={'width': '100%'})
        ], md=3),
    ]),
    html.Hr(),
    dbc.Button("提交分析", id="leverage-analysis-button", color="primary", className="mt-3",
               style={"marginRight": "10px"}),
    dbc.Button("刷新代码", id="leverage-signal-refresh-button", color="primary", className="mt-3"),
    html.Hr(),
    dbc.Row([
        dcc.Loading(
            id='domain-time-curve',
            type='default',
            children=[
                dcc.Graph(id='leverage-legacy-plot', style={'display': 'none'}),
                dcc.Graph(id='leverage-new-plot', style={'display': 'none'}),
            ]
        ),

    ], id="leverage-plots-group"),
])

@callback(
    Output('leverage-signal', 'options'),  # 修正为连字符，匹配布局里的 id
    Input('leverage-signal-refresh-button', 'n_clicks'),
    Input('select-strategy-store', 'data'),
)
def refresh_factor_options(n_clicks, select_stg):
    """
    触发点：点击刷新按钮或策略切换时。
    作用：尝试重载 mytools.factors 包下的子模块（如果存在），
    然后重新计算 list_all_factors() 并返回用于 dcc.Dropdown 的 options 列表。
    """
    try:
        reload_module_py("signals")
        factors_folder = Path(get_folder_path("signals"))
        signals = [f.stem for f in factors_folder.glob("*.py") if f.stem != "__init__"]
        return [{'label': f, 'value': f} for f in signals]
    except Exception:
        return []

@callback(
    Output('leverage-legacy-plot', 'figure'),
    Output('leverage-new-plot', 'figure'),
    Output('leverage-legacy-plot', 'style'),
    Output('leverage-new-plot', 'style'),
    Input('leverage-analysis-button', 'n_clicks'),
    State('select-strategy-store', 'data'),
    State('leverage-signal', 'value'),
    State('leverage-parameter', 'value'),
    prevent_initial_call=True,
)
def analysis_leverage(n_clicks, select_stg, signal_name, parameter_str: str):
    # Only run when the button is clicked (other controls are now State)
    if n_clicks is None or signal_name is None or parameter_str is None or select_stg is None:
        return None, None, {'display': 'none'}, {'display': 'none'}
    stg_result = BackTestingResults(name=select_stg)
    stg_config = stg_result.config
    equity = stg_result.equity.copy()
    parameter = []
    parameter_list = parameter_str.split(",")
    for param in parameter_list:
        try:
            parameter.append(float(param))
        except Exception:
            parameter.append(param)

    signal_timing = TimingSignal(signal_name, parameter)
    equity["leverage"] = signal_timing.get_dynamic_leverage(equity['equity'])
    start_time = pd.to_datetime(config.start_date)
    end_time = pd.to_datetime(config.end_date)
    equity = equity[
        (equity['candle_begin_time'] >= start_time) &
        (equity['candle_begin_time'] <= end_time)
        ]

    t = equity["candle_begin_time"]
    v = equity["净值"]
    lev = equity["leverage"].values
    lev_min, lev_max = lev.min(), lev.max()

    legacy_fig = go.Figure()

    # 主线（单色）
    legacy_fig.add_trace(go.Scattergl(
        x=t, y=v,
        mode="lines",
        line=dict(color="#555", width=1.5),
        name="净值"
    ))

    # 彩色标记（按杠杆映射颜色）
    legacy_fig.add_trace(go.Scattergl(
        x=t, y=v,
        mode="markers",
        marker=dict(
            size=1,
            color=lev,
            colorscale="Turbo",
            cmin=lev_min, cmax=lev_max,
            showscale=True,
            colorbar=dict(
                title="杠杆",
                x=1.1  # 负值表示移到左侧，默认为 1.02（右侧）
            )
        ),
        hovertemplate="时间: %{x}<br>净值: %{y:.4f}<br>杠杆: %{marker.color:.2f}<extra></extra>",
        name="杠杆"
    ))

    legacy_fig.update_layout(template="none", hovermode="x unified", title="净值随杠杆")

    df_spot_ratio, df_swap_ratio = agg_multi_strategy_ratio(stg_config, stg_result.select_coins)
    pivot_dict_spot = pd.read_pickle(config.raw_data_path / 'market_pivot_spot.pkl')  # 读取现货市场的枢纽点数据
    pivot_dict_swap = pd.read_pickle(config.raw_data_path / 'market_pivot_swap.pkl')  # 读取合约市场的枢纽点数据
    stg_config.timing = signal_timing
    account_df2, rtn2, year_return2 = simu_re_timing(
        stg_config, df_spot_ratio, df_swap_ratio, pivot_dict_spot, pivot_dict_swap
    )

    new_fig = show_plot_performance(stg_result.config, account_df2)

    return legacy_fig, new_fig, {'display': 'block'}, {'display': 'block'}



