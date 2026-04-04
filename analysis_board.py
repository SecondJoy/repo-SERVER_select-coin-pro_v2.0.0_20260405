import datetime
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONLEGACYWINDOWSSTDIO"] = "1"  # 针对 Windows 环境的特殊补丁
import _locale
_locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])

import dash
from dash import Dash, dcc, html, Output, Input

import config

import dash_bootstrap_components as dbc

from core.utils.path_kit import get_file_path
from mytools.func import cleanup_analysis_cache
from mytools.streamlist_tools import list_all_strategy
import os
import shutil

# 初始化 Dash 应用，加载 Bootstrap 主题
app = Dash(__name__,
           use_pages=True,
           title="马代查看器(选币版)",
           pages_folder="analysis_dashboard",
           external_stylesheets=[dbc.themes.SIMPLEX],
           suppress_callback_exceptions=True)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

all_strategy = list_all_strategy()
strategy_name = config.backtest_name
sidebar = html.Div(
    [
        html.H2("功能列表", className="display-4"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("交易记录", href="/transaction", active="exact"),
                dbc.NavLink("热力图", href="/parameter_plane", active="exact"),
                dbc.NavLink("IC分析", href="/ic", active="exact"),
                dbc.NavLink("分域分析", href="/domain_page", active="exact"),
                dbc.NavLink("杠杆分析", href="/leverage_page", active="exact"),
                dbc.NavLink("MF热力图", href="/mf_parameter_plane", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
        html.Hr(),
        html.P("策略", className="lead"),
        dcc.Store(id='select-strategy-store'),
        dcc.Store(id='transaction-store'),
        dcc.Store(id='stg-drawback-store'),
        dcc.Dropdown(
            id='select-strategy',
            options=[{'label': strategy, 'value': strategy} for strategy in all_strategy],
            placeholder="选择策略",
            value=strategy_name,
            clearable=False,
        ),
        dcc.Loading(
            id="loading-strategy",
            type="default",  # 可选："default", "circle", "dot", "graph"
            children=html.Div(id="loading-strategy-finish")
        ),
        # 这里显示页面内容
    ],
    style=SIDEBAR_STYLE,
)

# 定义主布局
content = html.Div(id="page-content", style=CONTENT_STYLE, children=[dash.page_container])

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])


# # 主页布局
# home_layout = html.Div([
#     html.H1('Welcome to the Dash Multi-Page App'),
#     html.P('Navigate to other pages using the menu above.')
# ])

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page_content(pathname: str):
    if pathname in (None, "/"):
        return html.Div("好好研究策略，相聚马尔代夫",
                        style={'font-family': 'Arial, sans-serif', 'font-size': '58px', 'display': 'flex',
                               'justify-content': 'center', 'align-items': 'center', 'height': '100vh'})
    # 其它路径：让 Dash Pages 渲染对应页面
    return dash.page_container


@app.callback(      Output('select-strategy-store', 'data'),
              Input("select-strategy", "value"))
def refresh_select_stg(select_strategy: str):
    # 其它路径：让 Dash Pages 渲染对应页面
    return select_strategy


# 设置默认页面
app.validation_layout = html.Div([
    content,
    sidebar,
])

# 运行应用
if __name__ == '__main__':
    try:
        app.run(debug=False, port=8089)
    finally:
        cleanup_analysis_cache()
        pass
