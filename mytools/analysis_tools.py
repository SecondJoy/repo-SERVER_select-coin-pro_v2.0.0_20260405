#JoyAdded 马代代码
import concurrent.futures
import datetime
from pathlib import Path
from typing import List, Tuple, Any

import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.subplots import make_subplots
from tqdm import tqdm

import config
from core.utils.factor_hub import FactorHub
from core.utils.log_kit import get_logger
from mytools.constants import DEFAULT_PERIOD_CONFIG
from mytools.models import OHLCData

logger = get_logger()


class TransactionAnalyzer:

    def __init__(self):
        self._spot_data = pd.read_pickle(config.spot_path)
        self._swap_data = pd.read_pickle(config.swap_path)

    def calculate_factor(self, source: pd.DataFrame, factors: List[Tuple[str, int]], start_suffixes: str = "_start",
                         end_suffixes: str = "_end") -> pd.DataFrame:

        symbol_group = source.groupby(by="symbol")

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.job_num) as executor:
            futures = [executor.submit(self.cal_factor_one_symbol, data, symbol, factors, start_suffixes, end_suffixes)
                       for
                       symbol, data in symbol_group]
            res = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                res.append(future.result())
        return pd.concat(res, ignore_index=True)

    def get_data(self, symbol) -> Any | None:
        if symbol in self._swap_data:
            return self._swap_data[symbol]
        elif symbol in self._spot_data:
            return self._spot_data[symbol]
        else:
            logger.error(f"没有发现{symbol}的数据")
            return None

    def test(self, transaction_data: pd.DataFrame, symbol) -> int:
        print("ok")
        return 1

    def cal_factor_one_symbol(self, transaction_data: pd.DataFrame, symbol, factors: List[Tuple[str, int]],
                              start_suffixes: str = "_start",
                              end_suffixes: str = "_end") -> pd.DataFrame:
        data = self.get_data(symbol)
        for f in factors:
            factor_name = f[0]
            hold_hour = f[1]
            temp = f'{factor_name}_{hold_hour}'
            addition_factor = FactorHub.get_by_name(factor_name)
            data = addition_factor.signal(data, hold_hour, temp)
            transaction_data = pd.merge(transaction_data, data, left_on='start_time', right_on='candle_begin_time',
                                        how='left', suffixes=('', start_suffixes))
            transaction_data = pd.merge(transaction_data, data, left_on='end_time', right_on='candle_begin_time',
                                        how='left', suffixes=('', end_suffixes))
        return transaction_data


_PERIOD_REPORT_HEADER = [
    "category", "时期", "开始时间", "结束时间", "开始数值", "结束数值", "benchmark开始", "benchmark结束",
    "变动百分比", "benchmark百分比"
]


class EquityCurveAnalyzer:
    def __init__(self, period_config: dict = None):
        if period_config is None:
            period_config = DEFAULT_PERIOD_CONFIG
        self._period_config = period_config

    def export_period_compare_report(self, equity: pd.Series, benchmark: pd.Series = None,
                                     plot_folder: Path = None,
                                     transactions: pd.DataFrame = None,
                                     factor: tuple[str, int] = None) -> pd.DataFrame:
        if benchmark is None:
            ohlc = OHLCData()
            btc = ohlc.get_data_by_symbol("BTC-USDT")
            btc = btc.set_index("candle_begin_time")
            benchmark = btc['close']

        factor_val = None
        factor_name = None
        factor_at_open = None
        factor_at_close = None
        if factor is not None:
            factor_name = f"{factor[0]}_{factor[1]}"
            all_factor_val = OHLCData().calculate_factors([factor])
            factor_val = all_factor_val.groupby("candle_begin_time")[factor_name].mean()
            if transactions is not None:
                factor_when_transaction = pd.merge(transactions, all_factor_val,
                                                   left_on=['start_time', 'symbol'],
                                                   right_on=['candle_begin_time', 'symbol'], how='left')
                factor_when_transaction.to_csv(plot_folder / f"transaction_at_open_with_factors.csv",index=False,encoding="utf-8-sig")
                factor_at_open = factor_when_transaction.groupby("start_time")[factor_name].mean()

                factor_when_transaction_close = pd.merge(transactions, all_factor_val,
                                                   left_on=['end_time', 'symbol'],
                                                   right_on=['candle_begin_time', 'symbol'], how='left')
                factor_at_close = factor_when_transaction_close.groupby("end_time")[factor_name].mean()
                factor_when_transaction_close.to_csv(plot_folder / f"transaction_at_close_with_factors.csv",index=False,encoding="utf-8-sig")

        res = []
        for category, v in self._period_config.items():
            for period, time_period in self._period_config[category].items():
                start = datetime.datetime.strptime(time_period[0], '%Y-%m-%d')
                end = datetime.datetime.strptime(time_period[1], '%Y-%m-%d')

                data = equity.loc[start:end]
                benchmark_period = benchmark[start:end]

                equity_start = max(data.iloc[0], 1e-10)
                equity_end = data.iloc[-1]
                equity_pct = equity_end / equity_start - 1
                benchmark_start = max(benchmark_period.iloc[0], 1e-10)
                benchmark_end = benchmark_period.iloc[-1]
                benchmark_pct = benchmark_end / benchmark_start - 1

                export_data = [category, period, start, end,
                               equity_start, equity_end, benchmark_start, benchmark_end,
                               equity_pct, benchmark_pct]
                if factor_val is not None:
                    factor_mean = factor_val[start:end].mean()
                    export_data.append(factor_mean)
                if factor_at_open is not None:
                    transaction_open_mean = factor_at_open[start:end].mean()
                    transaction_close_mean = factor_at_close[start:end].mean()
                    export_data.append(transaction_open_mean)
                    export_data.append(transaction_close_mean)

                res.append(export_data)

                if plot_folder is not None:
                    _draw_equity_curve_plotly(equity, benchmark, start, end, plot_folder, category, period,
                                              factor_val=factor_val,
                                              open_factor_val=factor_at_open, close_factor_val=factor_at_close)

        header = _PERIOD_REPORT_HEADER.copy()
        if factor_val is not None:
            header.append(f"all_{factor_name}_mean")
        if factor_at_open is not None:
            header.append(f"{factor_name}_open_mean")
            header.append(f"{factor_name}_close_mean")

        return pd.DataFrame(res, columns=header)


def _draw_equity_curve_plotly(data: pd.Series, benchmark: pd.Series,
                              start: datetime.datetime, end: datetime.datetime, export_path: Path,
                              category: str, period: str,
                              factor_val: pd.Series = None,
                              open_factor_val: pd.Series = None,
                              close_factor_val: pd.Series = None,
                              pic_size=None):
    if pic_size is None:
        pic_size = [1500, 800]
    plot_start = start - datetime.timedelta(days=30)
    plot_end = end + datetime.timedelta(days=30)
    draw = data[plot_start:plot_end]
    # 绘制左轴数据
    rows = 1
    specs = [[{"secondary_y": True}]]
    row_height = [6]
    if factor_val is not None and open_factor_val is not None:
        rows = 4
        specs = [[{"secondary_y": True}]] + [[{}]] * 3
        row_height = [6, 1, 1, 1]
    if factor_val is not None and close_factor_val is None:
        rows = 2
        specs = [[{"secondary_y": True}]] + [[{}]]
        row_height = [6, 1]

    fig = make_subplots(rows=rows,
                        cols=1,
                        shared_xaxes=True,
                        specs=specs,
                        row_heights=row_height)

    fig.add_trace(go.Scatter(x=draw.index, y=draw.values, name="资金曲线", ))

    benchmark_draw = benchmark[plot_start:plot_end]
    fig.add_trace(go.Scatter(x=benchmark_draw.index, y=benchmark_draw.values, name="基线", yaxis='y2'),
                  secondary_y=True)
    formatted_start = start.strftime("%Y年%m月%d日")
    formatted_end = end.strftime("%Y年%m月%d日")
    title = f"{category}-{period}-{formatted_start}-{formatted_end}"

    max_value = draw.max() * 1.1
    fig.add_shape(type='line',
                  x0=start, y0=0, x1=start, y1=max_value,
                  line=dict(color='red', width=2))
    fig.add_shape(type='line',
                  x0=end, y0=0, x1=end, y1=max_value,
                  line=dict(color='red', width=2))
    fig.update_layout(
        template="none", title_text=title,
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

    if factor_val is not None:
        draw_factor = factor_val[plot_start:plot_end]
        fig.add_trace(
            go.Scatter(x=draw_factor.index, y=draw_factor.values, name="因子平均值", line=dict(width=2)),
            row=2, col=1)
        fig.update_yaxes(title_text="因子平均值", row=2, col=1)

    if open_factor_val is not None:
        draw_open_factor = open_factor_val[plot_start:plot_end]
        draw_close_factor = close_factor_val[plot_start:plot_end]
        fig.add_trace(
            go.Scatter(x=draw_open_factor.index, y=draw_open_factor.values, name="因子开仓平均值", line=dict(width=2)),
            row=3, col=1)
        fig.add_trace(
            go.Scatter(x=draw_close_factor.index, y=draw_close_factor.values, name="因子平仓平均值",
                       line=dict(width=2)),
            row=4, col=1)
        fig.update_yaxes(title_text="因子开仓平均值", row=3, col=1)
        fig.update_yaxes(title_text="因子平仓平均值", row=4, col=1)

    html_export = str(export_path / f"{title}.html")
    result = plot(figure_or_data=fig, include_plotlyjs=True, output_type='file', filename=html_export, auto_open=False)

    return result
