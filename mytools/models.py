import asyncio
import collections
import concurrent.futures
import dataclasses
import datetime
import importlib
import os
import re
import shutil
import threading
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Set, Dict, Literal
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from joblib import Parallel, delayed
from plotly.subplots import make_subplots
from tqdm import tqdm
import config
from core.backtest import step2_load_data
from core.model.backtest_config import BacktestConfig
from core.model.strategy_config import StrategyConfig
from core.utils.factor_hub import FactorHub, DummyFactor
from core.utils.log_kit import get_logger
from core.utils.path_kit import get_folder_path, get_file_path
from mytools.func import pct_str_to_float, top_drawdowns, cache_data

logger = get_logger()

BASE_MERGE_COL = ['candle_begin_time', 'symbol']


class CAPData:
    _instance = None
    _lock = threading.Lock()

    HEAD_COLUMN = [
        "candle_begin_time",
        "symbol",
        "id",
        "name",
        "date_added",
        "max_supply",
        "circulating_supply",
        "total_supply",
        "usd_price",
        "max_mcap",  # 理论总市值
        "circulating_mcap",  # 流通市值
        "total_mcap"  # 实际总市值
    ]

    def __new__(cls, source_path: Path = None):
        if source_path is None:
            source_path = Path(config.data_source_dict['coin-cap'][1])

        if cls._instance is None:
            with cls._lock:  # 确保线程安全
                if cls._instance is None:  # 再次检查
                    cls._instance = super(CAPData, cls).__new__(cls)
                    cls._instance.initialize(source_path)  # 初始化数据
        return cls._instance

    def get_data_by_symbol(self, symbol: str) -> pd.DataFrame:
        return self.data[self.data.symbol == symbol]

    def initialize(self, source_path: Path = None):
        # 这里进行数据的初始化
        cache = source_path / "cache.parquet"
        if cache.exists():
            logger.info(f"loading cap data  from {cache}")
            self.data = pd.read_parquet(cache)
            return

        csv_files = source_path.glob('*.csv')
        # 创建事件循环
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cap_data = []
        job_num = max(os.cpu_count() - 1, 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=job_num) as executor:
            futures = [executor.submit(_read_one_cap, csv)
                       for csv in csv_files]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                cap_data.append(future.result())
        res: pd.DataFrame = pd.concat(cap_data, ignore_index=True)

        time_group = res.groupby('candle_begin_time')
        res['circulating_mcap_rank'] = time_group['circulating_mcap'].rank(method='min')
        res['total_mcap_rank'] = time_group['total_mcap'].rank(method='min')

        self.data = res

        self.data.to_parquet(cache)
        loop.close()


def _read_one_cap(csv_path: Path):
    # 模拟一个异步任务，随机等待一段时间
    csv = pd.read_csv(csv_path, parse_dates=["candle_begin_time"], encoding="GBK", skiprows=1)
    expanded_rows = []
    for index, row in csv.iterrows():
        date = row['candle_begin_time']
        # 获取当前行的所有值
        values = row.to_dict()  # 将行转换为字典
        for hour in range(24):
            # 创建一个新的字典，替换日期为当前小时
            new_row = values.copy()  # 复制当前行的所有值
            new_row['candle_begin_time'] = date + pd.Timedelta(hours=hour)  # 替换日期为小时
            expanded_rows.append(new_row)
    return pd.DataFrame(expanded_rows)  # 返回结果


@dataclasses.dataclass
class AdditionInfo:
    hold_hour: int
    shift_num: int = None
    shift_col_name: str = None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=['symbol'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


class OHLCData:
    def __init__(self, use_spot: bool = True):
        self.initialize(use_spot=use_spot)

    def initialize(self, use_spot: bool = True):
        if hasattr(self, '_use_spot') and self._use_spot == use_spot:
            return
        logger.info("refresh ohlc data with spot:{}".format(use_spot))
        self._use_spot = use_spot
        my_config = BacktestConfig("test")
        my_config.is_use_spot = True  # 强制设置为 True，加载所有数据
        if use_spot:
            my_config.select_scope_set = {'mix'}
        else:
            my_config.select_scope_set = {'swap'}
        step2_load_data(my_config)
        cache_path = Path(get_folder_path('data', 'cache')) / "all_candle_df_list.pkl"
        data = pd.read_pickle(cache_path)
        self._data = {f"{i['symbol'].iloc[-1]}_{i['is_spot'].iloc[-1]}": clean_data(i) for i in data}

    def symbols(self, with_spots=True) -> Set[str]:
        swap = pd.read_pickle(config.spot_path)
        swaps_symbols = set(swap.keys())
        if with_spots:
            logger.debug("开始使用现货")
            spot = pd.read_pickle(config.swap_path)
            spots_symbols = set(spot.keys())
            return spots_symbols.union(swaps_symbols)
        return swaps_symbols

    def calculate_factors(self, factors: List[tuple[str, int]], hold_info: AdditionInfo = None,
                          use_spots: bool = True) -> pd.DataFrame:

        self.initialize(use_spots)
        symbols = self._data.keys()
        if use_spots:
            symbols = [_.replace('_0', '').replace('_1', '') for _ in symbols]
            symbols = list(set(symbols))
        else:
            symbols = [_.replace('_0', '').replace('_1', '') for _ in symbols if _.endswith(f'_0')]

        # 这一段代码很垃圾，先算，然后再处。
        time_factor = []
        sections_factor = {}
        for factor_name, hold_hour in factors:
            factor = FactorHub.get_by_name(factor_name)
            if factor.is_cross:
                col_name = f"{factor_name}_{hold_hour}"
                sections_factor[col_name] = self.cal_section_factor(factor, hold_hour)[
                    ['candle_begin_time', 'symbol', 'is_spot', col_name]]
            else:
                time_factor.append((factor_name, hold_hour))

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.job_num) as executor:
            futures = [executor.submit(self.cal_factor_one_symbol, symbol, time_factor, hold_info, use_spots)
                       for
                       symbol in symbols]
        res = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            result = future.result()
            if not result.empty:
                res.append(result)
        res = pd.concat(res, ignore_index=True)
        for section_factor_name in sections_factor:
            section_df = sections_factor[section_factor_name]
            res = pd.merge(res, section_df, how='inner', on=['candle_begin_time', 'symbol', 'is_spot'])
        return res

    @staticmethod
    def get_factor_by_name(factor_name) -> DummyFactor:
        try:
            try:
                module = 'factors'
                factor_module = importlib.import_module(f"{module}.{factor_name}")
            except ModuleNotFoundError:
                module = 'sections'
                factor_module = importlib.import_module(f"{module}.{factor_name}")

            # 创建一个包含模块变量和函数的字典
            factor_content = {
                name: getattr(factor_module, name) for name in dir(factor_module)
                if not name.startswith("__")
            }
            factor_content['is_cross'] = module == 'sections'

            # 创建一个包含这些变量和函数的对象
            factor_instance = type(factor_name, (), factor_content)

            # 缓存策略对象
            FactorHub._factor_cache[factor_name] = factor_instance

            return factor_instance
        except ModuleNotFoundError:
            raise ValueError(f"Factor {factor_name} not found.")
        except AttributeError:
            raise ValueError(f"Error accessing factor content in module {factor_name}.")

    def cal_section_factor(self, section_factor: DummyFactor, n: int, use_spot: bool = True) -> pd.DataFrame:
        """
        完成全新的截面因子。

        """
        col_name = f"{section_factor.__name__}_{n}"

        def do_cal() -> pd.DataFrame:
            merge_col = BASE_MERGE_COL + [col_name]
            factor_list = section_factor.get_factor_list(n)
            data = OHLCData().calculate_factors(factor_list, use_spots=use_spot)
            return section_factor.signal(data, n, col_name)[merge_col]

        factor_vale = cache_data(do_cal, name=col_name)

        return pd.merge(self.all_in_one(), factor_vale, how='inner', on=BASE_MERGE_COL)

    def cal_factor_one_symbol(self, symbol: str, factor_list: List[tuple[str, int]],
                              hold_info: AdditionInfo = None, use_spot: bool = True,
                              section_data: pd.DataFrame | None = None) -> pd.DataFrame:
        try:
            data = self.get_data_by_symbol(symbol, use_spot)
        except ValueError as e:
            print(f'{symbol} {"现货" if use_spot else "合约"} 数据不存在')
            return pd.DataFrame()

        for factor_name, hold_period in factor_list:
            factor_col_name = f"{factor_name}_{hold_period}"
            factor = FactorHub.get_by_name(factor_name)

            if factor.is_cross:
                res = self.cal_section_factor(factor, hold_period, use_spot=use_spot)
                return res[res['symbol'] == symbol]

            if hasattr(factor, 'extra_data_dict') and factor.extra_data_dict:
                from core.utils.functions import merge_data
                for data_name in factor.extra_data_dict.keys():
                    extra_data_dict = merge_data(data, data_name, factor.extra_data_dict[data_name])
                    for extra_data_name, extra_data_series in extra_data_dict.items():
                        data[extra_data_name] = extra_data_series.values
            data = factor.signal(data, hold_period, factor_col_name)

        if hold_info is not None:
            """
            因为原始数据从0点开始。所以比较方便做这方面的事情
            """
            hold_hour = hold_info.hold_hour
            res = []

            df = data.copy()
            # 这里算 offset 不够严谨，目前的两个页面走不到这个逻辑，暂不处理
            df['offset'] = data.index % hold_hour

            if hold_info.shift_col_name is not None:
                df[hold_info.shift_col_name] = df['close'].shift(hold_info.shift_num)
            df.dropna(subset=["symbol"], inplace=True)
            res.append(df)

            data = df

            data.sort_values(by='candle_begin_time', inplace=True)
            return data

        return data

    def get_data_by_symbol(self, symbol: str, _: bool = True) -> pd.DataFrame:
        """
        默认首先取swap的数据，然后再取spot的数据
        :param _:
        :param symbol:
        :return:
        """
        symbol_name = f"{symbol}_{1 if _ else 0}"
        if symbol_name in self._data:
            return self._data[symbol_name].copy()
        symbol_name = f"{symbol}_{0 if _ else 1}"
        if symbol_name in self._data:
            return self._data[symbol_name].copy()
        raise ValueError(f"{symbol_name}不存在")

    def all_in_one(self) -> pd.DataFrame:
        return pd.concat(self._data.values(), ignore_index=True)

    def refresh_close_and_profit(self, trans: pd.DataFrame) -> pd.DataFrame:
        all_in_one = self.all_in_one()[["symbol", "candle_begin_time", "is_spot", "close"]]
        trans = pd.merge(trans, all_in_one, how='left',
                         left_on=["symbol", "is_spot", "start_time"],
                         right_on=["symbol", "is_spot", "candle_begin_time"])
        trans = pd.merge(trans, all_in_one, how='left',
                         left_on=["symbol", "is_spot", "end_time"],
                         right_on=["symbol", "is_spot", "candle_begin_time"],
                         suffixes=('_start', '_end'))

        trans = trans[
            ["symbol", "start_time", "end_time", "close_start", "close_end", "direction", "offset", "is_spot"]]

        trans = trans.rename(
            columns={"close_start": "start_close",
                     "close_end": "end_close", }
        )
        trans['profit'] = (trans['end_close'] / trans['start_close'] - 1) * trans['direction']
        return trans


@lru_cache(maxsize=2)
def get_ohlc_data(use_spot=True):
    return OHLCData(use_spot)


@dataclasses.dataclass
class AnalysisTransactions:
    symbol: str
    start_date: datetime.datetime
    start_close: float
    is_spot: int
    direction: int
    offset: Optional[int] = 0
    end_date: Optional[datetime.datetime] = None
    end_close: Optional[float] = None
    strategy_name: str = None

    @property
    def profit(self) -> float:
        return (self.end_close / self.start_close - 1) * self.direction


class BackTestingResults:

    def __init__(self, name: str = None, root: Path = config.backtest_path):
        if name is None:
            name = config.backtest_name
        self._bt_name = name
        self._root = root / name
        self._cache = root / "cache"
        self.transactions_root = self._root / "transactions"

    @property
    def export_folder(self):
        return self._root

    @property
    def select_coins(self) -> pd.DataFrame:
        select_coin_path = self._root / "选币结果.pkl"
        return pd.read_pickle(select_coin_path)

    @property
    def rtn(self) -> pd.DataFrame:
        rtn_path = self._root / "策略评价.csv"
        csv = pd.read_csv(rtn_path, encoding="utf-8-sig", index_col=0)
        csv.rename(columns={'0': 0}, inplace=True)
        return csv

    @property
    def config(self) -> BacktestConfig:
        select_coin_path = self._root / "config.pkl"
        return pd.read_pickle(select_coin_path)

    @property
    def equity(self) -> pd.DataFrame:
        return pd.read_csv(self._root / "资金曲线.csv", parse_dates=["candle_begin_time"], encoding="utf-8-sig",
                           index_col=0)

    def drawdown_list(self, top_n: int = 5) -> pd.DataFrame:
        # 回撤列表
        equity = self.equity.set_index("candle_begin_time")

        # Convert equity curve to returns
        returns = equity['equity'].pct_change()
        max_dd = top_drawdowns(returns, top_n)
        return max_dd

    @lru_cache(maxsize=2)
    def list_transactions(self, merge_trans: bool = True, use_cache=False) -> List[AnalysisTransactions]:
        """

        :param merge_trans: 是否要把transaction做合并。
        :param use_cache:  是否使用缓存
        :return:
        """
        stg_list = self.config.strategy_list
        select_coins = self.select_coins
        cache_path = self.transactions_root / "transactions.pkl"
        if use_cache:
            if cache_path.exists():
                logger.info(f"loading transactions from {cache_path}")
                return pd.read_pickle(cache_path)

        res = []

        for stg in stg_list:
            cache_dict = collections.defaultdict(list)
            hold_hour_str = stg.hold_period
            hold_time = int(hold_hour_str[:-1])
            hold_time_type = hold_hour_str[-1]
            if hold_time_type.upper() == "H":
                delta_time = datetime.timedelta(hours=hold_time)
            else:
                delta_time = datetime.timedelta(days=hold_time)
            stg_select_coins = select_coins.loc[select_coins["strategy"] == stg.name]
            stg_coins_groups = stg_select_coins.groupby(['symbol', 'offset'], observed=False)

            for (symbol, offset), group in tqdm(stg_coins_groups, desc="分析transactions"):
                transactions = to_transactions(group, delta_time, symbol, self._bt_name)
                cache_dict[symbol].extend(transactions)
            if merge_trans:
                results = Parallel(n_jobs=config.job_num)(
                    delayed(merge_transactions)(l, hold_time) for _, l in
                    tqdm(cache_dict.items(), desc="merge transactions"))
                res.extend([item for sublist in results for item in sublist])
            else:
                res.extend([item for sublist in cache_dict.values() for item in sublist])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(res, cache_path)
        return res

    def transaction_df(self, merge_trans: bool = True, use_cache=False) -> pd.DataFrame:
        transactions = self.list_transactions(merge_trans=merge_trans, use_cache=use_cache)
        data = {
            "symbol": [t.symbol for t in transactions],
            "start_time": [t.start_date for t in transactions],
            "end_time": [t.end_date for t in transactions],
            "start_close": [t.start_close for t in transactions],
            "end_close": [t.end_close for t in transactions],
            "direction": [t.direction for t in transactions],
            "offset": [t.offset for t in transactions],
            "is_spot": [t.is_spot for t in transactions],
            "profit": [t.profit for t in transactions],
            "strategy_name": [t.strategy_name for t in transactions]
        }

        res = pd.DataFrame(data)
        res = OHLCData().refresh_close_and_profit(res)
        # 参数覆盖会出现重复的记录
        res = res.drop_duplicates(subset=['symbol', 'start_time', 'end_time', 'offset', 'profit'])
        res.sort_values("profit", inplace=True, ascending=False)
        res['strategy'] = self._bt_name
        return res


def merge_transactions(transactions: List[AnalysisTransactions], hold_hours: int, merge_type: str = "max") -> List[
    AnalysisTransactions]:
    """
    因为在实际的transaction中，相临近的offset可能有相同的，但是在分析的时候，就会相同的的transactions
    输入的transactions需要相同的symbol
    :param transactions: 需要merge的transactions
    :param hold_hours: 持仓时间
    :param merge_type: merge的模式，我现在想到有遗下集中。现在只是实现max模式
        max:  连着的 transaction取收益最大的一个。如果相同，取第一个
        long：最长,  在transaction中，取持续时间最大的一个。如果相同，取第一个
        combine：start和end都取最大的一个。
    :return:
    """
    if len(transactions) <= 1:
        return transactions

    sorted_transaction = sorted(transactions, key=lambda trans: trans.start_date)
    res = []
    prev = sorted_transaction[0]
    ready_to_merge = [prev]
    cache = sorted_transaction[1:]
    for t in cache:
        gap_hour = int((t.start_date - prev.start_date).total_seconds() / 3600)
        # (next_number - current) % (end - start + 1)  这个公式的简化。
        gap_offset = (t.offset - prev.offset) % hold_hours
        if gap_offset == gap_hour and t.direction == prev.direction:
            """
            因为计算出来的transaction,相同的offset，在下一个周期会做合并。所以这里没有考虑这里做特殊处理。
            但是可能有一些情况
            比如一个3H。
            第一个周期的offset，o1,o2,o3都有
            第二个周期的offset, 空，o2,o3
            那么按道理里说。这里应该会成为一个transaction。
            但是这里，会成merge成两个。太复杂了。不想考虑了。
            """
            ready_to_merge.append(t)
        else:
            max_t = max(ready_to_merge, key=lambda tran: tran.profit)
            res.append(max_t)
            ready_to_merge = [t]
        prev = t
    if len(ready_to_merge) > 0:
        max_t = max(ready_to_merge, key=lambda tran: tran.profit)
        res.append(max_t)
    return res


def to_transactions(df: pd.DataFrame, hold_period: datetime.timedelta, symbol: str, stg_name: str) -> List[
    AnalysisTransactions]:
    res = []
    curr_transactions = None
    prev_time = None
    for index, row in df.iterrows():
        cbt = row['candle_begin_time']
        close = row['close']
        next_close = 0
        direction = row['方向']
        offset = row['offset']
        is_spot = row['is_spot']
        next_date = cbt + hold_period

        if curr_transactions is None:
            curr_transactions = _new_transaction(cbt, close, next_date, next_close, direction, is_spot, symbol, offset)
            prev_time = cbt
            continue

        if direction != curr_transactions.direction:
            res.append(curr_transactions)
            curr_transactions = _new_transaction(cbt, close, next_date, next_close, direction, is_spot, symbol, offset)
            continue

        if (cbt - prev_time) > hold_period:
            res.append(curr_transactions)
            curr_transactions = _new_transaction(cbt, close, next_date, next_close, direction, is_spot, symbol, offset)
        else:
            curr_transactions.end_close = close
            curr_transactions.end_date = next_date
        prev_time = cbt
        curr_transactions.strategy_name = stg_name

    res.append(curr_transactions)

    return res


def _new_transaction(start_date, start_close, end_date, end_close, direction, is_spot, symbol, offset):
    curr_transactions = AnalysisTransactions(
        symbol=symbol,
        start_date=start_date,
        start_close=start_close,
        end_date=end_date,
        end_close=end_close,
        direction=direction,
        is_spot=is_spot,
        offset=offset
    )
    return curr_transactions


def plotly_transactions(df, show_list, back_hour, enter_time, exit_time, symbol,
                        mark_point_list=None, hold_text=None, title=None, return_fig=False,
                        addition_factor: List[str] = None, download_png_name: str = None):
    has_addition_factor = False
    other_factors = set()
    other_factors_length = 0
    if addition_factor is not None:
        other_factors = set(addition_factor) & set(df.columns)
        other_factors_length = len(other_factors)
        if other_factors_length > 0:
            has_addition_factor = True

        # 创建带有双Y轴的图表
    if has_addition_factor:
        rows = 2 + other_factors_length
        specs = [[{"secondary_y": True}], [{}]] + [[{}]] * other_factors_length
        subplot_titles = [title, None] + [None] * other_factors_length
        row_height = [4, 1] + [1] * other_factors_length
        fig = make_subplots(rows=rows,
                            cols=1,
                            shared_xaxes=True,
                            # vertical_spacing=0.01,
                            vertical_spacing=0.0,
                            specs=specs,  # 添加次级Y轴
                            subplot_titles=subplot_titles,
                            # row_heights=[4, 3])
                            row_heights=row_height)  # 修改两个区域的占比
    else:
        fig = make_subplots(rows=2,
                            cols=1,
                            shared_xaxes=True,
                            # vertical_spacing=0.01,
                            vertical_spacing=0.0,
                            specs=[[{"secondary_y": True}], [{}]],  # 添加次级Y轴
                            subplot_titles=[title, None],
                            # row_heights=[4, 3])
                            row_heights=[4, 1])

        # 添加K线图
    fig.add_trace(
        go.Candlestick(x=df['candle_begin_time'],
                       open=df['open'],
                       high=df['high'],
                       low=df['low'],
                       close=df['close'],
                       name="K线",
                       increasing=dict(line_color='red', fillcolor='red'),
                       decreasing=dict(line_color='green', fillcolor='green')),
        secondary_y=False,
    )

    if has_addition_factor:
        row_start = 3
        for factor_name in other_factors:
            fig.add_trace(
                go.Scatter(x=df['candle_begin_time'], y=df[factor_name], name=factor_name, line=dict(width=2)),
                row=row_start, col=1)
            row_start = row_start + 1

    # 根据show_list添加显示的线
    for col in show_list:
        fig.add_trace(
            go.Scatter(x=df['candle_begin_time'], y=df[col], name=col, line=dict(width=2, color='blue')),
            secondary_y=True
        )

    # 添加成交额图
    volume_color = ['#3D9970' if close_price >= open_price else '#FF4136' for open_price, close_price in
                    zip(df['open'], df['close'])]
    volume_bar = go.Bar(x=df['candle_begin_time'],
                        y=df['quote_volume'],
                        name='成交额',
                        marker_color=volume_color
                        )
    fig.add_trace(volume_bar, row=2, col=1)

    # 设置图表标题
    # fig.update_layout(title=f'{symbol}'"走势线图和"f'{factor}', xaxis_rangeslider_visible=False)
    fig.update_layout(xaxis_rangeslider_visible=False, margin=dict(t=40, r=0, b=0))

    # 设置Y轴标题
    fig.update_yaxes(title_text="K线价格", secondary_y=False)
    fig.update_yaxes(title_text="成交额", row=2, col=1)
    fig.update_yaxes(title_text=f'因子值', secondary_y=True)
    if has_addition_factor:
        row_num = 3
        for factor_name in other_factors:
            fig.update_yaxes(title_text=factor_name, row=row_num, col=1)
            row_num = row_num + 1

    if hold_text is not None:
        max_time = df['candle_begin_time'].max()
        # 添加垂直虚线和背景色
        shapes = [
            dict(type="line", x0=enter_time, y0=0, x1=enter_time, y1=1, xref='x', yref='paper',
                 line=dict(color="green", width=2, dash="dash")),
            dict(type="line", x0=exit_time, y0=0, x1=exit_time, y1=1, xref='x', yref='paper',
                 line=dict(color="blue", width=2, dash="dash")),
            dict(type="line", x0=enter_time - pd.Timedelta(hours=back_hour), y0=0,
                 x1=enter_time - pd.Timedelta(hours=back_hour), y1=1, xref='x', yref='paper',
                 line=dict(color="red", width=2, dash="dash")),
            dict(type="rect", x0=enter_time, y0=0, x1=exit_time, y1=1, xref='x', yref='paper', fillcolor="#D5E1DF",
                 opacity=1.0, layer="below", line_width=0),
            dict(type="rect", x0=enter_time - pd.Timedelta(hours=back_hour), y0=0, x1=enter_time, y1=1, xref='x',
                 yref='paper', fillcolor="#E6DAE0", opacity=1.0, layer="below", line_width=0),
            dict(type="rect", x0=exit_time, y0=0, x1=max_time, y1=1, xref='x', yref='paper', fillcolor="#E6DAE0",
                 opacity=1.0, layer="below", line_width=0),
        ]
        fig.update_layout(shapes=shapes)

        # 在虚线旁边添加文字
        enter_x = enter_time - pd.Timedelta(hours=round(back_hour / 2))
        hold_x = enter_time + (exit_time - enter_time) / 2
        exit_x = exit_time + (exit_time - enter_time) / 2
        annotations = [
            dict(text='', x=0.5, y=0.99, xref='paper', yref='paper', showarrow=False),
            dict(text="因子计算区间", x=enter_x, y=0.99, xref='x', yref='paper', showarrow=False, font=dict(size=14)),
            dict(text=hold_text, x=hold_x, y=0.99, xref='x', yref='paper', showarrow=False, font=dict(size=14)),
            dict(text="平仓后", x=exit_x, y=0.99, xref='x', yref='paper', showarrow=False, font=dict(size=14))
        ]
        if mark_point_list is not None:
            annotations += mark_point_list
        # , font=dict(size=15,)
        fig.update_layout(annotations=annotations)

    # 更新布局设置
    grid_color = '#fafafa'
    fig.update_layout(
        xaxis=dict(
            showgrid=True,  # 不显示垂直网格线
            gridcolor=grid_color  # 设置垂直网格线的颜色
        ),
        yaxis=dict(
            showgrid=False,  # 不显示水平网格线
            gridcolor=grid_color  # 设置水平网格线的颜色
        ),
        yaxis2=dict(
            showgrid=True,  # 显示水平网格线
            gridcolor=grid_color  # 设置水平网格线的颜色
        )
    )

    # 设置volume_bar的x轴和y轴的网格颜色
    fig.update_xaxes(showgrid=True, row=2, col=1, gridcolor=grid_color)
    fig.update_yaxes(showgrid=True, row=2, col=1, gridcolor=grid_color)

    # fig.update_layout(
    #     hovermode='x',
    #     xaxis=dict(showspikes=True, spikecolor="gray", spikesnap="cursor", spikemode="across+toaxis"),
    #     yaxis=dict(showspikes=False)
    # )

    if return_fig:
        # html_str = fig.to_html(full_html=False, default_height='50%')
        # with open(file_path, "w", encoding="utf-8") as file:
        #     file.write(html_str)
        if download_png_name is not None:
            plot_config = {
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": download_png_name,
                    "height": None,
                    "width": None,
                    "scale": 1
                },
                "displayModeBar": True
            }
            setattr(fig, "_plotly_config", plot_config)
        return fig
    else:
        # 显示图表
        fig.show()
    return None


@dataclasses.dataclass
class SearchResultEntry:
    bt_config: BacktestConfig
    returns: float
    drawdown: float
    drawdown_hours: float
    annual_drawdown_ratio: float
    year_returns: dict[str, float]
    quarter_returns: dict[str, float]
    month_returns: dict[str, float]

    @property
    def factors(self) -> Dict[str, set]:
        return self.bt_config.factor_params_dict

    @property
    def first_st_long_select_num(self) -> int | float | tuple:
        return self.bt_config.strategy_list[0].long_select_coin_num

    @property
    def first_st_use_spot(self) -> bool:
        return self.bt_config.strategy_list[0].is_use_spot

    @property
    def first_st_short_select_num(self) -> int | float | tuple:
        return self.bt_config.strategy_list[0].short_select_coin_num

    @property
    def first_st_hold_period(self) -> str:
        return self.bt_config.strategy_list[0].hold_period


def load_one_search_parameter(path: Path) -> SearchResultEntry:
    config_data = pd.read_pickle(path / "config.pkl")

    stats_temp = pd.read_csv(path / "策略评价.csv", encoding='utf-8')
    stats_temp.columns = ['evaluation_indicator', 'value']
    stats_temp = stats_temp.set_index('evaluation_indicator')
    all_return = stats_temp.loc["累积净值", 'value']
    drawdown = pct_str_to_float(stats_temp.loc["最大回撤", 'value']) / 100
    date_format = "%Y-%m-%d %H:%M:%S"
    max_drawdown_start = datetime.datetime.strptime(stats_temp.loc["最大回撤开始时间", 'value'], date_format)
    max_drawdown_end = datetime.datetime.strptime(stats_temp.loc["最大回撤结束时间", 'value'], date_format)
    drawdown_hours = (max_drawdown_end - max_drawdown_start).total_seconds() / 3600
    annual_drawdown_ratio = stats_temp.loc["最大回撤", 'value']

    years_return_df = pd.read_csv(path / "年度账户收益.csv", encoding="utf-8")
    year_return = dict()
    for _, row in years_return_df.iterrows():
        year_return[row['candle_begin_time']] = pct_str_to_float(row['涨跌幅']) / 100

    quarter_returns_df = pd.read_csv(path / "季度账户收益.csv", encoding="utf-8")
    quarter_return = dict()
    for _, row in quarter_returns_df.iterrows():
        quarter_return[row['candle_begin_time']] = pct_str_to_float(row['涨跌幅']) / 100
    month_returns_df = pd.read_csv(path / "月度账户收益.csv", encoding="utf-8")
    month_return = dict()
    for _, row in month_returns_df.iterrows():
        month_return[row['candle_begin_time']] = pct_str_to_float(row['涨跌幅']) / 100
    return SearchResultEntry(
        config_data,
        pct_str_to_float(all_return),
        drawdown,
        drawdown_hours,
        annual_drawdown_ratio,
        year_return,
        quarter_return,
        month_return,
    )


class SearchResult:
    def __init__(self, root: Path | None):
        if root is None:
            return
        sub_folders = [x for x in root.iterdir() if x.is_dir()]
        results: List[SearchResultEntry] = []
        cache_path = root / "returns.pkl"
        self._root = root
        if cache_path.exists():
            self.results = pd.read_pickle(cache_path)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.job_num) as executor:
                futures = [executor.submit(load_one_search_parameter, sub_folder)
                           for sub_folder in sub_folders]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    results.append(future.result())
            pd.to_pickle(results, cache_path)
            self.results = results
        month_returns = self.to_returns(time_type="month")
        month_returns.to_csv(root / "month_returns.csv", index=False, encoding="utf-8")
        quarter_returns = self.to_returns(time_type="quarter")
        quarter_returns.to_csv(root / "quarter_returns.csv", index=False, encoding="utf-8")
        year_returns = self.to_returns(time_type="year")
        year_returns.to_csv(root / "year_returns.csv", index=False, encoding="utf-8")

    def to_df(self, factors: List[str] = None, time_type: str = "year") -> pd.DataFrame:

        data = []
        times = self.get_times(time_type)
        if factors is None:
            factors = self.get_factors()

        columns = ["hold_period", "long_select_num", "short_select_num", "use_spot", "all", "drawdown",
                   "drawdown_hours",
                   "annual_drawdown_ratio"] + factors + times
        for r in self.results:
            row = [r.first_st_hold_period, r.first_st_long_select_num, r.first_st_short_select_num,
                   r.first_st_use_spot, r.returns, r.drawdown, r.drawdown_hours,
                   r.annual_drawdown_ratio]
            for f in factors:
                if f not in r.factors:
                    row.append(np.nan)
                else:
                    row.append(list(r.factors[f])[0])
            for t in times:
                if time_type == "quarter":
                    row.append(r.quarter_returns[t])
                elif time_type == "month":
                    row.append(r.month_returns[t])
                else:
                    row.append(r.year_returns[t])

            data.append(row)

        res = pd.DataFrame(data, columns=columns)
        return res

    def to_returns(self, time_type: str = "year") -> pd.DataFrame:
        times = self.get_times(time_type)
        res = self.to_df(time_type=time_type).copy()
        res['all_gap'] = res['all'].max() - res['all']
        gap_cols = []
        gap_cols_pct = []
        gap_cols_z = []
        for c in times:
            max_value = res[c].max()
            min_value = res[c].min()
            gap_col = f"{c}_gap"
            gap_col_pct = f"{c}_gap_pct"
            gap_col_z = f"{c}_gap_z"
            res[gap_col] = max_value - res[c]
            res[gap_col_pct] = 1 - (res[c] / max_value)
            res[gap_col_z] = 1 - (res[c] - min_value) / (max_value - min_value)
            gap_cols_pct.append(gap_col_pct)
            gap_cols.append(gap_col)
            gap_cols_z.append(gap_col_z)
        cal = res[gap_cols].agg(['mean', 'std'], axis=1)
        cal_pct = res[gap_cols_pct].agg(['mean', 'std'], axis=1)
        cal_pct = cal_pct.rename(columns={'mean': 'mean_pct', 'std': 'std_pct'})
        cal_z = res[gap_cols_z].agg(['mean', 'std'], axis=1)
        cal_z = cal_z.rename(columns={'mean': 'mean_z', 'std': 'std_z'})
        factors = self.get_factors()
        columns = ["hold_period", "long_select_num", "short_select_num", "use_spot", "all", "drawdown",
                   "drawdown_hours",
                   "annual_drawdown_ratio"] + ['mean_z', 'std_z', 'mean_pct', 'std_pct', 'mean',
                                               'std'] + factors + times
        res = pd.concat([res, cal, cal_pct, cal_z], axis=1)
        return res[columns]

    def get_times(self, time_type: str = "year") -> List[str]:
        """
        这里最好的是循环，但是为了快，就取第一个
        :return:
        """

        res = self.results[0]
        if time_type == "quarter":
            return list(res.quarter_returns.keys())
        elif time_type == "month":
            return list(res.month_returns.keys())
        else:
            return list(res.year_returns.keys())

    def get_factors(self) -> List[str]:
        res_cas = self._root / "最优参数.xlsx"
        res_df = pd.read_excel(res_cas)

        return [re.match(r'#FACTOR-(\w+)', item).group(1) for item in res_df.columns if re.match(r'#FACTOR-\w+', item)]


def load_may_fly_search_result(zip_ref: zipfile.ZipFile) -> tuple[
    dict, zipfile.ZipInfo | None, zipfile.ZipInfo | None, zipfile.ZipInfo | None]:
    """
    返回zip包内所有文件夹及其包含的文件信息。
    返回值: {folder_name: {filename: file_info, ...}, ...}
    """
    folder_dict = {}
    para_xlsx = None
    month_returns = None
    quarter_returns = None
    for file_info in zip_ref.infolist():
        filename = file_info.filename
        if '/' in filename and not filename.endswith('/'):
            folder, file = filename.split('/', 1)
            if folder not in folder_dict:
                folder_dict[folder] = {}
            folder_dict[folder][file] = file_info
        elif filename == "最优参数.xlsx":
            para_xlsx = file_info
        elif filename == "month_returns.csv":
            month_returns = file_info
        elif filename == "quarter_returns.csv":
            quarter_returns = file_info

    return folder_dict, para_xlsx, month_returns, quarter_returns


def load_one_may_fly_search_parameter(zip_ref, file_info_dic: dict[str, zipfile.ZipInfo]) -> SearchResultEntry:
    config_file = file_info_dic.get("config.pkl", None)
    with zip_ref.open(config_file.filename) as config:
        config_data = pd.read_pickle(config)

    assertment_file = file_info_dic.get("策略评价.csv", None)
    with zip_ref.open(assertment_file.filename) as assertment:
        stats_temp = pd.read_csv(assertment, encoding='utf-8')
        stats_temp.columns = ['evaluation_indicator', 'value']
        stats_temp = stats_temp.set_index('evaluation_indicator')
        all_return = stats_temp.loc["累积净值", 'value']
        drawdown = pct_str_to_float(stats_temp.loc["最大回撤", 'value']) / 100
        date_format = "%Y-%m-%d %H:%M:%S"
        max_drawdown_start = datetime.datetime.strptime(stats_temp.loc["最大回撤开始时间", 'value'], date_format)
        max_drawdown_end = datetime.datetime.strptime(stats_temp.loc["最大回撤结束时间", 'value'], date_format)
        drawdown_hours = (max_drawdown_end - max_drawdown_start).total_seconds() / 3600
        annual_drawdown_ratio = stats_temp.loc["最大回撤", 'value']

    years_return_file = file_info_dic.get("年度账户收益.csv", None)
    with zip_ref.open(years_return_file.filename) as years_return:
        years_return_df = pd.read_csv(years_return, encoding="utf-8")
        year_return = dict()
        for _, row in years_return_df.iterrows():
            year_return[row['candle_begin_time']] = pct_str_to_float(row['涨跌幅']) / 100

    quarter_returns_file = file_info_dic.get("季度账户收益.csv", None)
    with zip_ref.open(quarter_returns_file.filename) as quarter_return:
        quarter_returns_df = pd.read_csv(quarter_return, encoding="utf-8")
        quarter_return = dict()
        for _, row in quarter_returns_df.iterrows():
            quarter_return[row['candle_begin_time']] = pct_str_to_float(row['涨跌幅']) / 100

    month_returns_file = file_info_dic.get("月度账户收益.csv", None)
    with zip_ref.open(month_returns_file.filename) as month_returns_file:
        month_returns_df = pd.read_csv(month_returns_file, encoding="utf-8")
        month_return = dict()
        for _, row in month_returns_df.iterrows():
            month_return[row['candle_begin_time']] = pct_str_to_float(row['涨跌幅']) / 100

    return SearchResultEntry(
        config_data,
        pct_str_to_float(all_return),
        drawdown,
        drawdown_hours,
        annual_drawdown_ratio,
        year_return,
        quarter_return,
        month_return,
    )


class MayflySearchResult(SearchResult):
    """
    回测时候组织批量返回结果的方式
    """

    def __init__(self, zip_file: Path):
        super().__init__(None)

        def cal_results():
            results = []
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                file_list, param_xls, month_returns, quarter_returns = load_may_fly_search_result(zip_ref)
                # print(f"找到{len(file_list)}文件夹")
                with zip_ref.open(param_xls.filename) as xls:
                    param_xls_df = pd.read_excel(xls)
                with zip_ref.open(month_returns.filename) as csv:
                    month_returns_df = pd.read_csv(csv, encoding="utf-8-sig")
                with zip_ref.open(quarter_returns.filename) as csv:
                    quarter_returns = pd.read_csv(csv, encoding="utf-8-sig")
                for key, value in file_list.items():
                    results.append(load_one_may_fly_search_parameter(zip_ref, value))
            return results, param_xls_df, file_list, month_returns_df, quarter_returns

        result_df, param_xls, files, month_returns_df, quarter_returns_df = cache_data(cal_results, zip_file.stem)
        self.results = result_df
        self.param_xls = param_xls
        self.files = files
        self.month_returns_df = month_returns_df
        self.quarter_returns = quarter_returns_df

    def get_times(self, time_type: str = "year") -> List[str]:
        """
        这里最好的是循环，但是为了快，就取第一个
        :return:
        """

        res = self.results[0]
        if time_type == "quarter":
            return list(res.quarter_returns.keys())
        elif time_type == "month":
            return list(res.month_returns.keys())
        else:
            return list(res.year_returns.keys())

    def get_factors(self) -> List[str]:
        return [re.match(r'#FACTOR-(\w+)', item).group(1) for item in self.param_xls.columns if
                re.match(r'#FACTOR-\w+', item)]


class EquityCurve:
    _STG_HEADERS = ['stg_id', 'group', 'returns', 'distance_from_center',
                    'long_factor', 'long_factor_parameter', 'short_factor', 'short_factor_parameter',
                    'long_filter', 'long_filter_parameter', 'short_filter', 'short_filter_parameter',
                    'long_filter_post', 'long_filter_post_parameter', 'short_filter_post',
                    'short_filter_post_parameter']

    def __init__(self, root_folder: Path):
        self._root_folder = root_folder

    def get_stg_id_list(self, type: Literal['long', 'short', 'neutral']) -> List[str]:
        """
        按照数据，获得相应的策略
        :param type: long: 纯多。 short: 空， neutral：中性
        :return: 策略的id list
        """

        def filter_strategy(stg_config_path: Path) -> str | None:
            """

            :param stg_config: 文件地址
            :return: 空为过滤，否则返回id
            """
            bt_config = pd.read_pickle(stg_config_path)
            stg_config: StrategyConfig = bt_config.strategy_list[0]
            long_cap_weight = stg_config.long_cap_weight
            short_cap_weight = stg_config.short_cap_weight

            if type == "long":
                if short_cap_weight == 0 and long_cap_weight != 0:
                    return stg_config_path.stem
            elif type == "short":
                if long_cap_weight == 0 and short_cap_weight != 0:
                    return stg_config_path.stem
            elif type == "neutral":
                if short_cap_weight != 0 and long_cap_weight != 0:
                    return stg_config_path.stem
            return None

        bt_config_files = [file for file in self._root_folder.glob('*.pkl')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 使用executor.map函数来并行执行read_file函数
            futures = [executor.submit(filter_strategy, file_path) for file_path in bt_config_files]
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(bt_config_files)):
            stg = future.result()
            if stg is not None:
                results.append(stg)
        return results

    def compose_col_value(self, base_column: str = "equity") -> pd.DataFrame:
        bt_config_files = [file for file in self._root_folder.glob('*.pkl')]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 使用executor.map函数来并行执行read_file函数
            futures = {executor.submit(self._read_equity_curve, file_path, base_column): file_path for
                       file_path in bt_config_files}
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(bt_config_files)):
            file_path = futures[future]
            try:
                series = future.result()
                results.append(series)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        return pd.concat(results, axis=1)

    def amend_all_data(self, group_info: pd.DataFrame, distance_data: pd.DataFrame, center_map: dict) -> pd.DataFrame:
        """
        补全信息：
        1. 每个策略的收益
        2. 每个策略因子
        3. 每个每个测路和中心因子的距离

        :param center_map: 中心点的行号
        :param distance_data: 距离信息
        :param group_info: 组的信息，有两列一列为id，一列为组号
        :return: 补全的信息
        """
        res = group_info.copy()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 使用executor.map函数来并行执行read_file函数
            futures = {
                executor.submit(self.amend_one_row, row['group'], str(row['stg_name']), row, distance_data, center_map)
                for
                _, row in res.iterrows()}
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=res.shape[0]):
            one_stg_data = future.result()
            results.append(one_stg_data)
        return pd.DataFrame(results, columns=self._STG_HEADERS)

    def export_stg(self, stg_with_group: dict[str, str], export_folder: Path) -> pd.DataFrame:
        """
        补全信息：
        1. 每个策略的收益
        2. 每个策略因子
        :param stg_with_group: 策略的id和组，key是id，val是组
        :return: 补全的信息
        """
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 使用executor.map函数来并行执行read_file函数
            futures = [
                executor.submit(self.amend_one_row, group, stg_id, None, None, export_folder)
                for stg_id, group in stg_with_group.items()]
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            one_stg_data = future.result()
            results.append(one_stg_data)
        return pd.DataFrame(results, columns=self._STG_HEADERS)

    def amend_one_row(self, group_num, stg_id, distance_data: pd.DataFrame = None,
                      center_map: dict = None, export_folder: Path = None) -> List:
        # 在这里编写你的逻辑

        config_pkl = self._root_folder / f"{stg_id}.pkl"
        if export_folder is not None:
            shutil.copy(config_pkl, export_folder / f"{stg_id}.pkl")
        bt_config = pd.read_pickle(config_pkl)
        equity = pd.read_parquet(self._root_folder / f"{stg_id}.parquet")
        equity_col = equity['equity']
        return_val = (equity_col.iloc[-1] - equity_col.iloc[0]) / equity_col.iloc[0]
        if center_map is not None and distance_data is not None:
            center_id = center_map[group_num]
            distance_from_group_center = distance_data.at[center_id, stg_id]
        else:
            distance_from_group_center = 0
        stg_config = bt_config.strategy_list[0]
        long_factor, long_factor_parameter = parser_factor_list(stg_config.long_factor_list)
        short_factor, short_factor_parameter = parser_factor_list(stg_config.short_factor_list)
        long_filter, long_filter_parameter = parser_factor_list(stg_config.long_filter_list)
        short_filter, short_filter_parameter = parser_factor_list(stg_config.short_filter_list)
        long_filter_post, long_filter_post_parameter = parser_factor_list(stg_config.long_filter_list_post)
        short_filter_post, short_filter_post_parameter = parser_factor_list(stg_config.short_filter_list_post)

        return [stg_id, group_num, return_val, distance_from_group_center,
                long_factor, long_factor_parameter, short_factor, short_factor_parameter,
                long_filter, long_filter_parameter, short_filter, short_filter_parameter,
                long_filter_post, long_filter_post_parameter, short_filter_post, short_filter_post_parameter]

    def _read_equity_curve(self, config_file: Path, base_column: str) -> pd.Series:
        name = config_file.stem
        data_df = pd.read_parquet(self._root_folder / f"{name}.parquet", engine='pyarrow')
        data_df.set_index("candle_begin_time", drop=True, inplace=True)
        data_df = data_df.rename(columns={base_column: name})
        return data_df[name]


def parser_factor_list(factor_list: List, first_index: int = 0, second_index: int = 2):
    if len(factor_list) != 0:
        factor = factor_list[0]
        return factor.name, factor.param
    else:
        return "", ""


def first_factor_define(bt_config: BacktestConfig) -> str:
    stg_config = bt_config.strategy_list[0]
    long_factor = stg_config.long_factor_list
    short_factor = stg_config.short_factor_list
    if len(long_factor) == 0:
        factor = short_factor[0]
    else:
        factor = long_factor[0]
    return f"{factor.name}_{factor.param}"


def first_all_define(bt_config: BacktestConfig) -> str:
    factors = sorted(bt_config.factor_col_name_list)
    return "_".join(factors)
