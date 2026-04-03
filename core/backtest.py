"""
邢不行｜策略分享会
选币策略框架𝓟𝓻𝓸

版权所有 ©️ 邢不行
微信: xbx1717

本代码仅供个人学习使用，未经授权不得复制、修改或用于商业用途。

Author: 邢不行
"""

import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from config import job_num, raw_data_path
from core.equity import calc_equity, show_plot_performance
from core.figure import mat_heatmap
from core.model.backtest_config import BacktestConfig
from core.model.backtest_config import BacktestConfigFactory
from core.model.timing_signal import TimingSignal
from core.select_coin import calc_factors, select_coins, concat_select_results, process_select_results, \
    agg_multi_strategy_ratio, calc_cross_sections
from core.utils.functions import load_spot_and_swap_data, save_performance_df_csv
from core.utils.log_kit import logger, divider


def step2_load_data(conf: BacktestConfig):
    """
    读取回测所需数据，并做简单的预处理
    :param conf:
    :return:
    """
    divider('读取数据', with_timestamp=False, sep='-')
    s_time = time.time()

    # 读取数据
    # 针对现货策略和非现货策略读取的逻辑完全不同。
    # - 如果是纯合约模式，只需要读入 swap 数据并且合并即可
    # - 如果是现货模式，需要读入 spot 和 swap 数据并且合并，然后添加 tag
    load_spot_and_swap_data(conf)  # 串行方式，完全等价
    logger.ok(f'完成读取数据中心数据，花费时间：{time.time() - s_time:.2f}秒')


def step3_calc_factors(conf: BacktestConfig):
    """
    计算因子
    :param conf: 配置
    :return:
    """
    divider('时序因子计算', with_timestamp=False, sep='-')
    s_time = time.time()
    calc_factors(conf)
    logger.ok(f'完成计算时序因子，花费时间：{time.time() - s_time:.2f}秒')

    # 计算截面因子
    divider('截面因子计算', with_timestamp=False, sep='-')
    s_time = time.time()
    calc_cross_sections(conf)
    logger.ok(f'完成计算截面因子，花费时间：{time.time() - s_time:.2f}秒')


def step4_select_coins(conf: BacktestConfig):
    """
    选币
    :param conf: 配置
    :return:
    """
    divider('条件选币', with_timestamp=False, sep='-')
    s_time = time.time()
    select_coins(conf)  # 选币
    logger.ok(f'完成选币，花费时间：{time.time() - s_time:.3f}秒')


def step5_aggregate_select_results(conf: BacktestConfig, save_final_result=False):
    logger.info(f'整理{conf.name}选币结果...')
    # 整理选币结果
    concat_select_results(conf)  # 合并多个策略的选币结果
    select_results = process_select_results(conf)  # 生成整理后的选币结果
    logger.debug(
        f'💾 选币结果df大小：{select_results.memory_usage(deep=True).sum() / 1024 / 1024:.4f} M\n'
        f'ℹ️ 选币结果：\n{select_results}', )
    if save_final_result:
        # 存储最终的选币结果
        select_results.to_pickle(conf.get_result_folder() / 'final_select_results.pkl')
    logger.ok(f'完成{conf.name}结果整理.')

    # 聚合大杂烩中多策略的权重，以及多offset选币的权重聚合
    s_time = time.time()
    logger.debug(f'🔃 开始{conf.name}权重聚合...')
    df_spot_ratio, df_swap_ratio = agg_multi_strategy_ratio(conf, select_results)
    logger.ok(f'完成权重聚合，花费时间： {time.time() - s_time:.3f}秒')

    return df_spot_ratio, df_swap_ratio


# ====================================================================================================
# 模拟交易
# 1. 计算初始资金曲线
# 2. 保存回测结果，包括资金曲线和各项收益评价指标
# 3. 可选：显示资金曲线图表
# 4. 如有择时信号，执行再择时模拟并保存结果
# ====================================================================================================
def step6_simulate_performance(conf: BacktestConfig, df_spot_ratio, df_swap_ratio, pivot_dict_spot, pivot_dict_swap,
                               if_show_plot=False):
    """
    模拟交易，计算资金曲线和收益指标
    :param conf: 回测配置
    :param df_spot_ratio: 现货目标资金占比
    :param df_swap_ratio: 永续合约目标资金占比
    :param pivot_dict_spot: 现货行情
    :param pivot_dict_swap: 永续合约行情
    :param if_show_plot:    是否显示资金曲线
    :return: 资金曲线、策略收益、年度收益、季度收益和月度收益
    """
    divider('回测表现', with_timestamp=False, sep='-')
    # 记录回测开始的信息，计算回测的总时间范围（小时和天数）
    logger.info(f'开始模拟交易，累计回溯 {len(df_spot_ratio):,} 小时（~{len(df_spot_ratio) / 24:,.0f}天）...')

    # 1. 计算初始资金曲线
    # - 通过现货和合约的持仓比例数据，计算回测期间的资金曲线和收益指标
    account_df, rtn, year_return, month_return, quarter_return = calc_equity(
        conf, pivot_dict_spot, pivot_dict_swap,
        df_spot_ratio, df_swap_ratio
    )

    # 2. 保存初始回测结果
    # - 保存计算出的资金曲线、策略评价、年度、季度和月度的收益数据
    save_performance_df_csv(
        conf,
        资金曲线=account_df,
        策略评价=rtn,
        年度账户收益=year_return,
        季度账户收益=quarter_return,
        月度账户收益=month_return
    )

    # 检查配置中是否启用了择时信号
    has_timing_signal = isinstance(conf.timing, TimingSignal)

    # 3. 可选：绘制初始回测的资金曲线图表
    if if_show_plot:
        # 绘制资金曲线并显示各项收益指标
        show_plot_performance(conf, account_df, rtn, year_return, title_prefix='初始-')

    # 4. 如果配置中有择时信号，执行动态杠杆再择时模拟
    if has_timing_signal:
        # 进行再择时回测，计算动态杠杆后的资金曲线和收益指标
        account_df2, rtn2, year_return2 = simu_re_timing(
            conf, df_spot_ratio, df_swap_ratio, pivot_dict_spot, pivot_dict_swap
        )

        # 可选：绘制再择时的资金曲线图表
        if if_show_plot:
            # 绘制再择时后的资金曲线并显示各项收益指标
            show_plot_performance(
                conf,
                account_df2,
                rtn2,
                year_return2,
                title_prefix='再择时-',
                再择时前资金曲线=account_df['净值']
            )

    # 返回最终的回测报告，用于进一步分析或评估
    return conf.report


# ====================================================================================================
# 动态杠杆再择时模拟
# 1. 生成动态杠杆
# 2. 进行动态杠杆再择时的回测模拟
# 3. 保存结果
# ====================================================================================================
def simu_re_timing(conf: BacktestConfig, df_spot_ratio, df_swap_ratio, pivot_dict_spot, pivot_dict_swap):
    """
    动态杠杆再择时模拟
    :param conf: 回测配置
    :param df_spot_ratio: 现货目标资金占比
    :param df_swap_ratio: 永续合约目标资金占比
    :param pivot_dict_spot: 现货行情
    :param pivot_dict_swap: 永续合约行情
    :return: 资金曲线，策略收益，年化收益
    """
    divider(f'{conf.get_fullname(as_folder_name=True)} 资金曲线择时，生成动态杠杆', sep='-', with_timestamp=False)
    logger.warning(f'注意：因子计算和再择时是针对1H的资金曲线进行的。')
    time.sleep(1)

    # 记录开始时间，用于计算耗时
    s_time = time.time()

    # 读取资金曲线数据，作为动态杠杆计算的基础
    account_df = pd.read_csv(conf.get_result_folder() / '资金曲线.csv', index_col=0, encoding='utf-8-sig')

    # 生成动态杠杆，根据资金曲线的权益变化进行杠杆调整
    # 检查哪个动态杠杆方法有实现，优先使用 get_dynamic_leverage_for_dataframe
    if conf.timing.impl_flags.get('dynamic_leverage_for_dataframe', False):
        leverage = conf.timing.get_dynamic_leverage_for_dataframe(account_df)
    elif conf.timing.impl_flags.get('dynamic_leverage', False):
        leverage = conf.timing.get_dynamic_leverage(account_df['equity'])
    else:
        raise NotImplementedError(f'择时信号 {conf.timing.name} 必须实现 dynamic_leverage 或 dynamic_leverage_for_dataframe 方法之一')
    logger.ok(f'完成生成动态杠杆，花费时间： {time.time() - s_time:.3f}秒')

    # 记录时间，用于后续动态杠杆再择时的耗时统计
    s_time = time.time()
    logger.info(
        f'开始动态杠杆再择时模拟交易，累计回溯 {len(df_spot_ratio):,} 小时（~{len(df_spot_ratio) / 24:,.0f}天）...')

    # 进行资金曲线的再择时回测模拟
    # - 使用动态杠杆调整后的持仓计算资金曲线
    # - 包括现货和合约的比例数据
    # - 计算回测的总体收益、年度收益、季度收益和月度收益
    account_df, rtn, year_return, month_return, quarter_return = calc_equity(
        conf,
        pivot_dict_spot,
        pivot_dict_swap,
        df_spot_ratio,
        df_swap_ratio,
        leverage
    )

    # 保存回测结果，包括再择时后的资金曲线和收益评价指标
    save_performance_df_csv(
        conf,
        资金曲线_再择时=account_df,
        策略评价_再择时=rtn,
        年度账户收益_再择时=year_return,
        季度账户收益_再择时=quarter_return,
        月度账户收益_再择时=month_return
    )

    logger.ok(f'完成动态杠杆再择时模拟交易，花费时间：{time.time() - s_time:.3f}秒')

    # 返回再择时后的资金曲线和收益结果，用于后续分析或评估
    return account_df, rtn, year_return


def simu_performance_on_select(conf: BacktestConfig):
    import logging
    logger.setLevel(logging.WARNING)  # 可以减少中间输出的log
    logger.debug(conf.get_fullname())
    # ====================================================================================================
    # 5. 整理大杂烩选币结果
    # - 把大杂烩中每一个策略的选币结果聚合成一个df
    # ====================================================================================================
    df_spot_ratio, df_swap_ratio = step5_aggregate_select_results(conf)

    pivot_dict_spot = pd.read_pickle(raw_data_path / 'market_pivot_spot.pkl')
    pivot_dict_swap = pd.read_pickle(raw_data_path / 'market_pivot_swap.pkl')
    return step6_simulate_performance(conf, df_spot_ratio, df_swap_ratio, pivot_dict_spot, pivot_dict_swap)


# ====================================================================================================
# ** 回测主程序 **
# 1. 准备工作
# 2. 读取数据
# 3. 计算因子
# 4. 选币
# 5. 整理选币数据
# 6. 添加下一个每一个周期需要卖出的币的信息
# 7. 计算资金曲线
# ====================================================================================================
def run_backtest(conf: BacktestConfig):
    r_time = time.time()  # 记录当前时间戳，用于后续计算耗时

    # ====================================================================================================
    # 1. 准备工作
    # - 包括删除旧缓存、记录时间戳、保存当前配置等，确保环境的清洁和配置的正确性
    # ====================================================================================================
    conf.delete_cache()  # 删除可能存在的旧缓存数据，避免对本次回测产生影响
    conf.save()  # 将当前的回测配置保存到磁盘，便于后续重现相同的配置

    # ====================================================================================================
    # 2. 读取回测所需数据，并做简单的预处理
    # - 加载历史行情数据或其他相关数据
    # ====================================================================================================
    step2_load_data(conf)  # 读取并预处理回测数据

    # ====================================================================================================
    # 3. 计算因子
    # - 根据回测策略的要求，计算出选币所需的各种因子数据，如技术指标、基本面数据等
    # ====================================================================================================
    step3_calc_factors(conf)  # 计算用于选币的因子，可以包括技术指标

    # ====================================================================================================
    # 4. 选币
    # - 根据计算的因子进行选币，筛选出符合条件的加密货币
    # - 选币结果将被保存到硬盘，以便后续分析或复现
    # ====================================================================================================
    step4_select_coins(conf)  # 执行选币操作，将结果保存以供后续使用

    # ====================================================================================================
    # 5. 整理选币结果并形成目标持仓
    # - 对选币结果进行汇总，生成买入或卖出的目标持仓比例
    # ====================================================================================================
    df_spot_ratio, df_swap_ratio = step5_aggregate_select_results(conf, save_final_result=True)  # 汇总选币结果，形成现货和合约的目标持仓
    # logger.ok(f'目标持仓信号已完成，花费时间：{(time.time() - r_time):.3f}秒')  # 记录目标持仓计算完成的时间

    # ====================================================================================================
    # 6. 根据目标持仓计算资金曲线
    # - 使用目标持仓对资金进行模拟回测，生成资金曲线
    # - 回测过程中可以选择是否显示绘图
    # ====================================================================================================
    pivot_dict_spot = pd.read_pickle(raw_data_path / 'market_pivot_spot.pkl')  # 读取现货市场的枢纽点数据
    pivot_dict_swap = pd.read_pickle(raw_data_path / 'market_pivot_swap.pkl')  # 读取合约市场的枢纽点数据
    step6_simulate_performance(
        conf,
        df_spot_ratio,
        df_swap_ratio,
        pivot_dict_spot,
        pivot_dict_swap,
        if_show_plot=True
    )  # 根据目标持仓计算资金曲线，并显示图表（可选）
    logger.info(f'回测结果存储在: {conf.get_result_folder()}')

    logger.ok(f'完成回测，累计消耗时间：{time.time() - r_time:.3f}秒')  # 记录整个回测过程的总耗时


# ====================================================================================================
# ** 寻找最优参数 **
# 1. 准备工作：清理环境，初始化参数配置
# 2. 读取数据：加载回测所需的行情数据
# 3. 计算因子：根据策略要求计算选币因子
# 4. 选币和计算资金曲线：并行回测选币结果并生成资金曲线
# 5. 展示最优参数：从回测结果中筛选出最优参数组合
# ====================================================================================================
def find_best_params(factory: BacktestConfigFactory, show_heat_map=False):
    # ====================================================================================================
    # 1. 准备工作
    # - 初始化输出文件夹，清理旧数据，准备新的参数遍历
    # ====================================================================================================
    divider('参数遍历开始', '*')  # 输出分隔符，表示参数遍历的开始
    iter_results_folder = factory.result_folder

    shutil.rmtree(iter_results_folder, ignore_errors=True)  # 删除可能存在的旧结果文件夹，避免干扰
    time.sleep(0.2)  # 给文件系统一点时间来完成文件删除操作

    iter_results_folder.mkdir(parents=True, exist_ok=True)  # 创建输出文件夹，用于保存回测结果

    conf_list = factory.config_list
    for index, conf in enumerate(conf_list):
        logger.debug(f'ℹ️ 参数组合{index + 1}｜共{len(conf_list)}')  # 记录当前参数组合的索引
        logger.debug(f'{conf.get_fullname()}')  # 输出参数组合的详细名称
        conf.save()  # 保存当前配置到磁盘，便于后续使用
        print()
    logger.ok(f'一共需要回测的参数组合数：{len(conf_list)}')  # 输出需要遍历的参数组合总数

    r_time = time.time()  # 记录开始时间，用于计算回测总耗时

    # ====================================================================================================
    # 2. 读取回测所需数据，并做简单的预处理
    # - 根据策略的要求，加载现货和/或合约的历史行情数据
    # ====================================================================================================
    logger.info(f'读取数据...')
    s_time = time.time()
    conf = factory.generate_all_factor_config()

    # 读取数据，区分现货和合约策略的不同处理方式
    # - 纯合约模式：读取合约数据
    # - 现货模式：读取现货和合约数据并合并，处理标签
    load_spot_and_swap_data(conf)  # 数据读取和预处理
    logger.ok(f'完成读取数据中心数据，花费时间：{time.time() - s_time:.3f}秒')

    # ====================================================================================================
    # 3. 计算因子
    # - 基于策略需要计算选币所用的因子，如技术指标或量化信号
    # ====================================================================================================
    s_time = time.time()
    logger.info(f'时序因子计算...')
    calc_factors(conf)  # 执行时序因子计算
    logger.ok(f'完成计算时序因子，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    # 计算截面因子
    s_time = time.time()
    logger.info(f'截面因子计算...')
    calc_cross_sections(conf)  # 执行截面因子计算
    logger.ok(f'完成计算截面因子，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    # ====================================================================================================
    # 4. 选币
    # - 并行执行选币操作，并将结果保存到硬盘
    # ====================================================================================================
    s_time = time.time()
    logger.info(f'并行选币（时间最久，耐心等等）...')
    logger.debug('⚠️ 如果崩掉，修改 select_coins 函数参数 multi_process=False ，让他串行。')
    select_coins(factory.config_list, multi_process=True)  # 并行执行选币
    logger.ok(f'完成选币，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    # ====================================================================================================
    # 5. 回测模拟
    # - 聚合选币结果，并对每个参数组合进行回测模拟，生成资金曲线
    # ====================================================================================================
    logger.info(f'并行回测模拟（时间也会比较久）...')
    logger.debug(f'并行任务数：{job_num}。如果崩溃也可以把下方代码改成串行')  # 记录并行任务数量
    s_time = time.time()
    report_list = []

    # 并行回测每个参数组合
    with ProcessPoolExecutor(max_workers=job_num) as executor:
        futures = [executor.submit(simu_performance_on_select, conf) for conf in conf_list]
        for future in tqdm(as_completed(futures), total=len(conf_list), desc='回测进度'):
            try:
                report = future.result()
                report_list.append(report)
                if len(report_list) > 65535:
                    logger.debug(f'回测报表数量为 {len(report_list)}，超过 65535，后续可能会占用大量内存')
            except Exception as e:
                logger.exception(e)
                exit(1)
    logger.ok(f'回测模拟完成，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    # ====================================================================================================
    # 6. 展示最优参数
    # - 根据回测结果筛选最优参数组合，并保存到 Excel 文件
    # ====================================================================================================
    s_time = time.time()
    logger.info(f'展示最优参数...')
    if len(report_list) > 65535:
        logger.warning(f'回测参数列表超过 65535，会占用大量内存，请手动合并结果')
        return None

    all_params_map = pd.concat(report_list, ignore_index=True)
    report_columns = all_params_map.columns

    # 合并参数细节
    sheet = factory.get_name_params_sheet()
    if len(sheet.columns) > 1023:
        logger.warning(f'回测参数列表超过 1023，结果不包含因子列')
    else:
        all_params_map = all_params_map.merge(sheet, left_on='param', right_on='fullname', how='left')

    # 按累积净值排序并保存结果
    all_params_map.sort_values(by='累积净值', ascending=False, inplace=True)
    all_params_map = all_params_map[[*sheet.columns, *report_columns]].drop(columns=['param'])
    all_params_map.to_excel(iter_results_folder / f'最优参数.xlsx', index=False)
    print(all_params_map)
    logger.ok(f'完成展示最优参数，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    # ====================================================================================================
    # 7. 展示热力图（仅支持双因子）
    # ====================================================================================================
    s_time = time.time()
    if not show_heat_map or len(list(filter(lambda x: x.startswith('#FACTOR-'), sheet.columns))) > 2:
        return all_params_map

    # 热力图绘制设置
    indicator = '累积净值'

    series_col = []
    for col in list(sheet.columns) + [indicator]:
        if not col.startswith('#FACTOR-'):
            continue
        all_params_map[col] = all_params_map[col].astype(float)
        series_col.append(col)
    series_col = sorted(series_col)  # 甲一十七老板指出，避免出现排序混乱

    # 创建热力图的透视表
    pivot_table = pd.pivot_table(all_params_map,
                                 values=indicator,
                                 index=all_params_map[series_col[0]],
                                 columns=all_params_map[series_col[1]])
    pivot_table = pivot_table.astype(float)
    mat_heatmap(pivot_table, f'热力图 {indicator}')

    logger.info(f'完成展示双因子热力图，花费时间：{time.time() - s_time:.3f}秒，累计时间：{(time.time() - r_time):.3f}秒')

    return all_params_map
