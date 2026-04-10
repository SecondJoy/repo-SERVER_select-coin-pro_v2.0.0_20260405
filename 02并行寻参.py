#JoyAddded CTA币池因子趋势强度评估_20260329直播
import sys
import os
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import copy
import traceback
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
#               用户配置区域
# ==========================================
class UserConfig:
    """
    用户可在此处统一配置回测参数
    """
    # --- 1. 回测参数 ---
    # 观察数列表 (n): 决定因子的计算周期参数
    N_LIST = [72, 120, 192, 312, 504, 816]
    # N_LIST = [72] # 用于快速测试

    # 因子列表配置
    # 若为空列表 []，则默认读取 factors 目录下的所有因子。
    # 若不为空，则仅使用列表中的因子进行回测。例如: ['Factor_VMRatio', 'Factor_NetValueRatio']
    FACTOR_LIST = ["BollingWidth"]
    
    # 回测时间范围
    START_DATE = '2024-01-01 00:00:00'
    #END_DATE = '2026-01-01 00:00:00'
    END_DATE = datetime.now().strftime('%Y-%m-%d %H:00:00')  # 回测结束时间  #JoyUpdated
    
    # --- 2. 选币配置 ---
    # 选币数量
    SELECT_COIN_NUM = 20
    # 选币方向: 'long' (只做多), 'short' (只做空), 'both' (多空都做)
    SELECT_DIRECTION = 'long'
    
    # --- 3. 并行执行配置 ---
    # 并行进程数 (根据CPU核心数调整)
    MAX_WORKERS = 2
    
    # --- 4. 结果标识 ---
    # 迭代轮次名称 (用于区分不同的回测批次)
    ITER_ROUND = "batch_test"

# ==========================================

def process_single_task(factor_name, n, eval_config_dict, select_config=None):
    """
    Worker function to run backtest for a single factor and parameter n.
    """
    pid = os.getpid()
    print(f"[PID:{pid}] Starting task: Factor={factor_name}, n={n}")
    
    try:
        # 1. Modify config.job_num to 1 to avoid nested parallelism explosion
        import config
        config.job_num = 1
        
        # 2. Import modules AFTER modifying config
        from core.model.backtest_config import BacktestConfig
        from cta_core.backtest import step3_calc_factors, step4_select_coins, step5_aggregate_select_results
        from cta_core.evaluate_picks import compute_forward_eff_amp
        
        # Explicitly disable parallelism in select_coin to prevent nested process pools
        import config
        config.job_num = 1
        
        # --- A. Construct Config ---
        # Initialize config
        conf = BacktestConfig.init_from_config(load_strategy_list=False)
        
        # Apply Time Range Config
        conf.start_date = UserConfig.START_DATE
        conf.end_date = UserConfig.END_DATE
        
        # Determine selection params
        if select_config is None:
            # Fallback to UserConfig if not provided
            select_config = {
                'num': UserConfig.SELECT_COIN_NUM, 
                'direction': UserConfig.SELECT_DIRECTION
            }
            
        sel_num = select_config.get('num', 20)
        direction = select_config.get('direction', 'both')
        
        long_num = 0
        short_num = 0
        long_weight = 0
        short_weight = 0
        
        if direction in ['long', 'both']:
            long_num = sel_num
            long_weight = 1
            
        if direction in ['short', 'both']:
            short_num = sel_num
            short_weight = 1
        
        # Create new strategy list manually to avoid dependency on config.py's strategy_list
        # User requirement: Select coins based on config, no filters, factor list contains only the target factor.
        stg_config = {
            "strategy": f"Strategy_{factor_name}_{n}",
            "offset_list": [0],  # Default to offset 0 for efficiency
            "hold_period": "24H", # Default hold period
            "is_use_spot": False, # Default to Swap only (matches typical f1001 setup)
            "market": "swap_swap", # Default market type
            "cap_weight": 1,
            "long_cap_weight": long_weight,
            "short_cap_weight": short_weight,
            "long_select_coin_num": long_num, 
            "short_select_coin_num": short_num, 
            
            # Factor List: (FactorName, Ascending?, Parameter, Weight)
            # Defaulting Ascending=True (Smaller is better) as in previous script version.
            # If factor direction is unknown, this is just an assumption.
            "factor_list": [
                (factor_name, True, n, 1.0) 
            ],
            
            # Empty Filter Lists
            "filter_list": [],
            "long_filter_list": [],
            "short_filter_list": [],
            "filter_list_post": [],
            "long_filter_list_post": [],
            "short_filter_list_post": [],
            
            "use_custom_func": False,
        }
        
        new_strategy_list = [stg_config]
        
        # Load strategy
        conf.load_strategy_config(new_strategy_list, getattr(config, 're_timing', None))
        
        # Set unique name to avoid cache collision between processes
        conf.name = f"{factor_name}_{n}"
        conf.iter_round = UserConfig.ITER_ROUND 
        
        # Clean cache
        conf.delete_cache()
        conf.save()
        
        # --- B. Execute Backtest ---
        # 1. Calc factors
        step3_calc_factors(conf)
        
        # 2. Select coins
        step4_select_coins(conf)
        
        # 3. Aggregate results
        step5_aggregate_select_results(conf, save_final_result=True)
        
        # --- C. Evaluation ---
        result_folder = conf.get_result_folder()
        picks_path = result_folder / 'final_select_results.pkl'
        
        if not picks_path.exists():
            return {
                'factor': factor_name,
                'n': n,
                'bars': 0,
                'count': 0,
                'mean': None,
                'median': None,
                'status': 'no_result'
            }
        
        # Read picks
        picks_df = pd.read_pickle(picks_path)
        if 'candle_begin_time' in picks_df.columns:
            picks_df['date'] = pd.to_datetime(picks_df['candle_begin_time']).dt.date
        elif 'date' in picks_df.columns:
            picks_df['date'] = pd.to_datetime(picks_df['date']).dt.date
        
        picks_df['symbol'] = picks_df['symbol'].astype(str)
        picks_unique = picks_df[['date', 'symbol']].drop_duplicates()
        
        # Eval params
        eval_bars = eval_config_dict.get('bars', 72)
        eval_period = eval_config_dict.get('kline_period', '1h')

        #JoyTestStart
        # JoyChange 20260409 原因：评估阶段兼容kline_root为swap_dict.pkl（dict: symbol->DataFrame），避免误当目录导致Not a directory
        kroot = eval_config_dict.get('kline_root')
        eval_paths = {}
        if isinstance(kroot, str) and kroot.lower().endswith('.pkl') and os.path.isfile(kroot):
            try:
                _kdata = pd.read_pickle(kroot)
                if isinstance(_kdata, dict):
                    eval_paths['kline_dict'] = _kdata
                    eval_paths['kline_parquet'] = None
                else:
                    eval_paths['kline_parquet'] = kroot
            except Exception:
                eval_paths['kline_parquet'] = kroot
        else:
            eval_paths['kline_parquet'] = kroot
        #JoyTestEnd

        fr = compute_forward_eff_amp(
            picks_unique, 
            eval_paths, 
            kline_period=eval_period, 
            bars=eval_bars,
            segment_start_mode=eval_config_dict.get('segment_start_mode', 'next-day'),
            penalty_coef_dd=eval_config_dict.get('penalty_dd', 0.5),
            penalty_coef_zz=eval_config_dict.get('penalty_zz', 0.5),
            amp_mode=eval_config_dict.get('amp_mode', 'pct'),
            amp_beta=eval_config_dict.get('amp_beta', 1.0)
        )
        
        res_row = {}
        if not fr.empty and 'eff_amp_logms_er_smooth_penal' in fr.columns:
            overall = pd.to_numeric(fr['eff_amp_logms_er_smooth_penal'], errors='coerce')
            res_row = {
                'factor': factor_name,
                'n': n,
                'bars': eval_bars,
                'count': overall.count(),
                'mean': overall.mean(),
                'median': overall.median(),
                'status': 'success'
            }
        else:
            res_row = {
                'factor': factor_name,
                'n': n,
                'bars': eval_bars,
                'count': 0,
                'mean': None,
                'median': None,
                'status': 'empty_eval'
            }
            
        # --- Cleanup ---
        try:
            if picks_path.exists():
                os.remove(picks_path)
            
            # 清理缓存目录
            result_folder = picks_path.parent
            import shutil
            if result_folder.exists():
                shutil.rmtree(result_folder)
            
            parent_folder = result_folder.parent
            if parent_folder.exists():
                shutil.rmtree(parent_folder)

            eval_out_dir = Path(eval_config_dict['out_dir'])
            if eval_out_dir.exists():
                shutil.rmtree(eval_out_dir)
        except Exception:
            pass
            
        print(f"[PID:{pid}] Finished task: Factor={factor_name}, n={n}, Status={res_row['status']}")
        return res_row

    except Exception as e:
        print(f"[PID:{pid}] Error in task {factor_name}_{n}: {e}")
        # traceback.print_exc()
        return {
            'factor': factor_name,
            'n': n,
            'status': f'error: {str(e)}'
        }

def run_batch_parallel():
    # 1. Params
    # Use config from UserConfig
    n_list = UserConfig.N_LIST
    
    # === 选币配置 ===
    select_config = {
        'num': UserConfig.SELECT_COIN_NUM,
        'direction': UserConfig.SELECT_DIRECTION
    }
    
    factors_dir = PROJECT_ROOT / 'factors'
    
    # Get all factor names
    if UserConfig.FACTOR_LIST:
        factor_names = UserConfig.FACTOR_LIST
        print(f"使用配置的因子列表: {factor_names}")
    else:
        factor_files = [f for f in factors_dir.glob('*.py') if not f.name.startswith('__')]
        factor_names = [f.stem for f in factor_files]
        factor_names.sort()
        print(f"扫描 factors 目录，找到 {len(factor_names)} 个因子")
    
    print(f"参数列表: {n_list}")
    print(f"时间范围: {UserConfig.START_DATE} ~ {UserConfig.END_DATE}")
    print(f"选币配置: {select_config}")
    
    # 2. Configs
    # Import config just to trigger necessary initializations if any, but we don't use its strategy_list
    import config
    # We DO NOT read config.strategy_list anymore
    
    # Get Eval Config
    from cta_core.eval_config import CONFIG as EVAL_CONFIG
    eval_config_dict = dict(EVAL_CONFIG)
    
    # --- Ensure Data is Loaded ---
    from core.utils.path_kit import get_file_path
    from cta_core.backtest import step2_load_data
    from core.model.backtest_config import BacktestConfig
    
    data_cache_file = get_file_path('data', 'cache', 'all_candle_df_list.pkl', as_path_type=True)
    if not data_cache_file.exists():
        print("Data cache not found. Loading data...")
        # Init a temp config to load data
        temp_conf = BacktestConfig.init_from_config(load_strategy_list=False)
        
        # We need to set scope manually since we don't have a reference strategy
        # Default to Swap and Mix for general coverage
        temp_conf.select_scope_set.add('swap')
        
        step2_load_data(temp_conf)
        print("Data loaded successfully.")
    # -----------------------------
    
    output_file = PROJECT_ROOT / 'batch_backtest_results_parallel.csv'
    
    # Initialize CSV header if file doesn't exist
    if not output_file.exists():
        pd.DataFrame(columns=['factor', 'n', 'bars', 'count', 'mean', 'median', 'status']).to_csv(output_file, index=False)
    
    tasks = []
    
    # 3. Parallel Execution
    max_workers = UserConfig.MAX_WORKERS
    
    print(f"启动并行回测，进程数: {max_workers}")
    print(f"结果将实时追加至: {output_file}")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        for factor_name in factor_names:
            for n in n_list:
                # Note: removed original_strategy_list argument, added select_config
                tasks.append(executor.submit(process_single_task, factor_name, n, eval_config_dict, select_config))
        
        # Progress bar
        pbar = tqdm(total=len(tasks), desc="并行回测进度")
        
        # Collect results as they complete
        for future in as_completed(tasks):
            res = future.result()
            
            # Append to CSV immediately
            df_row = pd.DataFrame([res])
            df_row.to_csv(output_file, mode='a', header=False, index=False)
            
            pbar.update(1)
            
        pbar.close()

if __name__ == '__main__':
    # Ensure this script is run as main
    run_batch_parallel()
