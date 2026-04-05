import pandas as pd
import glob
import os
#example 
# python -c "import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000); print(pd.read_pickle(r'C:\Users\Public\0_QTclass\BBcode\select-coin-pro_v2.0.0_20260403\data\回测结果\测试策略1\选币结果.pkl').head(5))"
#按 candle_begin_time 降序排列
#python -c "import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000); df = pd.read_pickle(r'C:\Users\Public\0_QTclass\BBcode\select-coin-pro_v2.0.0_20260403\data\回测结果\测试策略1\选币结果.pkl'); print(df.sort_values('candle_begin_time', ascending=False).head(5))"

#python -c "import pandas as pd; data = pd.read_pickle(r'D:\quantclass-data\coin-binance-spot-swap-preprocess-pkl-1h-2026-04-02\swap_dict.pkl'); print({k: type(v) for k, v in data.items()})"
# ==========================================
# 📝 只需要在这里修改你的数据文件夹路径
# ==========================================
DATA_DIR = r'D:\quantclass-data\coin-binance-spot-swap-preprocess-pkl-1h-2026-04-02'  # 或者用 '.' 表示当前目录

def print_df_head_tail(df, key_name):
    """辅助函数：打印 DataFrame 的升序和降序前3行"""
    sort_col = 'candle_begin_time'
    
    if sort_col not in df.columns:
        print(f"  [Key: {key_name}] ⚠️ 未找到 {sort_col} 列，直接显示前3行：")
        print(df.head(3))
        return

    print(f"\n  --- 🔍 数据维度: {key_name} ---")
    
    # 1. 升序排列 (最旧的数据)
    print(f"  ⬆️ 【升序 - 最旧3行】(Order by {sort_col})")
    print(df.sort_values(sort_col, ascending=True).head(3))
    
    # 2. 降序排列 (最新的数据)
    print(f"  ⬇️ 【降序 - 最新3行】(Order by {sort_col})")
    print(df.sort_values(sort_col, ascending=False).head(3))
    
    print(f"  📏 形状: {df.shape}")

def main():
    # 设置 Pandas 显示选项
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.unicode.ambiguous_as_wide', True)
    pd.set_option('display.unicode.east_asian_width', True)

    # 获取所有 .pkl 文件
    search_path = os.path.join(DATA_DIR, "*.pkl")
    pkl_files = glob.glob(search_path)

    if not pkl_files:
        print(f"💡 在路径 [{DATA_DIR}] 下未找到 .pkl 文件。")
        return

    print(f"🚀 发现 {len(pkl_files)} 个文件，正在解析字典结构...\n")

    for path in pkl_files:
        file_name = os.path.basename(path)
        print(f"\n{'#'*60}")
        print(f"📂 文件: {file_name}")
        print(f"{'#'*60}")

        try:
            data = pd.read_pickle(path)
            
            # 情况 1：如果是字典 (OHLC 结构)
            if isinstance(data, dict):
                print(f"📦 检测到字典格式，包含数据项: {list(data.keys())}")
                for key, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        print_df_head_tail(df, key)
                    else:
                        print(f"  [Key: {key}] ⚠️ 不是 DataFrame 格式，类型为: {type(df)}")

            # 情况 2：如果是直接的 DataFrame
            elif isinstance(data, pd.DataFrame):
                print_df_head_tail(data, "Standard DataFrame")

            else:
                print(f"⚠️ 未知数据格式: {type(data)}")

        except Exception as e:
            print(f"❌ 读取文件 [{file_name}] 出错: {e}")

    print(f"\n{'='*60}")
    print("✅ 全部数据预览完成。")

if __name__ == "__main__":
    main()